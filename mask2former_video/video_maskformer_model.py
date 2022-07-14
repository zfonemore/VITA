# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from .modeling.criterion import VideoSetCriterion
from .modeling.matcher import VideoHungarianMatcher
from .utils.memory import retry_if_cuda_oom


from .modeling.transformer_decoder.swin import BasicLayer
from .modeling.transformer_decoder.transformer_detr import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from .modeling.transformer_decoder.utils import MLP

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class VideoMaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames

        hidden_dim = 256
        nheads = 8
        dropout = 0.0
        activation = 'relu'
        normalize_before = True
        dim_feedforward = 2048

        # video-level query
        # learnable query features
        self.video_query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.video_query_embed = nn.Embedding(num_queries, hidden_dim)

        # linear layer change query into token
        self.token_projector = nn.Linear(hidden_dim, hidden_dim)

        # object_encoder use BasicLayer from swin transformer in swin.py
        encoder_layer = TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.object_encoder = TransformerEncoder(encoder_layer, 3, encoder_norm)
        '''
        for i in range(3):
            self.object_encoder.append(
                BasicLayer(hidden_dim, 2,
                    self.num_heads, window_size=6)
            )
        '''

        # video decoder
        # change self-attention and cross attention in this layer in transformer.py
        decoder_layer = TransformerDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(hidden_dim)

        num_decoder_layers = 6
        self.obj_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=True)
        N_steps = hidden_dim // 2
        self.obj_pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        mask_dim = 256
        self.video_mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.video_decoder_norm = nn.LayerNorm(hidden_dim)
        self.video_class_embed = nn.Linear(hidden_dim, self.sem_seg_head.num_classes + 1)


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def video_forward_prediction_heads(self, output, mask_features, attn_mask_target_size=None):
        decoder_output = self.video_decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.video_class_embed(decoder_output)
        mask_embed = self.video_mask_embed(decoder_output)

        outputs_mask = torch.einsum("bqc,btchw->bqthw", mask_embed, mask_features)

        '''
        b, q, t, _, _ = outputs_mask.shape
        # NOTE: prediction is of higher-resolution
        # [B, Q, T, H, W] -> [B, Q, T*H*W] -> [B, h, Q, T*H*W] -> [B*h, Q, T*HW]
        attn_mask = F.interpolate(outputs_mask.flatten(0, 1), size=attn_mask_target_size, mode="bilinear", align_corners=False).view(
            b, q, t, attn_mask_target_size[0], attn_mask_target_size[1])
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        '''

        return outputs_class, outputs_mask #, attn_mask, decoder_output


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


    def video_process(self, frame_querys, mask_features):
        frame_querys = torch.cat(frame_querys)
        mask_features = torch.cat(mask_features, dim=1)
        obj_tokens = self.token_projector(frame_querys)
        T, q, bs = self.num_frames, self.num_queries, obj_tokens.size(1)
        pos_obj = self.obj_pe_layer(torch.permute(obj_tokens.view(T, 3*q, bs, -1), (2, 3, 0, 1))).flatten(2).permute(2, 0, 1)
        obj_tokens = self.object_encoder(obj_tokens, pos=pos_obj)

        query_embed = self.video_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.video_query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        output = self.obj_decoder(output, obj_tokens,
                    tgt_mask=None, memory_mask=None,
                    pos=pos_obj, query_pos=query_embed)
        outputs_class, outputs_mask = self.video_forward_prediction_heads(output[-1], mask_features, attn_mask_target_size=None)

        video_out = {
            'pred_logits': outputs_class,
            'pred_masks': outputs_mask,
            # here I do not use aux loss, you can use aux loss to compute loss for intermediate feature from obj_decoder
            # to get intermediate feature from obj_decoer, set return_intermediate to True in TransformerDecoder
            #self._set_aux_loss(
            #    outputs_class if self.mask_classification else None, outputs_mask
            #)
        }
        return video_out


    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []
        batch_size= len(batched_inputs)
        for video in batched_inputs:
            self.num_frames = len(video["image"])
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)

        clip_len, is_last  = 1, False
        if self.training:
            # mask classification target
            targets = self.prepare_targets_clip(batched_inputs, images, clip_len)

            losses_all = {}
            frame_querys, mask_features = [], []
            for frame in range(0, self.num_frames, clip_len):
                if frame+clip_len < self.num_frames:
                    indices = slice(frame, frame+self.num_frames*(batch_size-1)+1, self.num_frames)
                else:
                    is_last = True
                    frame = max(0, self.num_frames-clip_len)
                    indices = slice(frame, frame+self.num_frames*(batch_size-1)+1, self.num_frames)

                features_perframe = {}
                for key in features.keys():
                    features_perframe[key] = features[key][indices]
                targets_perframe = targets[indices]

                outputs = self.sem_seg_head(features_perframe, is_last=is_last)
                frame_querys.append(torch.cat(outputs['frame_query']))
                mask_features.append(outputs['mask_features'])

                # bipartite matching-based loss
                losses = self.criterion(outputs, targets_perframe)

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        if k in losses_all:
                            losses_all[k] += losses[k] * self.criterion.weight_dict[k]
                        else:
                            losses_all[k] = losses[k] * self.criterion.weight_dict[k]
                    else:
                        losses.pop(k)
            for k in list(losses.keys()):
                losses_all[k] /= self.num_frames
            outputs_video = self.video_process(frame_querys, mask_features)
            targets = self.prepare_targets(batched_inputs, images)
            losses_video = self.criterion(outputs_video, targets)
            for k in list(losses_video.keys()):
                    if k in self.criterion.weight_dict:
                        if k in losses_all:
                            losses_all[k] += losses_video[k] * self.criterion.weight_dict[k]
                        else:
                            losses_all[k] = losses_video[k] * self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)

            return losses_all
        else:
            for frame in range(0, self.num_frames, clip_len):
                if frame+clip_len < self.num_frames:
                    indices = slice(frame, frame+clip_len, 1)
                else:
                    is_last = True
                    frame = max(0, self.num_frames-clip_len)
                    indices = slice(frame, self.num_frames, 1)

                features_perframe = {}
                for key in features.keys():
                    features_perframe[key] = features[key][indices]

                outputs, outputs_video = self.sem_seg_head(features_perframe, is_last=is_last)

            mask_cls_results = outputs_video["pred_logits"]
            mask_pred_results = outputs_video["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            # upsample masks
            mask_pred_result = retry_if_cuda_oom(F.interpolate)(
                mask_pred_results[0],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs
            del outputs_video

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width)

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def prepare_targets_clip(self, targets, images, clip_len):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            for frame in range(0, self.num_frames, clip_len):
                mask_shape = [_num_instance, clip_len, h_pad, w_pad]
                gt_masks_per_clip = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

                gt_ids_per_clip = []
                for f_i in range(frame, frame+clip_len, 1):
                    targets_per_frame = targets_per_video["instances"][f_i]
                    targets_per_frame = targets_per_frame.to(self.device)
                    h, w = targets_per_frame.image_size

                    gt_ids_per_clip.append(targets_per_frame.gt_ids[:, None])
                    gt_masks_per_clip[:, f_i-frame, :h, :w] = targets_per_frame.gt_masks.tensor

                gt_ids_per_clip = torch.cat(gt_ids_per_clip, dim=1)
                valid_idx = (gt_ids_per_clip != -1).any(dim=-1)

                gt_classes_per_clip = targets_per_frame.gt_classes[valid_idx]          # N,
                gt_ids_per_clip = gt_ids_per_clip[valid_idx]                          # N, num_frames

                gt_instances.append({"labels": gt_classes_per_clip, "ids": gt_ids_per_clip})
                gt_masks_per_clip = gt_masks_per_clip[valid_idx].float()          # N, num_frames, H, W
                gt_instances[-1].update({"masks": gt_masks_per_clip})

        return gt_instances

    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output
