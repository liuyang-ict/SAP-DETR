# ------------------------------------------------------------------------
# SAP-DETR
# Copyright (c) 2023 Lenovo. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DAB-DETR (https://github.com/IDEA-Research/DAB-DETR)
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import copy
import math
from typing import Dict
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from util import box_ops
from util import box_loss
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid, gen_sineembed_for_position)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm, dice_loss)
from .transformer import build_transformer

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    return loss.mean(1).sum() / num_boxes


class SAPDETR(nn.Module):
    """ This is the SAP-DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                    aux_loss=False, 
                    iter_update=True,
                    query_dim=2, 
                    bbox_dim = 4,
                    bbox_embed_diff_each_layer=False,
                    class_embed_diff_each_layer=False,
                    meshgrid_refpoints_xy=False,
                    first_independent_head=True,
                    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         we recommend the queries with square numbers e.g., 400, 625, 900.
                         for aligning to orther DETR-like models based on 300 queries, we default for 306 (17x18) queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            iter_update: iterative update of boxes
            query_dim: query dimension. 2 for reference point.
            bbox_dim: query dimension. 4 for reference bbox.
            bbox_embed_diff_each_layer: dont share weights of regression heads. Default for False.(shared weights.)
            class_embed_diff_each_layer: dont share weights of classification heads. Default for False.(shared weights.)
            first_independent_head: share weights of prediction heads except the first head. Default for True.(shared weights except the first one.)

        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        num_decoder_layers = transformer.num_decoder_layers
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        self.class_embed_diff_each_layer = class_embed_diff_each_layer
        self.obj_embed = nn.Linear(hidden_dim, 1) 
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.point_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.class_embed =  MLP(hidden_dim, hidden_dim, num_classes, 3)
        

        # setting query dim
        # when query_dim=2 refpoint_embed is the keypoint_embed
        self.query_dim = query_dim
        assert query_dim == 2
        self.refpoint_embed = nn.Embedding(num_queries, query_dim)
        self.meshgrid_refpoints_xy = meshgrid_refpoints_xy
        if meshgrid_refpoints_xy:
            if self.num_queries == 306:
                mesh = torch.meshgrid(torch.arange(0, 1, 1/17), torch.arange(0, 1, 1/18))
            
            else:
                n = torch.arange(0, 1, 1/math.sqrt(self.num_queries))
                mesh = torch.meshgrid(n,n)
            # x -> col, y->row 
        reference_points=torch.cat([mesh[1].reshape(-1)[...,None], mesh[0].reshape(-1)[...,None]],-1)
        self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(reference_points[:,:])
        self.refpoint_embed.weight.requires_grad = False #learned or fixed ?

        # setting box dim
        # box_dim=4 box_dim is the left, top, right, and bottom referring the key point
        self.bbox_dim = bbox_dim
        assert bbox_dim == 4
        self.refbbox_embed = nn.Embedding(num_queries, bbox_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.iter_update = iter_update
                
        self.ref_bbox_head = MLP(self.bbox_dim // 2 * hidden_dim, hidden_dim, hidden_dim, 2)
        self.ref_point_head = MLP(self.query_dim // 2 * hidden_dim, hidden_dim, hidden_dim, 2)
        self.transformer.decoder.ref_bbox_head = self.ref_bbox_head
        self.transformer.decoder.ref_point_head = self.ref_point_head
        
        # init prior_prob setting for focal loss
        # init bbox_embed
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.obj_embed.bias.data = torch.ones(1) * bias_value
        
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.point_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.class_embed.layers[-1].bias.data, bias_value)
        
        
        # share the prediction heads
        if bbox_embed_diff_each_layer:
            self.bbox_embed = _get_clones(self.bbox_embed, num_decoder_layers)
            self.point_embed = _get_clones(self.point_embed, num_decoder_layers)
        else:
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_decoder_layers)])
            self.point_embed = nn.ModuleList([self.point_embed for _ in range(num_decoder_layers)])
            
        if class_embed_diff_each_layer:
            self.class_embed = _get_clones(self.class_embed, num_decoder_layers)
        elif first_independent_head:
            independent_class_embed = copy.deepcopy(self.class_embed)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_decoder_layers)])
            self.class_embed[0] = independent_class_embed
        else:
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_decoder_layers)])
        
        if self.iter_update:
            self.transformer.decoder.bbox_embed = self.bbox_embed
            self.transformer.decoder.point_embed = self.point_embed

    # iter update
    def update_point_or_bbox(self, tmp, reference, original=None, reference_point=None):
        dim_ref = tmp.shape[-1]
        assert dim_ref in [self.query_dim, self.bbox_dim]
        if dim_ref == self.bbox_dim:
            tmp[..., :dim_ref] += inverse_sigmoid(reference)
            outputs = tmp[..., :dim_ref].sigmoid()
            outputs = box_ops.box_xyltrb_to_cxcywh(reference_point, outputs)
        if dim_ref == self.query_dim:
            tmp[..., :dim_ref] += inverse_sigmoid(reference-original)
            if original.shape[1] == 306:
                new_reference = tmp[..., :dim_ref].sigmoid() * torch.tensor([[1/18, 1/17]]).to(tmp.device)
            else:
                new_reference = tmp[..., :dim_ref].sigmoid() * torch.tensor([[1/20.0]]).to(tmp.device) #(1 / math.sqrt(original.shape[1]))
            outputs = new_reference + original
#              # ablation for move the grid scale
#             tmp[..., :dim_ref] += inverse_sigmoid(reference)
#             outputs = tmp[..., :dim_ref].sigmoid()
        return outputs

#     from visualizer import get_local
    def forward(self, samples: NestedTensor,
               targets: Tensor=None,):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        bs, c, h, w = src.shape
        assert mask is not None
        
        # default pipeline
        refpoint_embedweight = self.refpoint_embed.weight        
        refbbox_embedweight = self.refbbox_embed.weight
        hs, reference_point, reference_bbox = self.transformer(self.input_proj(src), 
                                                               mask, refpoint_embedweight,
                                                               refbbox_embedweight, pos[-1])
        outputs_coords = []
        outputs_classes = []
        outputs_objs = []
        outputs_points = []
        for lvl in range(hs.shape[0]):
            bbox_embed = self.bbox_embed[lvl]
            point_embed = self.point_embed[lvl]
            class_embed = self.class_embed[lvl]
                

            tmp_bbox = bbox_embed(hs[lvl])
            tmp_point = point_embed(hs[lvl])
            outputs_point = self.update_point_or_bbox(tmp_point, reference_point[lvl], original=reference_point[0])
            outputs_coord = self.update_point_or_bbox(tmp_bbox, reference_bbox[lvl], reference_point=outputs_point)
            outputs_class = class_embed(hs[lvl]) 
            outputs_obj = self.obj_embed(hs[lvl])
            
            outputs_points.append(outputs_point)
            outputs_coords.append(outputs_coord)
            outputs_classes.append(outputs_class)
            outputs_objs.append(outputs_obj)
            
        outputs_point = torch.stack(outputs_points)
        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_obj = torch.stack(outputs_objs)


        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 
               'pred_objs': outputs_obj[-1], 'pred_points': outputs_point[-1]}
        
              
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_obj, outputs_point)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_obj, outputs_point):

        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        
        aux_outputs = [{'pred_logits': a, 'pred_boxes': b, 'pred_objs': c, 'pred_points': d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], 
                                      outputs_obj[:-1], outputs_point[:-1])]
        return aux_outputs


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha    
        
    def loss_inner(self, outputs, targets, indices, num_boxes):
        losses = {}

        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)   
        if target_boxes.size(0) <= 0:
            losses['loss_inner'] = torch.tensor(0.0).to(target_boxes.device)
            return losses
        
        target_boxes = box_ops.box_cxcywh_to_xyxy(target_boxes)
        left = (src_points[:,0] - target_boxes[:,0]) 
        top = (src_points[:,1] - target_boxes[:,1]) 
        right = (target_boxes[:,2] - src_points[:,0]) 
        bottom = (target_boxes[:,3] - src_points[:,1])
        distances = torch.stack([left, top, right, bottom], dim=-1)
        min_distances, idxs = distances.min(dim=-1)
        is_inner = (min_distances >= 0)
        loss_inner = (1.0 - is_inner.float())
        losses['loss_inner'] = loss_inner.sum() / num_boxes
        return losses
    
    def loss_objs(self, outputs, targets, indices, num_boxes, log=True):
        
        idx = self._get_src_permutation_idx(indices)
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        iou_score, _ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes),
                        box_ops.box_cxcywh_to_xyxy(target_boxes))
        iou_score = torch.diag(iou_score)
     
        assert 'pred_objs' in outputs
        src_objs = outputs['pred_objs']
        target_objs = torch.full(src_objs.shape[:2], 0, dtype=src_objs.dtype, device=src_objs.device)
        target_objs[idx] = iou_score
#         target_objs[idx] = 1 # hard label

        target_objs_onehot = torch.zeros([src_objs.shape[0], src_objs.shape[1], src_objs.shape[2]],
                                            dtype=src_objs.dtype, layout=src_objs.layout, device=src_objs.device)
        target_objs_onehot[:] = target_objs.unsqueeze(-1)[:]
        
        loss_ce_obj = sigmoid_focal_loss(src_objs, target_objs_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_objs.shape[1]
        losses = {'loss_ce_obj': loss_ce_obj}
        return losses
        
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)        
        loss_bbox = F.l1_loss(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes), reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'objs': self.loss_objs,
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'inner': self.loss_inner,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        guid_indices = self.matcher(outputs_without_aux, targets)
        if return_indices:
            indices_copy = guid_indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        guid_num_boxes = sum(len(t["labels"]) for t in targets)
        guid_num_boxes = torch.as_tensor([guid_num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(guid_num_boxes)
        guid_num_boxes = torch.clamp(guid_num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, guid_indices, guid_num_boxes))

        indices_list = []
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                
                indices = self.matcher(aux_outputs, targets)                    
                indices_list.append(indices)
                    
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}                        
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, guid_num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)                
        if return_indices:
            indices_list.append(indices_copy)
            return losses, indices_list
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox, out_obj = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_objs']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_SAPDETR(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    matcher = build_matcher(args)
    model = SAPDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=2,
        bbox_dim=4,
        bbox_embed_diff_each_layer=args.bbox_embed_diff_each_layer,
        class_embed_diff_each_layer=args.class_embed_diff_each_layer,
        meshgrid_refpoints_xy=args.meshgrid_refpoints_xy,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_ce_obj'] = args.obj_loss_coef
    weight_dict['loss_inner'] = args.inner_loss_coef # inner loss
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        # if new_init use args.dec_layers rather than args.dec_layers-1
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ['labels', 'boxes', 'objs', 'inner']

    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
