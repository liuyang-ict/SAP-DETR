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

import torch
import math
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.box_loss import sd_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_inner: float = 9999, focal_alpha = 0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
            cost_inner: This is the punishment weight of the inner loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_inner = cost_inner
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or \
               cost_inner != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha
        
    @torch.no_grad()  
    def compute_cost_inner(self, out_point, tgt_bbox, num_queries):
        '''
            out_point:  (nr_out, 2), 2 is (x, y)
            tgt_bbox:  (nr_tgt, 4), 4 is (cx, cy, w, h)
        '''
        if tgt_bbox.size(0) <= 0:
            return 1.0
        width = tgt_bbox[:,2]
        height = tgt_bbox[:, 3]

        tgt_bbox = box_cxcywh_to_xyxy(tgt_bbox)
        left = (out_point[:,0,None] - tgt_bbox[None,:,0]) 
        top = (out_point[:,1,None] - tgt_bbox[None,:,1]) 
        right = (tgt_bbox[None,:,2] - out_point[:,0,None]) 
        bottom = (tgt_bbox[None,:,3] - out_point[:,1,None])

        distances = torch.stack([left, top, right, bottom], dim=-1)
        
        min_distances, idxs = distances.min(dim=-1)

        is_inner = (min_distances >= 0)
        cost_inner = (1.0 - is_inner.float())

        return cost_inner


    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                 "pred_points": Tensor of dim [batch_size, num_queries, 2] with the predicted point coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_point = outputs["pred_points"].flatten(0, 1) # [batch_size * num_queries, 2]
        
        # with obj match
        out_obj_prob = outputs["pred_objs"].flatten(0, 1).sigmoid() # [batch_size * num_queries, 1]
        out_prob = out_prob * out_obj_prob
        
        
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]     
        cost_bbox = torch.cdist(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox), p=1)

        # Compute the giou cost betwen boxes
        # import ipdb; ipdb.set_trace()
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_inner = self.compute_cost_inner(out_point, tgt_bbox, num_queries)
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_inner * cost_inner
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(
        cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou, focal_alpha=args.focal_alpha
    )
