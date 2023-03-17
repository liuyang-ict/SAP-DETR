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

import math
import copy
import os
from typing import Optional, List
from util.misc import inverse_sigmoid

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention

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

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 20 ** (2 * (dim_t // 2) / 128) # as same as the pos in encoder
    assert pos_tensor.size(-1) in [2, 4], "Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1))

    poses = []
    for i in range(pos_tensor.size(-1)):
        embed = pos_tensor[:, :, i] * scale
        pos = embed[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
        poses.append(pos)

    poses = torch.cat(poses, dim=2)
    return poses

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=300, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=2, bbox_dim=4,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 bbox_embed_diff_each_layer=False,
                 sdg=False,
                 rm_self_attn_decoder=False,
                 newconvinit=False
                 ):

        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.newconvinit = newconvinit
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos,
                                                rm_self_attn_decoder=rm_self_attn_decoder, sdg=sdg)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim, bbox_dim=bbox_dim,
                                          keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        self._reset_parameters()
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        
        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        
        
        # conv for new init generation
        if self.newconvinit:
            self.point_transfer = nn.Sequential(
                    nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.GroupNorm(32, d_model),
                    nn.ReLU(),
                    nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.GroupNorm(32, d_model),
                    nn.ReLU(),
                    nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.GroupNorm(32, d_model),
                    nn.ReLU(),
            )
            for l in self.point_transfer.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)        
        # end

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, refpoint_embed, refbbox_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        key_pos = None
        mesh = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        # x -> col, y->row 
        key_pos=torch.cat([mesh[1].reshape(-1)[...,None], mesh[0].reshape(-1)[...,None]],-1).to(src.device)
        key_pos = key_pos.unsqueeze(0).repeat(bs, 1, 1)
        
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)
        refbbox_embed = refbbox_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)        
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        num_queries = refpoint_embed.shape[0]
        
        
        if self.newconvinit:
            memory_2d = memory.reshape(h, w, bs, c).permute(2, 3, 0, 1)    
            memory_2d = self.point_transfer(memory_2d)
            center_pos = torch.tensor([0.5, 0.5]).to(refpoint_embed.device)
            tgt = F.grid_sample(memory_2d, 
                               (refpoint_embed.sigmoid().transpose(0, 1).unsqueeze(1)-center_pos[None, None, None, :]) * 2, 
                               mode="bilinear", padding_mode="zeros", align_corners=False) #[bt, d_model, h, w]
            tgt = tgt.flatten(2).permute(2, 0, 1)
            
        else:
            tgt = torch.zeros(num_queries, bs, self.d_model, device=refpoint_embed.device)

            
        hs, reference_points, reference_bboxes  = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                               pos=pos_embed, refpoints_unsigmoid=refpoint_embed,
                                                               refbboxes_unsigmoid=refbbox_embed, key_pos=key_pos)
                          
        return hs, reference_points, reference_bboxes



class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.query_scales = _get_clones(MLP(d_model, d_model, d_model, 2), num_layers)
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer_id, layer in enumerate(self.layers):
            # rescale the content and pos sim
            pos_scales = self.query_scales[layer_id](output)
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos*pos_scales)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, 
                    d_model=256, query_dim=2, bbox_dim=4, keep_query_pos=False, 
                    query_scale_type='cond_elewise', bbox_embed_diff_each_layer=False,
                    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim
        self.bbox_dim = bbox_dim
        
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
        
        self.ref_bbox_head = None
        self.ref_point_head = None
        self.bbox_embed = None

        self.d_model = d_model
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer        
        
        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
#                 # pos transformation
#                 self.layers[layer_id + 1].ca_kpos_proj = None
                self.layers[layer_id + 1].ca_point_qpos_proj = None
                self.layers[layer_id + 1].ca_bbox_qpos_proj = None
        
    # iter update    
    def update_point_or_bbox(self, tmp, reference, original=None):
        dim_ref = tmp.shape[-1]
        assert dim_ref in [self.query_dim, self.bbox_dim]
        if dim_ref == self.bbox_dim:
            tmp[..., :dim_ref] += inverse_sigmoid(reference)
            new_reference = tmp[..., :dim_ref].sigmoid()
        if dim_ref == self.query_dim:
            tmp[..., :dim_ref] += inverse_sigmoid(reference-original)
            if original.shape[0] == 306:
                new_reference = tmp[..., :dim_ref].sigmoid() * torch.tensor([[1/18, 1/17]]).to(tmp.device)
            else:
                new_reference = tmp[..., :dim_ref].sigmoid() * (1 / math.sqrt(original.shape[0]))
            new_reference = new_reference + original

#              # ablation for move the grid scale
#             tmp[..., :dim_ref] += inverse_sigmoid(reference)
#             new_reference = tmp[..., :dim_ref].sigmoid()
        return new_reference
    
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 2
                refbboxes_unsigmoid: Optional[Tensor] = None,
                key_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        attn_weights = []
        reference_points = refpoints_unsigmoid.sigmoid()
        reference_bboxes = refbboxes_unsigmoid.sigmoid()
        original_points = reference_points
        ref_points = [reference_points]
        ref_bboxes = [reference_bboxes]  
        decoder_topk_indexes = []
        decoder_topk_index=None
        decoder_mask_predictions = []  

        for layer_id, layer in enumerate(self.layers):
            # get sine embedding for the query vector
            obj_point = reference_points[..., :self.query_dim] # [num_queries, batch_size, 2]
            obj_bbox = reference_bboxes[..., :self.bbox_dim] # [num_queries, batch_size, 4]
            query_sine_embed = gen_sineembed_for_position(obj_point) # [num_queries, batch_size, d_model]
            bbox_query_sine_embed = gen_sineembed_for_position(torch.cat([obj_point-obj_bbox[...,:2],obj_point+obj_bbox[...,2:]],dim=-1)) # [num_queries, batch_size, 2*d_model]
            query_pos = self.ref_point_head(query_sine_embed) # [num_queries, batch_size, d_model]
            bbox_query_pos = self.ref_bbox_head(bbox_query_sine_embed) # [num_queries, batch_size, d_model]

            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)

            else:
                pos_transformation = self.query_scale.weight[layer_id]

#             apply transformation
            query_sine_embed = query_sine_embed * pos_transformation
            pos_sine_embed = pos
        
            # add box transformation
            if layer_id != 0:
                bbox_query_sine_embed = bbox_query_sine_embed * pos_transformation.repeat(1,1,2)            
            
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos_sine_embed, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           bbox_query_pos=bbox_query_pos, bbox_query_sine_embed=bbox_query_sine_embed, 
                           is_first=(layer_id == 0), key_pos=key_pos, point_pos=reference_points, bbox_ltrb=reference_bboxes)
            
            # iter update
            if self.bbox_embed is not None:
                tmp_bbox = self.bbox_embed[layer_id](output)
                tmp_point = self.point_embed[layer_id](output)
                new_reference_bboxes = self.update_point_or_bbox(tmp_bbox, reference_bboxes)
                new_reference_points = self.update_point_or_bbox(tmp_point, reference_points, original_points)
                
                if layer_id != self.num_layers - 1:
                    ref_bboxes.append(new_reference_bboxes)
                    ref_points.append(reference_points)
#                     ref_points.append(new_reference_points)
                    
                reference_bboxes = new_reference_bboxes.detach()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))
        
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                    torch.stack(ref_bboxes).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2), 
                    reference_points.unsqueeze(0).transpose(1, 2),
                    reference_bboxes.unsqueeze(0).transpose(1, 2),
                ]

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    
    def forward(self,src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False, sdg=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_point_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_bbox_qpos_proj = nn.Linear(d_model, d_model)

            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_point_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_bbox_kpos_proj = nn.Linear(d_model, d_model)

            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)


        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_point_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_bbox_qpos_proj = nn.Linear(d_model, d_model)

        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)

        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_bbox_qpos_sine_proj = nn.Linear(d_model*2, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*3, nhead, dropout=dropout, vdim=d_model)
        
        self.sdg = sdg
        if self.sdg:
            self.gaussian_proj = MLP(d_model, d_model, 4*nhead, 3) # if sdg is True

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

#     from visualizer import get_local
#     @get_local('sa_attns', 'ca_attns')
    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     bbox_query_pos: Optional[Tensor] = None,
                     bbox_query_sine_embed = None,
                     is_first = False,
                     key_pos=None, 
                     point_pos=None, 
                     bbox_ltrb=None,
               ):
                     
        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            # target is the input of the first decoder layer. zero by default.
            
            k_content = self.sa_kcontent_proj(tgt)
            k_point_pos = self.sa_point_kpos_proj(query_pos)
            k_bbox_pos = self.sa_bbox_kpos_proj(bbox_query_pos)
            v = self.sa_v_proj(tgt)
            
            
            q_content = self.sa_qcontent_proj(tgt)      
            q_point_pos = self.sa_point_qpos_proj(query_pos)
            q_bbox_pos = self.sa_bbox_qpos_proj(bbox_query_pos)
                
            num_queries, bs, n_model = k_content.shape

            q = q_content + q_point_pos + q_bbox_pos
            k = k_content + k_point_pos + k_bbox_pos

            tgt2, attn_weights, attn_q, attn_k = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                                                key_padding_mask=tgt_key_padding_mask)

            # only for visualize
            content_attn = torch.bmm(attn_q, attn_k.transpose(1, 2))
            sa_attns = content_attn

            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            

        if self.sdg:
            # point_pos [num_queries, bs, 2]
            # key_pos [bs, len(memory), 2]
            w, h = key_pos[..., 0].max(-1)[0][0].item()+1, key_pos[..., 1].max(-1)[0][0].item()+1
            memory_size = torch.tensor([w, h]).to(point_pos.device)
            key_pos = key_pos.repeat(self.nhead, 1, 1)
            
            gaussian_mapping = self.gaussian_proj(tgt)
            offset = gaussian_mapping[..., :self.nhead*2].tanh() # if negative multiple left/top, elif positive multiple right/down
            
            point_pos = (point_pos * memory_size[None, None, :]).repeat(1, 1, self.nhead)
            bbox_ltrb = bbox_ltrb * memory_size[None, None, :].repeat(1, 1, 2)
            bbox_ltrb = torch.stack((-bbox_ltrb[..., :2], bbox_ltrb[..., 2:]), dim=2).repeat(1, 1, 1, self.nhead)
            sample_offset = bbox_ltrb * offset.unsqueeze(2)
            sample_offset = sample_offset.max(-2)
            sample_offset = sample_offset[0] * (2*sample_offset[1]-1)
            sample_point_pos = point_pos + sample_offset

#             # ablation on noinner
#             sample_point_pos = point_pos.repeat(1, 1, self.nhead) + offset

            sample_point_pos = sample_point_pos.reshape(num_queries, bs, self.nhead, 2).reshape(num_queries, bs*self.nhead, 2)
            scale = gaussian_mapping[..., self.nhead*2:].reshape(num_queries, bs, self.nhead, 2).reshape(num_queries, bs*self.nhead, 2).transpose(0, 1)         
            
            relative_position = (key_pos+0.5).unsqueeze(1) - sample_point_pos.transpose(0,1).unsqueeze(2)  
            gaussian_map = (relative_position.pow(2) * scale.unsqueeze(2).pow(2)).sum(-1)
            gaussian_map = -(gaussian_map - 0).abs() / 8.0   
        
        else:
            gaussian_map = None

        
        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        
        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we add the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
# #             for transformation
#             k_pos = self.ca_kpos_proj(pos)
            q_point_pos = self.ca_point_qpos_proj(query_pos)
            q_bbox_pos = self.ca_bbox_qpos_proj(bbox_query_pos)
            q = q_content + q_point_pos + q_bbox_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        # peca
        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        
        bbox_query_sine_embed = self.ca_bbox_qpos_sine_proj(bbox_query_sine_embed)
        bbox_query_sine_embed = bbox_query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed, bbox_query_sine_embed], dim=3).view(num_queries, bs, n_model * 3)

        
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos, k_pos], dim=3).view(hw, bs, n_model * 3)

        # relative positional encoing as bias adding to the attention map before softmax operation  
        tgt2, attn_weights, attn_q, attn_k = self.cross_attn(query=q,key=k,
                               value=v, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask,
                               gaussian_map=gaussian_map)

#         # only for visualize
#         head_model = n_model//self.nhead
#         content_attn = torch.bmm(attn_q[..., :head_model], attn_k[..., :head_model].transpose(1, 2))
#         point_attn = torch.bmm(attn_q[..., head_model:2*head_model], attn_k[..., head_model:2*head_model].transpose(1, 2))
#         side_attn = torch.bmm(attn_q[..., 2*head_model:], attn_k[..., 2*head_model:].transpose(1, 2))

#         if self.sdg:
#             ca_attns = torch.cat([content_attn,point_attn,side_attn, gaussian_map],dim=0)
#         else:
#             ca_attns = torch.cat([content_attn,point_attn,side_attn],dim=0)

        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=2,
        bbox_dim=4,
        activation=args.transformer_activation,
        sdg=args.sdg,
        rm_self_attn_decoder=args.rm_self_attn_decoder,
        bbox_embed_diff_each_layer=args.bbox_embed_diff_each_layer,
        newconvinit=args.newconvinit,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
