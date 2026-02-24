import torch
from torch import nn
import math
import warnings
from typing import Sequence
from torch.nn import functional as F

from .query_GL import MAGL
from .query_GL_topkv6 import MAGLtopkv6
from .attention import CAMMV1, MultiheadAttention as MHA, CAMM
from mmengine.model import BaseModule
from mmcv.cnn import Linear, Scale, build_activation_layer, build_norm_layer
from torch.nn import Sequential
from mmcv.cnn.bricks.drop import build_dropout
from .querydetection import QueryDetection
from .utils import gen_sineembed_for_position

from fvcore.nn import FlopCountAnalysis, parameter_count_table

class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=dict(type="LN"),
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels * 4 # 256 * 4
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)

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


class Decoder(nn.Module):
    def __init__(self, config, anchor_size=3, no_sine_embed=False, return_intermediate=True):
        super(Decoder,self).__init__()
        self.pc_range = config['cav_lidar_range']
        self.num_layers = config['num_layer']
        self.embed_dims = config['embed_dims']
        self.anchor_size = anchor_size
        self.no_sine_embed = no_sine_embed
        self.return_intermediate = return_intermediate
        self.query_scale = MLP( self.embed_dims,  self.embed_dims,  self.embed_dims, 2)
        if self.no_sine_embed:
            self.ref_point_head = MLP(self.anchor_size, self.embed_dims, self.embed_dims, 3)
        else:
            self.ref_point_head = MLP(self.embed_dims * self.anchor_size, self.embed_dims, self.embed_dims, 2)
            # self.ref_point_head = MLP(self.embed_dims * self.anchor_size, self.embed_dims, self.embed_dims, 2)

        if 'operation_order' not in config:
            self.operation_order = [
                "self-attention",
                "norm",
                "cross-attention",
                "norm",
                "ffn",
                "norm",
            ] * self.num_layers
        else:
            self.operation_order = config['operation_order']
        
        self_attention = config['self-attention']
        norm_layer_config = config['norm_layer']
        ffn_config = config['ffn']
        cross_attention = config['cross-attention']
        querydecoder = config['querydecoder']
        op_config_dict = {
            'self-attention': self_attention,
            'cross-attention': cross_attention,
            'ffn': ffn_config,
            'norm': norm_layer_config,
            'querydecoder': querydecoder,
            # 'local_attention': config['local_attention'],
        }

        self.agent_fuse = False
        if 'local_attention' in config:
            op_config_dict.update({
                'local_attention': config['local_attention'],
            })
            self.fusion_layer = config['fusion_layer']
            print("local_attention_layers:",config['fusion_layer'])
            if len(self.fusion_layer) == 0:
                print("Not use local-fusion!!!")
            else:
                self.agent_fuse = True
        
        if 'knn-fusion' in config:
            op_config_dict.update({
                'knn-fusion': config['knn-fusion'],
            })
            self.knn_fusion_layer = config['knn_fusion_layer']
            print("knn_fusion_layer:",config['knn_fusion_layer'])
            if len(self.knn_fusion_layer) == 0:
                print("Not use knn-fusion!!!")
            else:
                self.agent_fuse = True
            

        if 'query_all_once' in config:
            self.query_all_once = config['query_all_once']
        else:
            self.query_all_once = False


        # useless 
        if 'query-fusion' in config:

            if 'step_by_step' in config['query-fusion']:
                print("using global or local query step by step")
            else:
                print("using global and local iteratively.")
            self.agent_fuse = True
            self.fusion_layer = config['fusion_layer']
            queryfusion = config['query-fusion']
            op_config_dict.update(
                {
                    'query-fusion': queryfusion,
                }
            )
        
        self.layers = nn.ModuleList()
        for layer in range(self.num_layers):
            for op in self.operation_order:
                module_instance = None
                if op == 'self-attention':
                    module_instance = MHA(op_config_dict[op])
                if op == 'norm':
                    module_instance = nn.LayerNorm(**op_config_dict[op])
                if op == 'cross-attention':
                    if 'version' in op_config_dict[op]:
                        module_instance = CAMMV1(op_config_dict[op])
                    else:
                        module_instance = CAMM(op_config_dict[op])
                if op == 'ffn':
                    module_instance = AsymmetricFFN(**op_config_dict[op])
                if op == 'querydecoder':
                    module_instance = QueryDetection(**op_config_dict[op])
                    # module_instance = QueryDetection(embed_dims=self.embed_dims, pc_range=self.pc_range)

                if op == 'local_attention' and layer in self.fusion_layer:
                    module_instance = MAGL(op_config_dict[op])
                
                if op == 'knn-fusion' and layer in self.knn_fusion_layer:
                    module_instance = MAGLtopkv6(op_config_dict[op])

                self.layers.append(module_instance)
        
        #TODO: 
        # Init bbox_embed

    def forward(self,
            query,
            key=None,
            # value=pts_feat_flatten,
            pts_feats=None,
            img_feats=None,
            query_pos= None,
            key_padding_mask=None,
            reference_points=None,
            spatial_shapes=None,
            level_start_index=None,
            valid_ratios=None,
            reg_branches=None,
            data_dict=None,
            ):
        output = query  # num_query, B, C
        assert query_pos == None
        bs = query.shape[1]
        if len(reference_points.shape) == 2:
            reference_points = reference_points[None].repeat(bs, 1, 1) # bs, nq, 4(xywh)

        predictions = []
        boxes = []
        dirs = []
        i = 0 
        record_len = data_dict['record_len']
        for lid in range(self.num_layers):
            reference_points_input = reference_points.clone()
            if self.no_sine_embed:
                raw_query_pos = self.ref_point_head(reference_points_input)
            else:
                query_sine_embed = gen_sineembed_for_position(reference_points_input,self.embed_dims) # bs, nq, 256*2 
                raw_query_pos = self.ref_point_head(query_sine_embed) # bs, nq, 256
            pos_scale = self.query_scale(output) if lid != 0 else 1
            raw_query_pos = raw_query_pos.permute(1, 0, 2)
            query_pos = pos_scale * raw_query_pos
            for op in self.operation_order:
                if op == 'self-attention':
                    output = self.layers[i](
                        query=output,
                        query_pos=query_pos
                    )
                elif op == 'norm' or op == 'ffn':
                    output = self.layers[i](output) 
                
                elif op == 'cross-attention':      
                    output = self.layers[i](
                        query=output,
                        query_pos=query_pos,
                        key_padding_mask=key_padding_mask,
                        reference_points=reference_points_input,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        pts_feats=pts_feats,
                        img_feats=img_feats,
                        data_dict=data_dict,
                    )
                elif op == 'querydecoder':
                    output = output.permute(1, 0, 2)
                    cls, box, dir, new_reference_points = self.layers[i](output, reference_points, lid=lid)
                    reference_points = new_reference_points.detach()
                    # cls : N,900,1 --> B,900,1
                    predictions.append(cls)
                    boxes.append(box)
                    dirs.append(dir)                        
                    
                elif op == 'query-fusion' and lid in self.fusion_layer:
                    output = self.layers[i](
                        output, reference_points_input, data_dict, lid, self.layers[i+3].cls_layers, query_pos
                    )
                elif op == 'local_attention' and lid in self.fusion_layer:
                    output = self.layers[i](
                        output, reference_points_input, data_dict, lid, None, query_pos
                    )
                elif op == 'knn-fusion' and lid in self.knn_fusion_layer:
                    for param in self.layers[i+4].cls_layers.parameters():
                        param.requires_grad = False
                    output = self.layers[i](
                        output, reference_points_input, data_dict, lid, self.layers[i+4].cls_layers, query_pos
                    )
                    for param in self.layers[i+4].cls_layers.parameters():
                        param.requires_grad = True 

                        
                i = i + 1
            output = output.permute(1, 0, 2)
            output = torch.nan_to_num(output)

        return predictions, boxes, dirs


class QUERY_TransFormer(nn.Module):
    def __init__(self,config,anchor_size=3) -> None:
        super(QUERY_TransFormer,self).__init__()

        self.embed_dims = config['embed_dims']
        self.lvl_mbed_dims =  self.embed_dims  # PE is double feature 
        self.decoder = Decoder(config['decoder'], anchor_size=anchor_size)
        self.num_lidar_feature_levels = config['num_lidar_feature_levels']
        self.anchor_size = anchor_size
        self.init_layers()

    
    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_lidar_feature_levels, self.lvl_mbed_dims))

    def feats_prepare(self, mlvl_feats, mlvl_masks, mlvl_pos_embeds):
        if mlvl_feats is not None:
            batch_size = mlvl_feats[0].shape[0]
            feat_flatten = []
            mask_flatten = []
            lvl_pos_embed_flatten = []
            spatial_shapes = []
            for lvl, (feat, mask, pos_embed) in enumerate(
                    zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
                bs, c, h, w = feat.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
                feat = feat.flatten(2).transpose(1, 2)
                mask = mask.flatten(1)
                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
                feat_flatten.append(feat)
                mask_flatten.append(mask)
            feat_flatten = torch.cat(feat_flatten, 1)
            mask_flatten = torch.cat(mask_flatten, 1)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            spatial_shapes = torch.as_tensor(
                spatial_shapes, dtype=torch.long, device=feat_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros(
                (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_ratios = torch.stack(
                [self.get_valid_ratio(m) for m in mlvl_masks], 1)

            feat_flatten = feat_flatten # (H*W, bs, embed_dims)
            lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
                1, 0, 2)  # (H*W, bs, embed_dims)
        else:
            # bs, n, c = mlvl_feats[0].shape[:3]
            feat_flatten = None
            mask_flatten = None
            lvl_pos_embed_flatten = None
            spatial_shapes = None
            level_start_index = None
            valid_ratios = None
        
        return feat_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, \
                level_start_index, valid_ratios
    
    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
        
    def forward(self, 
                pts_feats, 
                image_feats, 
                mlvl_pts_masks, 
                query_embeds, 
                mlvl_positional_encodings, 
                attn_mask=None,
                data_dict=None,
                reg_branches=None,
                ):
        
        assert query_embeds is not None
        # batch_size = image_feats[0].shape[0]
        if pts_feats is not None:
            batch_size = pts_feats[0].shape[0]
            feat_flatten = []
            mask_flatten = []
            lvl_pos_embed_flatten = []
            spatial_shapes = []
            for lvl, (feat, mask, pos_embed) in enumerate(
                    zip(pts_feats, mlvl_pts_masks, mlvl_positional_encodings)):
                bs, c, h, w = feat.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
                feat = feat.flatten(2).transpose(1, 2)
                mask = mask.flatten(1)
                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
                feat = feat + lvl_pos_embed
                feat_flatten.append(feat)
                mask_flatten.append(mask)
            feat_flatten = torch.cat(feat_flatten, 1)
            mask_flatten = torch.cat(mask_flatten, 1)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            spatial_shapes = torch.as_tensor(
                spatial_shapes, dtype=torch.long, device=feat_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros(
                (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_ratios = torch.stack(
                [self.get_valid_ratio(m) for m in mlvl_pts_masks], 1)

            pts_feat_flatten = feat_flatten # (H*W, bs, embed_dims)
            lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
                1, 0, 2)  # (H*W, bs, embed_dims)
        else:
            bs = image_feats[0].shape[0]
            pts_feat_flatten = None
            mask_flatten = None
            lvl_pos_embed_flatten = None
            spatial_shapes = None
            level_start_index = None
            valid_ratios = None

        reference_points = query_embeds[..., self.embed_dims:]  # position x,y,z
        tgt = query_embeds[..., :self.embed_dims] # query
        if len(tgt.shape) == 2:
            query = tgt.unsqueeze(0).expand(bs, -1, -1)   # bs, N, 128
        else:
            query = tgt

        query = query.permute(1, 0, 2)
        
        predictions, boxes, dirs = self.decoder(
            query=query,
            key=None,
            # value=pts_feat_flatten,
            pts_feats=pts_feat_flatten,
            img_feats=image_feats,
            query_pos= None,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            data_dict=data_dict,
            )
        
        return predictions, boxes, dirs
