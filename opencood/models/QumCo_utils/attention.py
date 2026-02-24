import warnings
from einops import rearrange
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE

from .querydetection import linear_relu_ln
try:
    from ..ops import DeformableAggregationFunction as DAF
except:
    DAF = None

from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_

def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)



class CAMMV1(nn.Module):
    def __init__(self, cfg) -> None:
        super(CAMMV1,self).__init__()
        self.use_lidar = cfg['use_lidar']
        self.batch_first = False
        self.use_camera = cfg['use_camera']
        self.pc_range = cfg['pc_range']
        self.embed_dims = embed_dims = cfg['embed_dims']
        self.num_heads = num_heads = cfg['num_heads']
        self.num_points = num_points = cfg['num_points']
        self.lidar_levels = lidar_levels = cfg['lidar_level']
        self.camera_level = camera_level = cfg['camera_level']
        self.num_cam = cfg['num_cam']
        self.im2col_step = 64

        if 'fuse_method' in cfg:
            self.fuse_method = cfg['fuse_method']
        else:
            self.fuse_method = 'add'

        if self.use_lidar:
            self.sampling_offsets = nn.Linear(
                embed_dims, num_heads * lidar_levels * num_points * 2)
            self.attention_weights = nn.Linear(embed_dims,
                                            num_heads * lidar_levels * num_points)
            self.value_proj = nn.Linear(embed_dims, embed_dims)

            # self.output_proj = nn.Linear(embed_dims, embed_dims)

            self.output_proj = nn.Sequential(
                nn.Linear(self.embed_dims,self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(),
            )

        if 'use_deformable' in cfg:
            self.use_deformable = cfg['use_deformable']
        else:
            self.use_deformable = False
        if self.use_camera:
            if self.use_deformable:
                self.weights_fc = nn.Linear(self.embed_dims, self.num_heads * self.camera_level)
                self.attn_drop = 0.0
                self.camera_encoder = nn.Sequential(
                    *linear_relu_ln(embed_dims, 1, 2, 12)
                )
            else:
                self.img_attention_weights = nn.Linear(embed_dims,
                                           self.num_cam*camera_level)
                # self.img_output_proj = nn.Linear(embed_dims, embed_dims)
                self.img_output_proj = nn.Sequential(
                    nn.Linear(self.embed_dims,self.embed_dims),
                    nn.LayerNorm(self.embed_dims),
                    nn.ReLU(),
                )

                self.position_encoder = nn.Sequential(
                    nn.Linear(3, self.embed_dims), 
                    nn.LayerNorm(self.embed_dims),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims), 
                    nn.LayerNorm(self.embed_dims),
                    nn.ReLU(inplace=True),
                )

                self.weight_dropout = nn.Dropout(0.0)
                constant_init(self.img_attention_weights, val=0., bias=0.)
        # *2 or not 
        if self.fuse_method == 'add':
            self.fused_embed = self.embed_dims
        elif self.fuse_method == 'cat':
            self.fused_embed = self.embed_dims * 2

        self.modality_fusion_layer = nn.Sequential(
            nn.Linear(self.fused_embed, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=False),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
        )    
        self.dropout = nn.Dropout(0.1)
        self.init_weights()


    def _get_weights(self, q_feature, data_dict=None):
        bs, num_query = q_feature.shape[:2]
        if self.camera_encoder is not None:
            if data_dict['image_inputs']['camera_to_lidar'].shape == 3:
                camera_embed = self.camera_encoder(data_dict['image_inputs']['camera_to_lidar'][:,:,:3].reshape(bs,self.num_cam,-1))
            else:
                camera_embed = self.camera_encoder(
                    data_dict['image_inputs']['camera_to_lidar'][:, :,:,:3].reshape(bs, self.num_cam, -1))
            q_feature = q_feature[:, :, None] + camera_embed[:, None]
        weights = (
            self.weights_fc(q_feature)
            .reshape(bs, num_query, -1, self.num_heads)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_query,
                self.num_cam,
                self.camera_level,
                self.num_heads,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_query, self.num_cam, 1, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights
    

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.lidar_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        if self.use_lidar:
            constant_init(self.sampling_offsets, 0.)
            self.sampling_offsets.bias.data = grid_init.view(-1)
            constant_init(self.attention_weights, val=0., bias=0.)
            xavier_init(self.value_proj, distribution='uniform', bias=0.)
            

    def forward(self,
            query,
            key=None,
            value=None,
            identity=None,
            query_pos=None,
            key_padding_mask=None,
            reference_points=None,
            spatial_shapes=None,
            level_start_index=None,
            pts_feats=None,
            img_feats=None,
            data_dict=None,
            ):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
            
        if not self.batch_first: # False
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        
        bs, num_query, _ = query.shape

        if self.use_lidar and pts_feats is not None:
            value = pts_feats
            bs, num_value, _ = value.shape
            
            assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

            value = self.value_proj(value)
            if key_padding_mask is not None:
                value = value.masked_fill(key_padding_mask[..., None], 0.0)
            value = value.view(bs, num_value, self.num_heads, -1)
            sampling_offsets = self.sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.lidar_levels, self.num_points, 2)
            attention_weights = self.attention_weights(query).view(
                bs, num_query, self.num_heads, self.lidar_levels * self.num_points)
            attention_weights = attention_weights.softmax(-1)

            attention_weights = attention_weights.view(bs, num_query,
                                                    self.num_heads,
                                                    self.lidar_levels,
                                                    self.num_points)
            ref_points = reference_points.unsqueeze(2).expand(-1, -1, self.lidar_levels, -1)
            
            # ref_points = reference_points
            ref_points = ref_points[..., :2]
            if ref_points.shape[-1] == 2:
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                sampling_locations = ref_points[:, :, None, :, None, :] \
                    + sampling_offsets \
                    / offset_normalizer[None, None, None, :, None, :]
            else:
                raise ValueError(
                    f'Last dim of reference_points must be'
                    f' 2, but get {reference_points.shape[-1]} instead.')
            if ((IS_CUDA_AVAILABLE and value.is_cuda)
                    or (IS_MLU_AVAILABLE and value.is_mlu)):
                output = MultiScaleDeformableAttnFunction.apply( 
                    value, spatial_shapes, level_start_index, sampling_locations,
                    attention_weights, self.im2col_step)
            else:
                output = multi_scale_deformable_attn_pytorch(
                    value, spatial_shapes, sampling_locations, attention_weights)

            pts_output = self.output_proj(output)
        else:
            pts_output = None

        query = output
        if self.use_camera and img_feats is not None:
            if not self.use_deformable:
                bs_with_agents, num_cams, C,_,_ = img_feats[0].shape

                img_attention_weights = self.img_attention_weights(query).view(
                    bs, 1, num_query, self.num_cam, 1, self.camera_level)
                
                reference_points_3d, img_output, mask = self.get_sample_2d(img_feats, bs_with_agents, reference_points, data_dict)
                img_output = torch.nan_to_num(img_output)
                mask = torch.nan_to_num(mask)

                img_attention_weights = self.weight_dropout(img_attention_weights.sigmoid()) * mask
                img_output = img_output * img_attention_weights
                # output (B, emb_dims, num_query)
                img_output = img_output.sum(-1).sum(-1).sum(-1)
                # output (num_query, B, emb_dims)
                img_output = img_output.permute(0, 2, 1)

                img_output = self.img_output_proj(img_output) # BS,900,C
            else:
                # deformable attention for image feature. 
                weights = self._get_weights(
                    query, data_dict
                )
                
                # TODO: sample reference 2d points
                points_2d = (
                    self.project_points(
                        reference_points,
                        data_dict,
                    ).reshape(bs, num_query, self.num_cam, 2)
                )   # project 3D keypoints into 2D image feature ,  bs,900*13,6,2

                img_output = DAF.apply(
                    *img_feats, points_2d, weights).reshape(bs, num_query,self.embed_dims)

        else:
            img_output = None

        if self.use_camera and self.use_lidar and img_output is not None and pts_output is not None:
            if self.fuse_method == 'add':
                output = torch.sum((img_output, pts_output), dim=2)
            elif self.fuse_method == 'cat':
                output = torch.cat((img_output, pts_output), dim=2)
            output = self.modality_fusion_layer(output)
        elif img_output is not None:
            output = img_output
        else:
            output = pts_output
        output = output.permute(1, 0, 2)
        return self.dropout(output) + identity


    def project_points(self, key_points, data_dict, visualize_point=False):
        key_points = key_points.clone()

        key_points[..., 0:1] = key_points[..., 0:1] * \
            (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]  # points * H * grid == points * pc_range
        key_points[..., 1:2] = key_points[..., 1:2] * \
            (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        key_points[..., 2:3] = key_points[..., 2:3] * \
            (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        
        bs, num_query = key_points.shape[:2]
        image_inputs_dict = data_dict['image_inputs']
        x, rots, trans, intrins, post_rots, post_trans = \
            (image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'],
             image_inputs_dict['post_rots'], image_inputs_dict['post_trans'])
        _,_,_,H,W = x.shape
        B,N,_ = trans.shape

        points = key_points.reshape(bs,1,num_query,-1).repeat(1,N,1,1)

        homogeneous_points = torch.cat(
                (points, torch.ones_like(points[..., :1])), -1).unsqueeze(-1) # B,4,num_query,4,1
        
        
        camera_to_lidar = image_inputs_dict["camera_to_lidar"]
        ego_to_camera = torch.inverse(camera_to_lidar).reshape(bs,N,1,4,4)
        points_cam = torch.matmul(ego_to_camera,homogeneous_points)
        points_cam =  points_cam[...,:3,:] / points_cam[...,3:4,:]

        intrins = intrins.reshape(B,N,1,3,3)
        points_2d = torch.matmul(intrins,points_cam).squeeze(-1)  # must point -1 otherwise, will eliminate some other dims
        # points_2d = []

        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )

        # another method
        # points = key_points.reshape(bs,1,num_anchor,num_pts,-1).repeat(1,N,1,1,1)  # + post_trans.view(bs,N,1,1,-1)
        # points = points.unsqueeze(-1)
        # ones = torch.ones_like(points[..., :1, :])
        # homogeneous_points = torch.cat([points, ones], dim=4)
        #
        # ext_matrix = torch.inverse(camera_to_lidar)[...,:3, :4].reshape(bs,N,1,1,3,4)
        # img_pts = (intrins @ ext_matrix @ homogeneous_points).squeeze()
        # # b,4,900,17,3
        # depth = img_pts[...,2:3]
        # img_pts = img_pts[...,:2] / torch.clamp(
        #     depth, min=1e-6
        # )
        # ones = torch.ones_like(img_pts[..., :1])
        # img_pts = torch.cat([img_pts,ones],dim=-1)
        # post_rots = post_rots.reshape(bs,N,1,1,3,3).repeat(1,1,num_anchor,num_pts,1,1)
        # img_pts = post_rots.matmul(img_pts.unsqueeze(-1)).squeeze()
        # img_pts = img_pts + post_trans.view(bs,N,1,1,-1)
        # img_pts = img_pts[...,:2]


        # for postprocess in image
        ones = torch.ones_like(points_2d[..., :1])
        points_2d = torch.cat([points_2d,ones],dim=-1)
        post_rots = post_rots.reshape(bs,N,1,3,3).repeat(1,1,num_query,1,1)
        points_2d = post_rots.matmul(points_2d.unsqueeze(-1)).squeeze(-1)  # must point -1 otherwise, will eliminate some other dims
        points_2d = points_2d + post_trans.view(bs,N,1,-1)
        points_2d = points_2d[...,:2]

        # visualize point
        if visualize_point:
            imgs = []
            points_ =[]
            img_pts_ = []
            for i in range(N):
                imgs.append(data_dict['image_inputs']['imgs'][0,i,:3].permute(1,2,0))
                points_.append(points_2d[0][i].reshape(-1,2))
            visualize_projected_points(imgs,points_)

        image_wh = torch.tensor([W, H]).repeat(B, N, 1).cuda()

        points_2d = points_2d / image_wh[:, :, None, :]
        return points_2d
    

    def get_sample_2d(self, image_feats, bs, reference_points, data_dict, visualize_point=False):
            image_inputs_dict = data_dict['image_inputs']
            camera_to_lidar = image_inputs_dict["camera_to_lidar"]
            ego_to_camera = torch.inverse(camera_to_lidar) # B,4,4,4
            intrins = image_inputs_dict['intrins'] # B,4,3,3

            x, rots, trans, post_rots, post_trans = \
                (image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], 
                image_inputs_dict['post_rots'], image_inputs_dict['post_trans'])
            
            _,_,_,H,W = x.shape

            reference_points_3d = reference_points.clone()
            reference_points = reference_points.clone()

            reference_points[..., 0:1] = reference_points[..., 0:1] * \
                (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]  # points * H * grid == points * pc_range
            reference_points[..., 1:2] = reference_points[..., 1:2] * \
                (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            reference_points[..., 2:3] = reference_points[..., 2:3] * \
                (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            reference_points = torch.cat(
                (reference_points, torch.ones_like(reference_points[..., :1])), -1)

            B, num_query = reference_points.size()[:2]
            reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, 1, self.num_cam, 1, 1).unsqueeze(-1) # B,4,num_query,4,1
            

            ego_to_camera = ego_to_camera.view(
                B, self.num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
            intrins = intrins.view(
                B, self.num_cam, 1, 3, 3).repeat(1, 1, num_query, 1, 1)

            reference_points_cam = torch.matmul(ego_to_camera.to(torch.float32),
                                                reference_points.to(torch.float32)).squeeze(-1)  # B,K,num_query,3

            reference_points_cam_2d =  reference_points_cam[...,:3] # B,K,num_query,3

            reference_points_cam_2d = reference_points_cam_2d.unsqueeze(-1)

            points_2d = torch.matmul(intrins,reference_points_cam_2d).squeeze(-1) # B,K,num_query,3

            dist = points_2d[...,2].clone()

            # 左上角是坐标系的原点
            points_2d = points_2d[..., :2] / torch.clamp(
                points_2d[..., 2:3], min=1e-5
            )
            
            this_mask = (dist > 1e-5)
            
            # for postprocess in image
            ones = torch.ones_like(points_2d[..., :1])
            points_2d = torch.cat([points_2d,ones],dim=-1)
            post_rots = post_rots.reshape(1,bs,self.num_cam,1,3,3)
            points_2d = post_rots.matmul(points_2d.unsqueeze(-1)).squeeze(-1)  # must point -1 otherwise, will eliminate some other dims
            points_2d = points_2d + post_trans.view(1,bs,self.num_cam,1,-1)
            points_2d = points_2d[...,:2] 

            if visualize_point:
                imgs = []
                points_ =[]
                img_pts_ = []
                for i in range(self.num_cam):
                    imgs.append(data_dict['image_inputs']['imgs'][0,i,:3].permute(1,2,0))
                    points_.append(points_2d[0][i].reshape(-1,2))
                visualize_projected_points(imgs,points_)

            image_wh = torch.tensor([W, H]).cuda()
            points_2d = points_2d / image_wh

            points_2d = (points_2d - 0.5) * 2  # to [-1, +1]   do we really need this ? 

            this_mask = (this_mask & (points_2d[..., 0] > -1.0)
                            & (points_2d[..., 0] < 1.0)
                            & (points_2d[..., 1] > -1.0)
                            & (points_2d[..., 1] < 1.0) 
                            )
            this_mask = this_mask.view(B, self.num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
            this_mask = torch.nan_to_num(this_mask)  
            # sampling features for each point 
            sampled_feats = []
            for lvl, feat in enumerate(image_feats):
                B, N, C, H, W = feat.size()
                feat = feat.view(B*N, C, H, W)
                reference_points_cam_lvl = points_2d.view(B*N, num_query, 1, 2)
                sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
                sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
                sampled_feats.append(sampled_feat)
            sampled_feats = torch.stack(sampled_feats, -1)
            sampled_feats = sampled_feats.view(B, C, num_query, self.num_cam,  1, len(image_feats))
            return reference_points_3d, sampled_feats, this_mask




class MultiheadAttention_(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention_, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module',
                              UserWarning)

            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


def multi_head_attention_forward(query,  # type: Tensor
                                 key,  # type: Tensor
                                 value,  # type: Tensor
                                 embed_dim_to_check,  # type: int
                                 num_heads,  # type: int
                                 in_proj_weight,  # type: Tensor
                                 in_proj_bias,  # type: Tensor
                                 bias_k,  # type: Optional[Tensor]
                                 bias_v,  # type: Optional[Tensor]
                                 add_zero_attn,  # type: bool
                                 dropout_p,  # type: float
                                 out_proj_weight,  # type: Tensor
                                 out_proj_bias,  # type: Tensor
                                 training=True,  # type: bool
                                 key_padding_mask=None,  # type: Optional[Tensor]
                                 need_weights=True,  # type: bool
                                 attn_mask=None,  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,  # type: Optional[Tensor]
                                 k_proj_weight=None,  # type: Optional[Tensor]
                                 v_proj_weight=None,  # type: Optional[Tensor]
                                 static_k=None,  # type: Optional[Tensor]
                                 static_v=None,  # type: Optional[Tensor]
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.zeros((attn_mask.size(0), 1),
                                                   dtype=attn_mask.dtype,
                                                   device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
    
class MultiheadAttention(nn.Module):
    '''
    reference from sparse4D
    '''
    def __init__(self,cfg, identify=True):
        super(MultiheadAttention, self).__init__()

        self.embed_dims = cfg['embed_dims']
        self.num_heads = cfg['num_heads']
        self.batch_first = cfg['batch_first']

        self.attn_drop = cfg['dropout']

        self.attn = nn.MultiheadAttention(self.embed_dims, self.num_heads, self.attn_drop,self.batch_first)

        self.proj_drop = nn.Dropout(self.attn_drop)
        self.dropout_layer = nn.Dropout(self.attn_drop)

        self.identify = identify

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):

        if key is None:
            # self-attention
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            
        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        if self.identify:
            out = identity + self.dropout_layer(self.proj_drop(out))
        else:
            out = self.dropout_layer(self.proj_drop(out))

        return out



class CAMM(nn.Module):
    def __init__(self, cfg) -> None:
        super(CAMM,self).__init__()
        self.use_lidar = cfg['use_lidar']
        self.batch_first = False
        self.use_camera = cfg['use_camera']
        self.pc_range = cfg['pc_range']
        self.embed_dims = embed_dims = cfg['embed_dims']
        self.num_heads = num_heads = cfg['num_heads']
        self.num_points = num_points = cfg['num_points']
        self.lidar_levels = lidar_levels = cfg['lidar_level']
        self.camera_level = camera_level = cfg['camera_level']
        self.num_cam = cfg['num_cam']
        self.im2col_step = 64

        if 'fuse_method' in cfg:
            self.fuse_method = cfg['fuse_method']
        else:
            self.fuse_method = 'add'

        if self.use_lidar:
            self.sampling_offsets = nn.Linear(
                embed_dims, num_heads * lidar_levels * num_points * 2)
            self.attention_weights = nn.Linear(embed_dims,
                                            num_heads * lidar_levels * num_points)
            self.value_proj = nn.Linear(embed_dims, embed_dims)

            # self.output_proj = nn.Linear(embed_dims, embed_dims)

            self.output_proj = nn.Sequential(
                nn.Linear(self.embed_dims,self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(),
            )

        if 'use_deformable' in cfg:
            self.use_deformable = cfg['use_deformable']
        else:
            self.use_deformable = False
        if self.use_camera:
            if self.use_deformable:
                self.weights_fc = nn.Linear(self.embed_dims, self.num_heads * self.camera_level)
                self.attn_drop = 0.0
                self.camera_encoder = nn.Sequential(
                    *linear_relu_ln(embed_dims, 1, 2, 12)
                )
            else:
                self.img_attention_weights = nn.Linear(embed_dims,
                                           self.num_cam*camera_level)
                # self.img_output_proj = nn.Linear(embed_dims, embed_dims)
                self.img_output_proj = nn.Sequential(
                    nn.Linear(self.embed_dims,self.embed_dims),
                    nn.LayerNorm(self.embed_dims),
                    nn.ReLU(),
                )

                self.position_encoder = nn.Sequential(
                    nn.Linear(3, self.embed_dims), 
                    nn.LayerNorm(self.embed_dims),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embed_dims, self.embed_dims), 
                    nn.LayerNorm(self.embed_dims),
                    nn.ReLU(inplace=True),
                )

                self.weight_dropout = nn.Dropout(0.0)
                constant_init(self.img_attention_weights, val=0., bias=0.)
        # *2 or not 
        if self.fuse_method == 'add':
            self.fused_embed = self.embed_dims
        elif self.fuse_method == 'cat':
            self.fused_embed = self.embed_dims * 2

        self.modality_fusion_layer = nn.Sequential(
            nn.Linear(self.fused_embed, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=False),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
        )    
        self.dropout = nn.Dropout(0.1)
        self.init_weights()


    def _get_weights(self, q_feature, data_dict=None):
        bs, num_query = q_feature.shape[:2]
        if self.camera_encoder is not None:
            if data_dict['image_inputs']['camera_to_lidar'].shape == 3:
                camera_embed = self.camera_encoder(data_dict['image_inputs']['camera_to_lidar'][:,:,:3].reshape(bs,self.num_cam,-1))
            else:
                camera_embed = self.camera_encoder(
                    data_dict['image_inputs']['camera_to_lidar'][:, :,:,:3].reshape(bs, self.num_cam, -1))
            q_feature = q_feature[:, :, None] + camera_embed[:, None]
        weights = (
            self.weights_fc(q_feature)
            .reshape(bs, num_query, -1, self.num_heads)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_query,
                self.num_cam,
                self.camera_level,
                self.num_heads,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_query, self.num_cam, 1, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights
    

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.lidar_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        if self.use_lidar:
            constant_init(self.sampling_offsets, 0.)
            self.sampling_offsets.bias.data = grid_init.view(-1)
            constant_init(self.attention_weights, val=0., bias=0.)
            xavier_init(self.value_proj, distribution='uniform', bias=0.)
            

    def forward(self,
            query,
            key=None,
            value=None,
            identity=None,
            query_pos=None,
            key_padding_mask=None,
            reference_points=None,
            spatial_shapes=None,
            level_start_index=None,
            pts_feats=None,
            img_feats=None,
            data_dict=None,
            ):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
            
        if not self.batch_first: # False
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        
        bs, num_query, _ = query.shape


        if self.use_camera and img_feats is not None:
            if not self.use_deformable:
                bs_with_agents, num_cams, C,_,_ = img_feats[0].shape

                img_attention_weights = self.img_attention_weights(query).view(
                    bs, 1, num_query, self.num_cam, 1, self.camera_level)
                
                reference_points_3d, img_output, mask = self.get_sample_2d(img_feats, bs_with_agents, reference_points, data_dict)
                img_output = torch.nan_to_num(img_output)
                mask = torch.nan_to_num(mask)

                img_attention_weights = self.weight_dropout(img_attention_weights.sigmoid()) * mask
                img_output = img_output * img_attention_weights
                # output (B, emb_dims, num_query)
                img_output = img_output.sum(-1).sum(-1).sum(-1)
                # output (num_query, B, emb_dims)
                img_output = img_output.permute(0, 2, 1)

                img_output = self.img_output_proj(img_output) # BS,900,C
            else:
                # deformable attention for image feature. 
                weights = self._get_weights(
                    query, data_dict
                )
                
                # TODO: sample reference 2d points
                points_2d = (
                    self.project_points(
                        reference_points,
                        data_dict,
                    ).reshape(bs, num_query, self.num_cam, 2)
                )   # project 3D keypoints into 2D image feature ,  bs,900*13,6,2

                img_output = DAF.apply(
                    *img_feats, points_2d, weights).reshape(bs, num_query,self.embed_dims)

        else:
            img_output = None

        if self.use_lidar and pts_feats is not None:
            value = pts_feats
            bs, num_value, _ = value.shape
            
            assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

            value = self.value_proj(value)
            if key_padding_mask is not None:
                value = value.masked_fill(key_padding_mask[..., None], 0.0)
            value = value.view(bs, num_value, self.num_heads, -1)
            sampling_offsets = self.sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.lidar_levels, self.num_points, 2)
            attention_weights = self.attention_weights(query).view(
                bs, num_query, self.num_heads, self.lidar_levels * self.num_points)
            attention_weights = attention_weights.softmax(-1)

            attention_weights = attention_weights.view(bs, num_query,
                                                    self.num_heads,
                                                    self.lidar_levels,
                                                    self.num_points)
            ref_points = reference_points.unsqueeze(2).expand(-1, -1, self.lidar_levels, -1)
            
            # ref_points = reference_points
            ref_points = ref_points[..., :2]
            if ref_points.shape[-1] == 2:
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                sampling_locations = ref_points[:, :, None, :, None, :] \
                    + sampling_offsets \
                    / offset_normalizer[None, None, None, :, None, :]
            else:
                raise ValueError(
                    f'Last dim of reference_points must be'
                    f' 2, but get {reference_points.shape[-1]} instead.')
            if ((IS_CUDA_AVAILABLE and value.is_cuda)
                    or (IS_MLU_AVAILABLE and value.is_mlu)):
                output = MultiScaleDeformableAttnFunction.apply( 
                    value, spatial_shapes, level_start_index, sampling_locations,
                    attention_weights, self.im2col_step)
            else:
                output = multi_scale_deformable_attn_pytorch(
                    value, spatial_shapes, sampling_locations, attention_weights)

            pts_output = self.output_proj(output)
        else:
            pts_output = None

        
        if self.use_camera and self.use_lidar and img_output is not None and pts_output is not None:
            if self.fuse_method == 'add':
                output = img_output*pts_output + img_output + pts_output
            elif self.fuse_method == 'cat':
                output = torch.cat((img_output, pts_output), dim=2)
            output = self.modality_fusion_layer(output)
        elif img_output is not None:
            output = img_output
        else:
            output = pts_output
        output = output.permute(1, 0, 2)
        return self.dropout(output) + identity


    def project_points(self, key_points, data_dict, visualize_point=False):
        key_points = key_points.clone()

        key_points[..., 0:1] = key_points[..., 0:1] * \
            (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]  # points * H * grid == points * pc_range
        key_points[..., 1:2] = key_points[..., 1:2] * \
            (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        key_points[..., 2:3] = key_points[..., 2:3] * \
            (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        
        bs, num_query = key_points.shape[:2]
        image_inputs_dict = data_dict['image_inputs']
        x, rots, trans, intrins, post_rots, post_trans = \
            (image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'],
             image_inputs_dict['post_rots'], image_inputs_dict['post_trans'])
        _,_,_,H,W = x.shape
        B,N,_ = trans.shape

        points = key_points.reshape(bs,1,num_query,-1).repeat(1,N,1,1)

        homogeneous_points = torch.cat(
                (points, torch.ones_like(points[..., :1])), -1).unsqueeze(-1) # B,4,num_query,4,1
        
        
        camera_to_lidar = image_inputs_dict["camera_to_lidar"]
        ego_to_camera = torch.inverse(camera_to_lidar).reshape(bs,N,1,4,4)
        points_cam = torch.matmul(ego_to_camera,homogeneous_points)
        points_cam =  points_cam[...,:3,:] / points_cam[...,3:4,:]

        intrins = intrins.reshape(B,N,1,3,3)
        points_2d = torch.matmul(intrins,points_cam).squeeze(-1)  # must point -1 otherwise, will eliminate some other dims
        # points_2d = []

        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )

        # another method
        # points = key_points.reshape(bs,1,num_anchor,num_pts,-1).repeat(1,N,1,1,1)  # + post_trans.view(bs,N,1,1,-1)
        # points = points.unsqueeze(-1)
        # ones = torch.ones_like(points[..., :1, :])
        # homogeneous_points = torch.cat([points, ones], dim=4)
        #
        # ext_matrix = torch.inverse(camera_to_lidar)[...,:3, :4].reshape(bs,N,1,1,3,4)
        # img_pts = (intrins @ ext_matrix @ homogeneous_points).squeeze()
        # # b,4,900,17,3
        # depth = img_pts[...,2:3]
        # img_pts = img_pts[...,:2] / torch.clamp(
        #     depth, min=1e-6
        # )
        # ones = torch.ones_like(img_pts[..., :1])
        # img_pts = torch.cat([img_pts,ones],dim=-1)
        # post_rots = post_rots.reshape(bs,N,1,1,3,3).repeat(1,1,num_anchor,num_pts,1,1)
        # img_pts = post_rots.matmul(img_pts.unsqueeze(-1)).squeeze()
        # img_pts = img_pts + post_trans.view(bs,N,1,1,-1)
        # img_pts = img_pts[...,:2]


        # for postprocess in image
        ones = torch.ones_like(points_2d[..., :1])
        points_2d = torch.cat([points_2d,ones],dim=-1)
        post_rots = post_rots.reshape(bs,N,1,3,3).repeat(1,1,num_query,1,1)
        points_2d = post_rots.matmul(points_2d.unsqueeze(-1)).squeeze(-1)  # must point -1 otherwise, will eliminate some other dims
        points_2d = points_2d + post_trans.view(bs,N,1,-1)
        points_2d = points_2d[...,:2]

        # visualize point
        if visualize_point:
            imgs = []
            points_ =[]
            img_pts_ = []
            for i in range(N):
                imgs.append(data_dict['image_inputs']['imgs'][0,i,:3].permute(1,2,0))
                points_.append(points_2d[0][i].reshape(-1,2))
            visualize_projected_points(imgs,points_)

        image_wh = torch.tensor([W, H]).repeat(B, N, 1).cuda()

        points_2d = points_2d / image_wh[:, :, None, :]
        return points_2d
    

    def get_sample_2d(self, image_feats, bs, reference_points, data_dict, visualize_point=False):
            image_inputs_dict = data_dict['image_inputs']
            camera_to_lidar = image_inputs_dict["camera_to_lidar"]
            ego_to_camera = torch.inverse(camera_to_lidar) # B,4,4,4
            intrins = image_inputs_dict['intrins'] # B,4,3,3

            x, rots, trans, post_rots, post_trans = \
                (image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], 
                image_inputs_dict['post_rots'], image_inputs_dict['post_trans'])
            
            _,_,_,H,W = x.shape

            reference_points_3d = reference_points.clone()
            reference_points = reference_points.clone()

            reference_points[..., 0:1] = reference_points[..., 0:1] * \
                (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]  # points * H * grid == points * pc_range
            reference_points[..., 1:2] = reference_points[..., 1:2] * \
                (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            reference_points[..., 2:3] = reference_points[..., 2:3] * \
                (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            reference_points = torch.cat(
                (reference_points, torch.ones_like(reference_points[..., :1])), -1)

            B, num_query = reference_points.size()[:2]
            reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, 1, self.num_cam, 1, 1).unsqueeze(-1) # B,4,num_query,4,1
            

            ego_to_camera = ego_to_camera.view(
                B, self.num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
            intrins = intrins.view(
                B, self.num_cam, 1, 3, 3).repeat(1, 1, num_query, 1, 1)

            reference_points_cam = torch.matmul(ego_to_camera.to(torch.float32),
                                                reference_points.to(torch.float32)).squeeze(-1)  # B,K,num_query,3

            reference_points_cam_2d =  reference_points_cam[...,:3] # B,K,num_query,3

            reference_points_cam_2d = reference_points_cam_2d.unsqueeze(-1)

            points_2d = torch.matmul(intrins,reference_points_cam_2d).squeeze(-1) # B,K,num_query,3

            dist = points_2d[...,2].clone()

            # 左上角是坐标系的原点
            points_2d = points_2d[..., :2] / torch.clamp(
                points_2d[..., 2:3], min=1e-5
            )
            
            this_mask = (dist > 1e-5)
            
            # for postprocess in image
            ones = torch.ones_like(points_2d[..., :1])
            points_2d = torch.cat([points_2d,ones],dim=-1)
            post_rots = post_rots.reshape(1,bs,self.num_cam,1,3,3)
            points_2d = post_rots.matmul(points_2d.unsqueeze(-1)).squeeze(-1)  # must point -1 otherwise, will eliminate some other dims
            points_2d = points_2d + post_trans.view(1,bs,self.num_cam,1,-1)
            points_2d = points_2d[...,:2] 

            if visualize_point:
                imgs = []
                points_ =[]
                img_pts_ = []
                for i in range(self.num_cam):
                    imgs.append(data_dict['image_inputs']['imgs'][0,i,:3].permute(1,2,0))
                    points_.append(points_2d[0][i].reshape(-1,2))
                visualize_projected_points(imgs,points_)

            image_wh = torch.tensor([W, H]).cuda()
            points_2d = points_2d / image_wh

            points_2d = (points_2d - 0.5) * 2  # to [-1, +1]   do we really need this ? 

            this_mask = (this_mask & (points_2d[..., 0] > -1.0)
                            & (points_2d[..., 0] < 1.0)
                            & (points_2d[..., 1] > -1.0)
                            & (points_2d[..., 1] < 1.0) 
                            )
            this_mask = this_mask.view(B, self.num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
            this_mask = torch.nan_to_num(this_mask)  
            # sampling features for each point 
            sampled_feats = []
            for lvl, feat in enumerate(image_feats):
                B, N, C, H, W = feat.size()
                feat = feat.view(B*N, C, H, W)
                reference_points_cam_lvl = points_2d.view(B*N, num_query, 1, 2)
                sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
                sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
                sampled_feats.append(sampled_feat)
            sampled_feats = torch.stack(sampled_feats, -1)
            sampled_feats = sampled_feats.view(B, C, num_query, self.num_cam,  1, len(image_feats))
            return reference_points_3d, sampled_feats, this_mask
    

# plt.显示图像，左上角是坐标系的原点
def visualize_projected_points(images, points_2d):
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    H, W, C = images[0].shape
    save_dir = "/mnt/sdb/public/data/yangk/result/comm/vis/"

    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    for img_idx, img_tensor in enumerate(images):
        # 确保图像是 NumPy 数组
        img = img_tensor.cpu().numpy()

        # 如果图像是 (C, H, W)，转换为 (H, W, C)
        if img.shape[0] in [1, 3]:  # 灰度图或RGB图
            img = np.transpose(img, (1, 2, 0))

        # 处理彩色通道
        if img.shape[2] == 1:  # 灰度图
            img = np.squeeze(img, axis=2)
        elif img.shape[2] == 3:  # RGB图
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 确保图像像素值在0到255之间，并转换为 uint8 类型
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # 保存原图
        original_img_path = os.path.join(save_dir, f"original_image_{img_idx}.png")
        cv2.imwrite(original_img_path, img)

        # 画点
        for pt in points_2d[img_idx].detach().cpu().numpy():
            x, y = int(pt[0]), int(pt[1])
            if x > W or x < 0 or y > H or y < 0:
                continue
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # 绿色点

        # 保存添加了点的图像
        points_img_path = os.path.join(save_dir, f"points_image_{img_idx}.png")
        cv2.imwrite(points_img_path, img)

        # 显示图像
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Image {img_idx}')
        plt.axis('off')
        plt.show()



class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention with mask and confidence adjustments.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.dropout_layer = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, query, key, value, attn_mask=None):
        # Flatten the input if necessary (from 3D to 2D)
        if len(query.shape) == 3:
            batch_size = query.shape[0]
            seq_len = query.shape[1]
            query = query.reshape(-1, query.shape[-1])
            key = key.reshape(-1, key.shape[-1])
            value = value.reshape(-1, value.shape[-1])
        else:
            batch_size = 1
            seq_len = query.shape[0]

        # Compute dot-product attention scores
        score = torch.matmul(query, key.transpose(-2, -1)) / self.sqrt_dim 

        if attn_mask is not None:
            score = score + attn_mask
        # Compute attention weights
        attn = F.softmax(score, dim=-1)

        # Apply dropout to attention weights
        attn = self.dropout_layer(attn)

        # Compute the context vector as the weighted sum of the values
        context = torch.matmul(attn, value)  # B, N, C

        # Reshape context back to the original batch and sequence length
        context = context.reshape(batch_size, seq_len, query.shape[-1])
        
        # context = query + self.dropout_layer(self.proj_drop(context))
        return context



class ScaledDotProductAttention_KNN(nn.Module):
    """
    Scaled Dot-Product Attention with mask and confidence adjustments.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention_KNN, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.dropout_layer = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, query, key, value=None, attn_mask=None):
        '''
        query: 900, 1, C
        key: 900, KNN, C
        '''

        if value is None:
            value = key
        # Compute dot-product attention scores
        score = torch.matmul(query, key.transpose(-2, -1)) / self.sqrt_dim 

        if attn_mask is not None:
            score = score + attn_mask
        # Compute attention weights
        attn = F.softmax(score, dim=-1)

        # Apply dropout to attention weights
        attn = self.dropout_layer(attn)

        # Compute the context vector as the weighted sum of the values
        context = torch.matmul(attn, value)  # 900, 1 , C
        
        context = query + self.dropout_layer(self.proj_drop(context))
        
        return context
    



##  Top-K Selective Attention (TTSA)
class TopkAttention(nn.Module):
    def __init__(self, dim=128, num_heads=4, bias=0.):
        super(TopkAttention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))


        self.q_lin = nn.Linear(dim, dim, bias=bias)
        self.k_lin = nn.Linear(dim, dim, bias=bias)
        self.v_lin = nn.Linear(dim, dim, bias=bias)


        self.project_out = nn.Linear(dim, dim, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, q, k=None, v=None):
        '''
        q: 1, 900, C
        k,v: None or 1,900,C
        '''

        b, num_query, _ = q.shape

        if k is None and v is None:
            k = v = q
        else:
            assert k is not None, "key is None !!!!"
            v = k 
        identity = q

        q = self.q_lin(q)
        k = self.k_lin(k)
        v = self.v_lin(v)
        q = q.permute(0,2,1)
        k = k.permute(0,2,1)
        v = v.permute(0,2,1)

        q = rearrange(q, 'b (head c) n -> b head n c', head=self.num_heads, n=num_query)  #  1, 4, 32, 900 
        k = rearrange(k, 'b (head c) n -> b head n c', head=self.num_heads, n=num_query)
        v = rearrange(v, 'b (head c) n -> b head n c', head=self.num_heads, n=num_query)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, _, C = q.shape  # C=30=180（dim）/6（num_head）

        mask1 = torch.zeros(b, self.num_heads, num_query, num_query, device=q.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, num_query, num_query, device=q.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, num_query, num_query, device=q.device, requires_grad=False)
        # mask4 = torch.zeros(b, self.num_heads, num_query, num_query, device=q.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # b h n n

        index = torch.topk(attn, k=100, dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        # print(111, mask1.scatter_(-1, index, 1.))

        index = torch.topk(attn, k=300, dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=600, dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        # index = torch.topk(attn, k=num_query, dim=-1, largest=True)[1]
        # mask4.scatter_(-1, index, 1.)
        # attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))
        
        attn4 = attn

        attn1 = attn1.softmax(dim=-1)  # [1 6 30 30]
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v) 
        out2 = (attn2 @ v) 
        out3 = (attn3 @ v) 
        out4 = (attn4 @ v) 

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head n c -> b n (head c)', head=self.num_heads, n=num_query)

        out = self.project_out(out)

        out = torch.sum(out, dim=0, keepdim=True) # 1,900,128

        out = out + identity[0:1]
        return out

if __name__ == '__main__':
    attn = TopkAttention()
    q = torch.rand(2, 900, 128)
    k = torch.rand(2, 900, 128)
    output = attn(q, k)

    print(output.shape) # 1, num_query, C