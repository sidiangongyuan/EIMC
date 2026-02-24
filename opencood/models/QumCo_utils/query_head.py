import copy
import random
import time
from typing import List, Optional, Tuple, Union
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


from .query_transformer import QUERY_TransFormer
from ..sub_modules.detr_module import PositionEmbeddingSine
from .utils import inverse_sigmoid

try:
    from ..ops import DeformableAggregationFunction as DAF
except:
    DAF = None    

def bias_init_with_prob(prior_prob: float) -> float:
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
    

class queryfusion(nn.Module):
    def __init__(self,args, use_deformable_func=False):
        super(queryfusion,self).__init__()
        
        # base, just cat two bev features and multi-agent fusion.
        # BEVFormer-like
        self.num_views = args['num_views']
        self.pc_range = args['cav_lidar_range']
        self.embed_dims = args['embed_dims']
        self.num_query = args['num_query']
        self.num_classes = args['num_classes']
        self.with_box_refine = args['with_box_refine']
        self.num_reg_fcs = 2
        self.num_layers = args['num_layer']
        self.anchor_size = args['anchor_size']
        self.use_deformable_func = use_deformable_func

        # add noise to gt 
        if 'use_dn' in args:
            self.use_dn = args['use_dn']
        else:
            self.use_dn = False
        # TODO: add to args
        self.scalar = 10 
        self.bbox_noise_scale = 1.0
        self.bbox_noise_trans = 0.0
        self.split = 0.75
        if 'img_pe' in args:
            self.img_pe = args['img_pe']
            if self.img_pe:
                self.position_dim = 3
                self.position_encoder = nn.Sequential(
                    nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
                )

        else:
            self.img_pe = False
        
        # self.tgt_embed = nn.Embedding(self.num_query, self.embed_dims)
        # TODO: 初始值有待考究
        
        self.refpoint_embed = nn.Embedding(self.num_query, self.anchor_size) 

        self.transformer = QUERY_TransFormer(args['querytransfomer'],self.anchor_size)
            
        self.positional_encoding = PositionEmbeddingSine(self.embed_dims / 2, normalize=args['positional_encoding']['normalize'])
        self.__init__weights()

    def __init__weights(self):
        # 是否随机初始化anchor box的中心位置，并固定(即不训练)它
        # self.random_refpoints_xy = random_refpoints_xy
        self.refpoint_embed.weight.data[:, :3].uniform_(0, 1)
        # 取消 x,y 的梯度，使得每张图片在输入到 Decoder 第一层时，使用的位置先验中心点(x,y)都是随机均匀分布的，
        # 而后每一层再由校正模块(bbox_embed)进行调整。
        # 这样可在一定程度上避免模型基于训练集而学到过份的归纳偏置(即过拟合)，更具泛化性
        self.refpoint_embed.weight.data[:, :3].requires_grad = False


    def position_embeding(self, img_feats, gt_depth, data_dict):
        # multi-level feature postion embedding 
        layers = len(img_feats)

        # rots: [4,4,3,3]
        # trans: [4,4,3]
        # intrins: [4,4,3,3]
        rots, trans, intrins = \
            data_dict['image_inputs']['rots'], data_dict['image_inputs']['trans'], data_dict['image_inputs']['intrins']
        
        for level in range(layers):
            B, N, C, H, W = img_feats[level].shape  # N is multi-view 
            depth_map = gt_depth[level].reshape(B,N,1,H,W)  # B*N,1,H,W
            
            depth_mask = (depth_map > 0).float()  # B*N, 1, H, W

            coords_h, coords_w = torch.meshgrid(
                torch.arange(H, device=depth_map.device).float() / (H - 1),
                torch.arange(W, device=depth_map.device).float() / (W - 1),
                indexing='ij'
            )

            # 计算x, y, d
            x = coords_w * depth_map  # B,N, 1, H, W
            y = coords_h * depth_map  # B,N, 1, H, W
            d = depth_map  # B*N, 1, H, W

            points_ = torch.stack([x, y, d], dim=-1).permute(0,1,3,4,2,5).unsqueeze(-1)  # B*N, 1, H, W, 3 -> B, N, H, W, 1, 3, 1

            combine = rots.matmul(torch.inverse(intrins))

            points_cam = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points_).squeeze(-1) # B, N, H, W, 1, 3

            points_3d = points_cam + trans.view(B, N, 1, 1, 1, 3)

            points_3d[..., 0:1] = (points_3d[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
            points_3d[..., 1:2] = (points_3d[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
            points_3d[..., 2:3] = (points_3d[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

            coords_mask = (points_3d > 1.0) | (points_3d < 0.0) 
            coords_mask = coords_mask.flatten(-2).sum(-1).reshape(B,N,-1,H,W)
            coords3d = points_3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W) 

            coords3d = inverse_sigmoid(coords3d)

            coords_position_embeding = self.position_encoder(coords3d).view(B,N,-1,H,W)

            coords_position_embeding = coords_position_embeding * coords_mask.float() * depth_mask.float()

            
            # Lg-IPE visualize 
            # import cv2 
            # import matplotlib.pyplot as plt
            # image =  data_dict['original_imgs'][0][0][0].permute(1,2,0).cpu().numpy()
            # img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.savefig(f'/mnt/sdb/public/data/yangk/result/comm/vis/original_img_0.png',bbox_inches='tight', transparent=False, dpi=400)
            # visualize_and_save_feature_map(img_feats[0][0,0,0,...].sigmoid().cpu().numpy(), 'before_position_embeding.png')
            # add a scale for visualization
            img_feats[level] = img_feats[level] + coords_position_embeding

            # visualize_and_save_feature_map(img_feats[0][0,0,0,...].sigmoid().cpu().numpy(), 'position_embeding.png')

        
        return img_feats

               
    def get_lidar_mask_pos_enc(self, mlvl_feats):

        if mlvl_feats is not None:
            batch_size = mlvl_feats[0].size(0)
            input_img_h, input_img_w = mlvl_feats[0].shape[-2:]
            img_masks = mlvl_feats[0].new_zeros(
                (batch_size, input_img_h, input_img_w))

            mlvl_masks = []
            mlvl_positional_encodings = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(img_masks[None],
                                size=feat.shape[-2:]).to(torch.bool).squeeze(0))
                mlvl_positional_encodings.append(
                    self.positional_encoding(mlvl_masks[-1])) # sine pe
        else:
            mlvl_masks = None
            mlvl_positional_encodings = None
        
        return mlvl_masks, mlvl_positional_encodings
    

    def prepare_for_dn(self, batch_size, reference_points, data_dict):
        # only add in training phase
        if self.training:
            targets = data_dict['label_dict']['object_target'] # list for each batch
            batch_size = len(targets) 
            labels = [(torch.zeros(len(t))).cuda() for t in targets] # only need zero 

            known = [(torch.ones(len(t))).cuda() for t in targets]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known) 
            known_num = [len(t) for t in targets] # gt 数量
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([item for t in targets for item in t]).reshape(-1,7)
            batch_idx = torch.cat([torch.full((len(t), ), i) for i, t in enumerate(targets)])
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # add noise
            groups = min(self.scalar, self.num_query // max(known_num)) # 分成几组
            known_indice = known_indice.repeat(groups, 1).view(-1)
            known_labels = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_labels_raw = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(groups, 1).view(-1)
            known_bboxs = boxes.repeat(groups, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = sum(self.num_classes)

            single_pad = int(max(known_num)) # max gt 
            pad_size = int(single_pad * groups) # pad for each bs
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(groups)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)
            
             # 去噪任务 & 匹配任务 的 queries 总数
            tgt_size = pad_size + self.num_query
            # # (i,j) = True 代表 i 不可見 j
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            attn_mask[pad_size:, :pad_size] = True

            # reconstruct cannot see each other
            for i in range(groups):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == groups - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'known_labels_raw': known_labels_raw,
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            
        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None
                                           
        return padded_reference_points, attn_mask, mask_dict
    
    def forward(self,pts_feats, img_feats=None, data_dict=None):

        if img_feats is not None: 
            gt_depths = []
            if len(img_feats) == 2:
                gt_depths = img_feats[1]
                img_feats = img_feats[0]

            bs_with_agents,num_views,C,_,_ = img_feats[0].shape # 总共N个车，每个车有num_views个图片，C,H,W
            assert self.num_views == num_views, "error for multi-views"
            # _,_,H,W = pts_feats.shape
            
        if pts_feats is not None:
            mlvl_masks, mlvl_positional_encodings = self.get_lidar_mask_pos_enc(pts_feats)
        else:
            mlvl_masks = None
            mlvl_positional_encodings = None

        if self.img_pe and img_feats is not None:
            if len(gt_depths) != 0:
                # gt depth :   ours
                img_feats = self.position_embeding(img_feats, gt_depths, data_dict)
            else:
                # petr 
                pass 

        refanchor = self.refpoint_embed.weight      # nq, 3

        tgt_embed = refanchor.new_zeros(self.num_query, self.embed_dims)           # nq, 256
        
        # tgt_embed = self.tgt_embed.weight           # nq, 256
        query_embeds = torch.cat((tgt_embed, refanchor), dim=1)

        if self.use_dn:
            reference_points, attn_mask, mask_dict = self.prepare_for_dn(pts_feats[0].shape[0], refanchor, data_dict)
        else:
            attn_mask = None

        if self.use_deformable_func:
            img_feats = DAF.feature_maps_format(img_feats)
            
        predictions, boxes, dirs = self.transformer(
                    pts_feats,
                    img_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    attn_mask,
                    data_dict,
            )
        
        return predictions, boxes, dirs