import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from opencood.models.QumCo_utils.attention import MultiheadAttention
    

class SensorFusion(nn.Module):
    def __init__(self, args):
        super(SensorFusion, self).__init__()
        self.args = args
        self.attn = MultiheadAttention(args['mha'])
        self.fuse_method = args['method']
        self.embed_dims = args['mha']['embed_dims']
        self.use_attn = False
        if 'use_attn' in args:
            self.use_attn = args['use_attn']

        if self.fuse_method == 'cat':
            self.fuse_dim = self.embed_dims * 2
            self.fusion_conv = nn.Conv2d(self.fuse_dim, self.embed_dims, kernel_size=1, stride=1, padding=0)
        elif self.fuse_method == 'add':
            self.fuse_dim = self.embed_dims

    def forward(self, cam_f, pts_f):
        '''
        cam_f: B,H,W,C
        pts_f: B,H,W,C

        return: B,H,W,C
        '''
        c1 = cam_f.shape[-1]
        c2 = pts_f.shape[-1]
        B,H,W = cam_f.shape[0:3]
        if c1 != c2:
            # 确保特征通道一致
            cam_f = nn.Conv2d(c1, self.embed_dims, kernel_size=1)(cam_f)
            pts_f = nn.Conv2d(c2, self.embed_dims, kernel_size=1)(pts_f)

        if self.fuse_method == 'cat':
            fuse_bev = torch.cat([cam_f, pts_f], dim=-1)
            fuse_bev = self.fusion_conv(fuse_bev.permute(0,3,1,2)).permute(0,2,3,1) # B,H,W,C

        elif self.fuse_method == 'add':
            fuse_bev = cam_f + pts_f
        else:
            fuse_bev = 0
            
        if self.use_attn:
            # cross-attention 
            q = pts_f.reshape(B,-1,self.embed_dims) 
            k = cam_f.reshape(B,-1,self.embed_dims)

            q_up = self.attn(q,k).reshape(B,H,W,self.embed_dims)

            fuse_bev = fuse_bev + q_up
        
        else:
            fuse_bev = fuse_bev

        return fuse_bev.permute(0,3,1,2) # B,C,H,W
