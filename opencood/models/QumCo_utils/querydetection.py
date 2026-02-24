import torch.nn as nn
import torch
import torch
import torch.nn as nn
import numpy as np
from mmengine.model import xavier_init, bias_init_with_prob
from .utils import inverse_sigmoid
from torch.nn import functional as F
from mmcv.cnn import Linear,Scale

X, Y, Z, W, L, H, SIN_YAW, COS_YAW = list(range(8))
YAW = 6

def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class QueryDetection(nn.Module):
    def __init__(
        self,
        embed_dims=128,
        output_dim=8,
        num_cls=1,
        with_cls_branch=True,
        return_dir = False,
        pc_range=None,
    ):
        super(QueryDetection, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.return_dir = return_dir
        self.pc_range = pc_range

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )

        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, self.num_cls),
            )
        if self.return_dir:
            self.dir_layers =  nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, 2),
            )

        self.reg_layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(self.embed_dims, 8),
        )

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        query: torch.Tensor,
        reference_points,
        return_cls=True,
        lid=None,
    ):
        bbox = self.reg_layers(query)
        assert reference_points.shape[-1] == 3
        
        bbox[..., 0:3] = (bbox[..., 0:3] + inverse_sigmoid(reference_points[..., 0:3])).sigmoid()  # 归一化

        new_reference_points_output = bbox[..., 0:3].clone()
        # 实际位置
        bbox[...,0:1] = (bbox[...,0:1]*(self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]) # 实际位置
        bbox[...,1:2] = (bbox[...,1:2]*(self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        bbox[...,2:3] = (bbox[...,2:3]*(self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
        yaw = torch.atan2(bbox[:, :, SIN_YAW], bbox[:, :, COS_YAW])
        box = torch.cat(
            [
                bbox[..., 0:3],
                bbox[:, :, [W, L, H]].exp(),
                yaw[:, :, None],
            ],
            dim=-1,
        )
        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(query)
        else:
            cls = None
        if self.return_dir:
            dir = self.dir_layers(query)
        else:
            dir = None
        
        # vis query proposals 
        # caution: here box and cls is not only ego 
        # vis_bev_proposals(box[0:1], cls[0:1], [-76.8, -51.2, -3, 76.8, 51.2, 1],lid=lid)

        return  cls, box, dir, new_reference_points_output