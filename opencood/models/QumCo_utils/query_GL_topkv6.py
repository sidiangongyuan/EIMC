import math
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import seaborn as sns
from opencood.models.sub_modules.detr_module import _get_activation_fn
from .utils import gen_sineembed_for_position
from mmcv.cnn import ConvModule

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x


# global and local attention
class MAGLtopkv6(nn.Module):
    def __init__(self,config):
        super(MAGLtopkv6, self).__init__()

        self.embed_dims = config['MHA']['embed_dims']
        self.pc_range = config['pc_range']
        # self.local_attn = MultiheadAttention(config['MHA'])
        self.hidden = 512
        self.iterative = config['iterative']
        self.topk = config['topk']
        embed_dim = embed_dims = self.embed_dims
        self.with_pe = config['with_pe']

        self.heatmap = nn.Sequential(
            nn.Linear(embed_dims, 1),
        )

        if self.with_pe:
            self.bev_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims)
        )
        


    def forward(self, x, reference_points, datadict, lid, cls_head=None, query_pos=None):
        '''
        x: B,query,256
        reference_points: B,query,3 ,   [0,1] 
        '''
        x = x.permute(1,0,2)
        record_len = datadict['record_len']
        pairwise_t_matrix = datadict['pairwise_t_matrix']  # pairwise_t_matrix[i, j] is Tji, i_to_j
        B, _ = pairwise_t_matrix.shape[:2]  # L is max
        split_x = regroup(x, record_len)

        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]  # points * H * grid == points * pc_range
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        split_ref = regroup(reference_points,record_len)
        out = []
        for b in range(B):
            b_x = split_x[b]
            cav_num, num_query,C = b_x.shape
            assert cav_num == record_len[b], "error of car number!!!"
            b_reference_points = split_ref[b] # cav, query, 3
            t_matrix = pairwise_t_matrix[b][:cav_num, :cav_num, :, :].to(dtype=torch.float32)  # only ego to others.  t[i,j] --> from i to j 
            if cav_num == 1:
                out.append(b_x)
                continue
            b_ego = b_x[0:1].clone()
            b_neighbor = b_x[1:].clone()
            b_neighbor_flatten = b_neighbor.reshape(-1,C)
            neighbor_num = b_neighbor.shape[0]
            stable_b_x = b_x.clone() # N,900,256


            if cls_head is not None:
                # heat_query_neighbor = cls_head(b_neighbor_flatten) # N-1*900,C
                sparse_obj_heat = cls_head(b_x) # N,900,1
                heat_query_neighbor = sparse_obj_heat[1:].reshape(neighbor_num*num_query,-1) # N-1*900,C
            else:
                heat_query_neighbor = self.heatmap(b_neighbor_flatten) # N-1*900,C

            confidence_query_neighbor = heat_query_neighbor.sigmoid()
            topk_values, topk_indices = torch.topk(confidence_query_neighbor.squeeze(-1), self.topk, dim=-1)
            topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, C)
            topk_features = torch.gather(b_neighbor_flatten, dim=0, index=topk_indices_expanded)

            b_neighbor_topk_flatten = topk_features.view(1, -1, C)

            assert neighbor_num == cav_num - 1, "error of neighbor number!!!"

            t_matrix_to_ego = t_matrix[:,:1] # N,1,4,4
            b_reference_points_homo = torch.cat([b_reference_points,torch.ones(cav_num,num_query,1).to(device=b_reference_points.device)],dim=-1).unsqueeze(-1)
            b_reference_points_transform = torch.matmul(t_matrix_to_ego,b_reference_points_homo).squeeze(-1)
            xyz_transform = b_reference_points_transform[...,:3]
            b_reference_points_transform = b_reference_points_transform[...,:3].sigmoid()
            bev_embeds = self.bev_embedding(gen_sineembed_for_position(b_reference_points_transform,self.embed_dims)) # refpoints in the same coord         
            topk_xyz = torch.gather(xyz_transform[1:].reshape(neighbor_num*num_query,-1), dim=0, index=topk_indices.unsqueeze(-1).expand(-1, 3)) # topk,3

            ego_xyz = xyz_transform[0:1].reshape(num_query,-1)  # 900,3 / 2
            neighbor_xyz = topk_xyz # or xyz_transform[1:].reshape(neighbor_num*num_query,-1) # 900,3
            # calculate the distance
            distances = torch.cdist(ego_xyz, neighbor_xyz, p=2)  # 900, 900
            # knn 
            _, indices = torch.topk(distances, k=5, dim=-1, largest=False)  # 900, KNN
            # select the topk neighbors
            indices = indices.unsqueeze(-1).expand(-1, -1, C) # 900, KNN, C
            b_neighbor_knn_select = torch.gather(b_neighbor_topk_flatten.expand(num_query, -1, -1), 1, indices)  # 900, KNN, 256
        
            b_ego = b_ego + torch.max(b_neighbor_knn_select, dim=1, keepdim=True).values.permute(1, 0, 2)  # 900, 1, 256

            output_new = torch.cat([b_ego, b_neighbor],dim=0)
            out.append(output_new)
 
        out = torch.cat(out,dim=0)
        out = out.permute(1,0,2)
        return out

            

