import torch
import torch.nn.functional as F
import torch.nn as nn

from opencood.models.fuse_modules.fusion_in_one import ScaledDotProductAttention
from opencood.models.sub_modules.torch_transformation_utils import  warp_affine_simple_3d
from opencood.utils.transformation_utils import regroup


class Mocc(nn.Module):
    def __init__(self, args):
        super(Mocc, self).__init__()
        self.attn = ScaledDotProductAttention(args['occ_dims'])


    def forward(self, occ, data_dict, affine_matrix):
        _,H,W,D,C = occ.shape
        record_len = data_dict['record_len']
        bs = len(record_len) 
        split_x = regroup(occ, record_len)
        out = []
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        for b in range(bs):
            N = record_len[b]
            t_matrix = affine_matrix[b][:N, :N, :, :]
            x = split_x[b].permute(0,4,3,1,2) # N,H,W,D,C --> N,C,D,H,W
            x = warp_affine_simple_3d(x, t_matrix[0, :, :, :], (D, H, W))
            x = x.reshape(-1,N,C)
            h = self.attn(x,x,x)
            h = h.reshape(N,H,W,D,C)
            out.append(h[0:1])
            out.append(split_x[b][1:,...])
        
        out = torch.cat(out,dim=0)
        
        return out
            


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Mocc(nn.Module):
#     def __init__(self, args):
#         super(Mocc, self).__init__()
#         self.topk = 100
#         self.embed_dim = args['occ_dims']
        
#         # 交叉注意力层
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=self.embed_dim,
#             num_heads=4,
#             batch_first=True
#         )

#     def _select_topk_voxels(self, pred, k):
#         """
#         选择每个agent的topk体素
#         pred: [N, C, D, H, W] 体素预测置信度
#         k: 要选择的体素数
#         返回：
#         topk_indices: [N, k] 展平后的体素索引
#         """
#         N = pred.shape[0]
#         pred_flat = pred.view(N, -1)  # [N, D*H*W]
        
#         # 获取每个agent的topk索引
#         topk_values, topk_indices = torch.topk(pred_flat, k, dim=1)  # [N, k]
#         return topk_indices

#     def forward(self, occ_pred, occ, data_dict, affine_matrix):
#         _, H, W, D, C = occ.shape
#         record_len = data_dict['record_len']
#         bs = len(record_len)
#         split_x = regroup(occ, record_len)
#         split_pred = regroup(occ_pred, record_len)
#         out = []
        
#         for b in range(bs):
#             N = record_len[b]
#             if N == 1:
#                 out.append(split_x[b])
#                 continue
#             t_matrix = affine_matrix[b][:N, :N, :, :]
            
#             # 数据对齐
#             x = split_x[b].permute(0,4,3,1,2)  # [N, H, W, D, C] -> [N, C, D, H, W]
#             pred = split_pred[b].permute(0,4,3,1,2)  # [N, C, D, H, W]
            
#             x = warp_affine_simple_3d(x, t_matrix[0, :, :, :], (D, H, W))
#             pred = warp_affine_simple_3d(pred, t_matrix[0, :, :, :], (D, H, W))

#             # ------------------- 新增部分开始 -------------------
#             # Step 1: 分离ego和其他agent
#             ego_feat = x[0:1]  # [1, C, D, H, W]
#             other_feat = x[1:]  # [N-1, C, D, H, W]
#             other_pred = pred[1:]  # [N-1, C, D, H, W]

#             # Step 2: 为每个其他agent选择topk体素
#             B_other = other_pred.shape[0]
#             topk_indices = self._select_topk_voxels(other_pred, self.topk)  # [N-1, k]

#             # Step 3: 收集其他agent的topk体素特征
#             other_feat_flat = other_feat.view(B_other, C, -1)  # [N-1, C, D*H*W]
#             selected_feats = []
#             for i in range(B_other):
#                 selected_feats.append(other_feat_flat[i, :, topk_indices[i]].T)  # [k, C]
#             neighbor_feats = torch.cat(selected_feats, dim=0)  # [(N-1)*k, C]

#             # Step 4: 准备ego的所有体素作为query
#             ego_feat_flat = ego_feat.view(1, C, -1).permute(0, 2, 1)  # [1, D*H*W, C]

#             # Step 5: 交叉注意力
#             attn_output, _ = self.cross_attn(
#                 query=ego_feat_flat,  # [1, D*H*W, C]
#                 key=neighbor_feats.unsqueeze(0),  # [1, (N-1)*k, C]
#                 value=neighbor_feats.unsqueeze(0)  # [1, (N-1)*k, C]
#             )
            
#             # Step 6: 恢复ego特征形状
#             updated_ego = attn_output.view(1, C, D, H, W)  # [1, C, D, H, W]
#             # ------------------- 新增部分结束 -------------------

#             # 拼接结果 (保持其他agent不变)
#             out.append(updated_ego)
#             if N > 1:
#                 out.append(x[1:])
#             else:
#                 out.append(x)
        
#         out = torch.cat(out, dim=0)
#         return out
            

