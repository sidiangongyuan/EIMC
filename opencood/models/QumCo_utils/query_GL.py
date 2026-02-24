import math
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import seaborn as sns
from .attention import ScaledDotProductAttention,MultiheadAttention
from .utils import gen_sineembed_for_position

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x


# global and local attention
class MAGL(nn.Module):
    def __init__(self,config):
        super(MAGL, self).__init__()

        self.embed_dims = config['MHA']['embed_dims']
        self.pc_range = config['pc_range']
        self.local_attn = MultiheadAttention(config['MHA'])
        # self.global_attn = MultiheadAttention(config['MHA'])
        self.iterative = config['iterative']

        self.with_pe = config['with_pe']

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

        # query_pos = query_pos.permute(1,0,2)
        # split_pos = regroup(query_pos,record_len)
        #for visualize 
        save_path = "/mnt/sdb/public/data/yangk/result/comm/vis/query_GL/"

        out = []
        for b in range(B):
            b_x = split_x[b]
            # pos_b = split_pos[b]
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
            assert neighbor_num == cav_num - 1, "error of neighbor number!!!"

            if self.with_pe:
                t_matrix_to_ego = t_matrix[:,:1] # N,1,4,4
                b_reference_points_homo = torch.cat([b_reference_points,torch.ones(cav_num,num_query,1).to(device=b_reference_points.device)],dim=-1).unsqueeze(-1)
                b_reference_points_transform = torch.matmul(t_matrix_to_ego,b_reference_points_homo).squeeze(-1)
                b_reference_points_transform = b_reference_points_transform[...,:3].sigmoid()
                bev_embeds = self.bev_embedding(gen_sineembed_for_position(b_reference_points_transform,self.embed_dims)) # refpoints in the same coord             
            for iter in range(self.iterative):
                #PE
                b_ego_expand = b_ego.expand(neighbor_num, -1, -1)
                if not self.with_pe:  # the same as self-attention
                    output = self.local_attn(b_ego_expand,b_neighbor,b_neighbor)  # N-1, 900 , 256
                else:
                    query = b_ego_expand
                    key = b_neighbor 
                    key = key
                    output = self.local_attn(query, key=key, key_pos=bev_embeds[1:])
                b_ego = torch.sum(output, dim=0, keepdim=True) # only ego 
            output_new = torch.cat([b_ego, b_neighbor],dim=0).reshape(cav_num,-1,C)
            out.append(output_new)
 
        out = torch.cat(out,dim=0)
        out = out.permute(1,0,2)
        return out

                

def generate_anchor_heatmap(x, y, scores, title='Heatmap', figsize=(6, 4), save_path=None):
    """
    根据给定的 x, y 坐标和置信度分数生成热力图。

    参数:
    x (array): x 维度的坐标数组
    y (array): y 维度的坐标数组
    scores (array): 置信度分数数组
    title (str): 图像标题
    figsize (tuple): 图像大小
    save_path (str): 保存图像的路径 (可选)

    返回:
    None
    """
    # 确保输入数据在 CPU 上
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
    
    if isinstance(y, torch.Tensor):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()
    
    if isinstance(scores, torch.Tensor):
        if scores.is_cuda:
            scores = scores.cpu()
        scores = torch.sigmoid(scores).numpy().reshape(-1)  # 确保 scores 是一维数组

    # 创建热力图数据
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=100, weights=scores, range=[[x.min(), x.max()], [y.min(), y.max()]])

    # 绘制热力图
    plt.figure(figsize=figsize)
    X, Y = np.meshgrid(xedges, yedges)
    pcm = plt.pcolormesh(X, Y, heatmap.T, cmap='inferno', shading='auto')
    pcm.set_clim(0, 1)  # 固定颜色条范围

    # 添加颜色条
    cbar = plt.colorbar(pcm, pad=0.01)
    cbar.ax.tick_params(labelsize=12)

    # 设置标题和标签
    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    # 显示图像
    plt.tight_layout()

    # 保存图像（如果提供了路径）
    if save_path:
        plt.savefig(save_path, dpi=400, transparent=False)

    plt.close()

def visualize_queries(queries, positions, save_path='output.png'):
    # 将特征和位置从GPU移动到CPU并转换为numpy数组

    queries = queries.view(-1, queries.size(-1)).detach().cpu().numpy()
    positions = positions.view(-1, positions.size(-1)).detach().cpu().numpy()

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    reduced_queries = tsne.fit_transform(queries)  # (B * 900, 2)

    # 使用位置信息的前两个维度作为 x 和 y 坐标，使用降维后的特征的第一个维度作为分数
    x = torch.tensor(positions[:, 0])
    y = torch.tensor(positions[:, 1])
    scores = torch.tensor(reduced_queries[:, 0])


    # 生成热力图
    generate_anchor_heatmap(x, y, scores, title='Query Heatmap', figsize=(6, 4), save_path=save_path)