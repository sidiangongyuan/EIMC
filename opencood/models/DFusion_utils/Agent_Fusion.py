import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from mmcv.cnn import ConvModule
from opencood.models.QumCo_utils.attention import MultiheadAttention, MultiheadAttention_
from opencood.models.fuse_modules.fusion_in_one import ScaledDotProductAttention
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.utils.transformation_utils import regroup

from mmengine.model import BaseModule
from mmcv.cnn import Linear, Scale, build_activation_layer, build_norm_layer
from torch.nn import Sequential
from mmcv.cnn.bricks.drop import build_dropout


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



class Instane2SceneAtt(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1, args=None):
        super().__init__()

        self.multihead_attn = MultiheadAttention_(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if args is not None:
            self.FFN = AsymmetricFFN(args['ffn'])
        else:
            self.FFN = None
            

    def forward(self, query, key, query_scene, bs, H, W, attn_mask=None):
        # (bev_feats, x_ins, scene_feats, bs, self.bev_size)
        """
        :param query: B C N
        :param query_pos: B N 2
        :return:
        """

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        query2 = self.multihead_attn(query=query, key=key, value=key, attn_mask=attn_mask)[0]

        query = query + self.dropout(query2) # N,B,C

        if self.FFN is not None:
            query = self.FFN(query.permute(1, 2, 0)).permute(2, 0, 1)
        
        query = self.norm(query).permute(1, 2, 0) # B, C, N
        query_ins = query.reshape(bs, query.shape[1], H, W)
        attention_weights = torch.matmul(query_scene, query_ins.transpose(2, 3))
        attention_weights = F.softmax(attention_weights, dim=-1)
        attended_query_ins = torch.matmul(attention_weights, query_ins)

        return_feats = query_scene + attended_query_ins

        return return_feats
    


class AgentFusion(nn.Module):
    def __init__(self, args):
        super(AgentFusion, self).__init__()
        embed_dims = args['embed_dims']
        self.ms_layers = args['ms_layers']
        self.num_class = args['num_class']
        self.instance_num = args['instance_num']
        self.nms_kernel_size = 3

        self.down_convs = nn.ModuleList([
            DownsampleConv(args[f'down_head{i + 1}']) for i in range(self.ms_layers)
        ])
        self.instance_attns = nn.ModuleList()
        self.scene_attns = nn.ModuleList()
        self.conv_ins = nn.ModuleList()
        self.conv_scene = nn.ModuleList()
        self.conv_heatmap = nn.ModuleList()
        self.heatmap_head_1 = nn.ModuleList()
        self.heatmap_head_2 = nn.ModuleList()
        self.heatmap_head_3 = nn.ModuleList()
        self.query_pos_embed = nn.ModuleList()
        self.norm_layer = nn.ModuleList()
        self.topk_cross_attention = nn.ModuleList()
        self.depths = args['depth']

        if 'topk' in args:
            self.topk = args['topk']
        else:
            self.topk = 20
        print('topk:', self.topk)
        current_dims = embed_dims
        for i in range(self.ms_layers):
            self.conv_ins.append(self._create_conv_module(current_dims // 2, current_dims // 2))
            self.conv_scene.append(self._create_conv_module(current_dims, current_dims))
            self.conv_heatmap.append(self._create_conv_module(current_dims, current_dims // 2))
            self.heatmap_head_1.append(self._create_conv_module(current_dims // 2, current_dims // 4))
            self.heatmap_head_2.append(self._create_conv_module(current_dims // 4, current_dims // 4))
            self.heatmap_head_3.append(
                nn.Conv2d(current_dims // 4, self.num_class, kernel_size=3, stride=1, padding=1)
            )
            self.instance_attns.append(
                MultiheadAttention(args[f'mha{i+1}'], False)
            )
            self.scene_attns.append(
                Instane2SceneAtt(d_model=current_dims, nhead=8, dropout=0.1)
            )
            self.query_pos_embed.append(PositionEmbeddingLearned(2, current_dims))
            self.norm_layer.append(nn.LayerNorm(current_dims))
            self.topk_cross_attention.append(MultiheadAttention(args[f'mha_topk{i+1}'], False))
            current_dims = 2*current_dims
        
        self.msfusion = AdvancedMultiScaleFusionWithDeconv(embed_dims, 2*embed_dims, 4*embed_dims)
    def _create_conv_module(self, in_channels, out_channels, kernel_size=3, padding=1):
        return ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
        )
    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base
    
    
    def forward(self, features, data_dict, affine_matrix=None):
        # features: B,C,H,W
        features_new = features.clone()
        B, L = affine_matrix.shape[:2]
        record_len = data_dict['record_len']
        out_layer = []
        heatmap_layer = []
        for layer in range(self.ms_layers):
            features_new = self.down_convs[layer](features_new) # HWC  H/2 W/2 C*2    H/4,W/4 C*4
            bs,C,H,W = features_new.shape 
            bev_pos = self.create_2D_grid(W,H).to(features_new.device)
            bev_pos = bev_pos.expand(bs, -1, -1)

            split_features_new = regroup(features_new, record_len)
            regroup_feature_new = []
            # Todo: warp_affine_simple
            for b1 in range(B):
                ego = 0
                num_cav = record_len[b1]
                regroup_feature_new.append(warp_affine_simple(split_features_new[b1], affine_matrix[b1, ego, :num_cav,], (H, W)))
                
            regroup_feature_new = torch.cat(regroup_feature_new, dim=0) # N,C,H,W

            # features_ = features_new.permute(0, 3, 2, 1).contiguous()
            ins_heatmap = self.conv_heatmap[layer](regroup_feature_new.clone().detach()) # B,C,H,W
            ins_heatmap=self.heatmap_head_1[layer](ins_heatmap)
            ins_heatmap=self.heatmap_head_2[layer](ins_heatmap)
            ins_heatmap=self.heatmap_head_3[layer](ins_heatmap)

            heatmap = ins_heatmap.detach().sigmoid() # B,1,H,W
            for ag in range(bs):
                unique_prefix = generate_unique_name(f"before_{ag}")
                visualize_heatmap(
                    heatmap[ag], 
                    path=f'/mnt/sdb/public/data/yangk/result/heal/vis/{unique_prefix}.png', flip_vertical=True
                )

            if bs != 1:
                diff_heatmap = heatmap[0:1].expand_as(heatmap[1:]) - heatmap[1:]

                # x_scene: B, C, H, W (提取特征的目标)
                bs_minus_1, _, _, _ = diff_heatmap.shape
                for bm in range(bs_minus_1):
                    unique_prefix = generate_unique_name(f"diff_{bm}")
                    visualize_heatmap(diff_heatmap[bm],path=f'/mnt/sdb/public/data/yangk/result/heal/vis/{unique_prefix}.png', flip_vertical=True)
                top_k = self.topk

                # Step 1: 对每个 heatmap 找到 top-k 索引
                diff_heatmap_flat = diff_heatmap.view(bs_minus_1, -1)  # 展平为 (B-1, H*W)
                top_k_indices = diff_heatmap_flat.argsort(dim=-1, descending=False)[..., :top_k]  # B-1, 10

            padding = self.nms_kernel_size // 2

            local_max = torch.zeros_like(heatmap)
            # equals to nms radius = voxel_size * out_size_factor * kenel_size
            local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
            local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
            heatmap = heatmap * (heatmap == local_max)
            heatmap = heatmap.view(bs, heatmap.shape[1], -1)

            instance_num = self.instance_num[layer]
            top_proposals = heatmap.view(bs, -1).argsort(dim=-1, descending=True)[..., : instance_num]
            top_proposals_index = top_proposals % heatmap.shape[-1]
            query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)
            query_pos_new = torch.zeros_like(query_pos)
            query_pos_new[..., 0] = query_pos[..., 1]
            query_pos_new[..., 1] = query_pos[..., 0] # bs, 100 , 2
            reference_points = query_pos_new.sigmoid()
            query_pos_embed = self.query_pos_embed[layer](reference_points).permute(0, 2, 1)

            x_scene = self.conv_scene[layer](regroup_feature_new) # B,C,H,W
            x_scene_flatten = x_scene.view(bs, x_scene.shape[1], -1) # B,C,N
            x_ins = x_scene_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, x_scene.shape[1], -1),
                                                    dim=-1) 

            if bs != 1:
                x_scene_0_flatten = x_scene_flatten[0] #
                # 遍历 B-1 的样本，从第 0 个样本提取特征
                features_from_0 = []
                for b in range(bs_minus_1):
                    # 从第 0 个样本根据 top-k 索引提取
                    features_0 = x_scene_0_flatten[:, top_k_indices[b]].t()  # 10, C
                    features_from_0.append(features_0)
                features_from_0 = torch.stack(features_from_0, dim=0)  # B-1, 10, C
                # 从各自样本的 heatmap 提取特征
                features_self = []
                for b in range(bs_minus_1):
                    # 从各自样本提取
                    features_b = x_scene_flatten[b + 1, :, top_k_indices[b]].t()  # 10, C
                    features_self.append(features_b)
                features_self = torch.stack(features_self, dim=0)  # B-1, 10, C

                updated_features_from_0  = self.topk_cross_attention[layer](features_from_0, key = features_self)

                x_scene_flatten_updated = x_scene_flatten.clone()
                # 遍历 B-1 的样本，更新第 0 个样本的特征
                for b in range(bs_minus_1):
                    # 获取 top-k 索引
                    indices_b = top_k_indices[b]  # 10
                    # 将更新的特征写回到第 0 个样本的 flatten 特征中
                    x_scene_flatten_updated[0, :, indices_b] = updated_features_from_0[b].t()

                # Step 2: 复原 x_scene
                x_scene_updated = x_scene_flatten_updated.view(x_scene.shape)  # B, C, H, W
            else:
                x_scene_updated = x_scene


            out = x_scene_updated
            #refinement :
            split_x = regroup(x_ins, record_len)
            splite_querypos = regroup(query_pos_embed, record_len)
            features_in = regroup(regroup_feature_new, record_len)
            split_x_scene = regroup(x_scene_updated, record_len)
            out = []

            for b in range(B):
                N = record_len[b]
                xx_ins = split_x[b].permute(0,2,1) # B,N,C
                q_pos = splite_querypos[b]
                f_b = features_in[b]
                x_scene_b = split_x_scene[b]
                f_b_ = f_b.reshape(N, features_in[b].shape[1], -1) # N,C,-1
                xx_ins_input = xx_ins.reshape(1,-1,C)

                xx_ins_input = self.instance_attns[layer](xx_ins_input, query_pos=q_pos.reshape(1,-1,C)).permute(0,2,1).expand(N,-1,-1)

                out_feature = self.scene_attns[layer](f_b_, xx_ins_input, x_scene_b, N, H, W)

                out.append(out_feature)
            out = torch.cat(out, dim=0)


            out_layer.append(out)
            heatmap_layer.append(ins_heatmap)
        
        res = self.msfusion(out_layer[0], out_layer[1], out_layer[2])
        
        return res, heatmap_layer




class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding
    



class AdvancedMultiScaleFusionWithDeconv(nn.Module):
    def __init__(self, c1, c2, c3):
        super(AdvancedMultiScaleFusionWithDeconv, self).__init__()

        # 1x1 Conv for channel alignment
        self.conv1 = nn.Conv2d(c1, c1, kernel_size=1)
        self.conv2 = nn.Conv2d(c2, c1, kernel_size=1)
        self.conv3 = nn.Conv2d(c3, c1, kernel_size=1)

        # Transposed Convolutions for upsampling
        self.deconv2 = nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(c1, c1, kernel_size=8, stride=4, padding=2)

        # Multi-scale Context Extraction
        self.context = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        # Spatial Attention Module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )

        # Channel Attention Module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 4, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 4, c1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        )

    def forward(self, f1, f2, f3):
        # Align channels
        f1_aligned = self.conv1(f1)  # B, C1, H1, W1
        f2_aligned = self.conv2(f2)  # B, C1, H2, W2
        f3_aligned = self.conv3(f3)  # B, C1, H3, W3

        # Upsample using transposed convolutions
        f2_upsampled = self.deconv2(f2_aligned)  # Match spatial size with f1
        f3_upsampled = self.deconv3(f3_aligned)  # Match spatial size with f1

        # Combine features
        fused = f1_aligned + f2_upsampled + f3_upsampled

        # Apply multi-scale context extraction
        context_features = self.context(fused)

        # Apply spatial attention
        spatial_weights = self.spatial_attention(context_features)
        spatially_weighted = fused * spatial_weights

        # Apply channel attention
        channel_weights = self.channel_attention(spatially_weighted)
        channel_weighted = spatially_weighted * channel_weights

        # Final feature fusion
        fused_output = self.fusion(channel_weighted)

        return fused_output
    


def visualize_heatmap(data, cmap='hot', overlay=None, alpha=0.5, path=None, flip_vertical=False):
    """
    可视化热图的函数，颜色条高度与主图严格对齐。
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # 数据转换与验证
    if isinstance(data, torch.Tensor):
        data = data.squeeze().detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        data = data.squeeze()
    else:
        raise TypeError("输入数据必须是 torch.Tensor 或 numpy.ndarray")


    # 数据翻转处理 -------------------------------------------------
    if flip_vertical:
        data = np.flipud(data)  # 上下翻转数据
        if overlay is not None:
            overlay = np.flipud(overlay)  # 同时翻转底图


    # 动态色系选择
    has_negative = np.any(data < 0)
    if has_negative:
        vmin, vmax = np.min(data), np.max(data)
        cmap = 'coolwarm' if cmap == 'hot' else cmap
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        norm = plt.Normalize(vmin=0, vmax=1)

    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    
    # 绘制底图
    if overlay is not None:
        overlay = overlay.squeeze()
        if isinstance(overlay, torch.Tensor):
            overlay = overlay.detach().cpu().numpy()
        ax.imshow(overlay, cmap='gray', interpolation='bilinear')

    # 绘制热图
    im = ax.imshow(data, cmap=cmap, norm=norm, 
                  alpha=alpha if overlay else 1.0,
                  interpolation='bilinear')


    # 创建等高的颜色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # 宽度5%，间距0.1英寸
    cbar = fig.colorbar(im, cax=cax)

    # 优化颜色条样式
    cbar.ax.yaxis.set_tick_params(width=1, labelsize=12)  # 刻度线粗细和字体
    if has_negative:
        cbar.set_ticks([vmin, 0, vmax])
    else:
        cbar.set_ticks([0, 0.5, 1])

    # 隐藏坐标轴并保存
    ax.axis('off')
    plt.tight_layout(pad=0.1)
    if path:
        plt.savefig(path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()



def generate_unique_name(prefix: str) -> str:
    import time
    import os
    timestamp = time.strftime("%Y%m%d%H%M%S")
    microsecond = f"{int(time.time() * 1e6 % 1e6):06d}"
    random_suffix = f"{os.getpid()}_{os.urandom(2).hex()}"  # 进程ID + 随机字节
    return f"{timestamp}_{microsecond}_{random_suffix}_{prefix}"