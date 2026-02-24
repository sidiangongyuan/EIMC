# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.nx, self.ny, self.nz = model_cfg['grid_size']  # [704, 200, 1] 

    #FIXME: only support 2D 
    # def forward(self, batch_dict):
    #     """ 将生成的pillar按照坐标索引还原到原空间中
    #     Args:
    #         pillar_features:(M, 64)
    #         coords:(M, 4) 第一维是batch_index

    #     Returns:
    #         batch_spatial_features:(4, 64, H, W)
            
    #         |-------|
    #         |       |             |-------------|
    #         |       |     ->      |  *          |
    #         |       |             |             |
    #         | *     |             |-------------|
    #         |-------|

    #         Lidar Point Cloud        Feature Map
    #         x-axis up                Along with W 
    #         y-axis right             Along with H

    #         Something like clockwise rotation of 90 degree.

    #     """
    #     pillar_features, coords = batch_dict['pillar_features'], batch_dict[
    #         'voxel_coords']
    #     batch_spatial_features = []
    #     batch_size = coords[:, 0].max().int().item() + 1

    #     for batch_idx in range(batch_size):
    #         spatial_feature = torch.zeros(
    #             self.num_bev_features, # 64
    #             self.nz * self.nx * self.ny, 
    #             dtype=pillar_features.dtype,
    #             device=pillar_features.device)
    #         # batch_index的mask
    #         batch_mask = coords[:, 0] == batch_idx
    #         # 根据mask提取坐标
    #         this_coords = coords[batch_mask, :] # (batch_idx_voxel,4)  # zyx order, x in [0,706], y in [0,200]
    #         # 这里的坐标是b,z,y和x的形式,且只有一层，因此计算索引的方式如下
    #         indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
    #         # 转换数据类型
    #         indices = indices.type(torch.long)
    #         # 根据mask提取pillar_features
    #         pillars = pillar_features[batch_mask, :] # (batch_idx_voxel,64)
    #         pillars = pillars.t() # (64,batch_idx_voxel)
    #         # 在索引位置填充pillars
    #         spatial_feature[:, indices] = pillars
    #         # 将空间特征加入list,每个元素为(64, self.nz * self.nx * self.ny)
    #         batch_spatial_features.append(spatial_feature) 

    #     batch_spatial_features = \
    #         torch.stack(batch_spatial_features, 0)
    #     batch_spatial_features = \
    #         batch_spatial_features.view(batch_size, self.num_bev_features *
    #                                     self.nz, self.ny, self.nx) # It put y axis(in lidar frame) as image height. [..., 200, 704]
    #     batch_dict['spatial_features'] = batch_spatial_features

    #     return batch_dict

    def forward(self, batch_dict):
        """
        Scatter pillar features into spatial feature maps (2D from pillar_features, 3D from occ_voxel_features).

        Args:
            batch_dict: Dictionary containing:
                - pillar_features: (M, C), features for each voxel.
                - voxel_coords: (M, 4), coordinates for each voxel (batch_idx, z, y, x).
                - (optional) occ_voxel_features: (N, C), features for each 3D voxel.
                - (optional) occ_voxel_coords: (N, 4), coordinates for each 3D voxel.

        Returns:
            batch_dict with:
                - spatial_features: (B, C, ny, nx) for 2D features.
                - (optional) 3d_spatial_features: (B, C, nz, ny, nx) if nz > 1.
        """
        # Extract 2D features
        pillar_features = batch_dict['pillar_features']
        coords = batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1  # Number of batches

        for batch_idx in range(batch_size):
            # Initialize spatial feature grid for 2D (z=1)
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nx * self.ny,  # Flattened grid for 2D
                dtype=pillar_features.dtype,
                device=pillar_features.device
            )

            # Mask for the current batch
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]  # Shape: (num_voxels_in_batch, 4)

            # Calculate 2D indices (z=1 is implied)
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]  # y * nx + x
            indices = indices.type(torch.long)

            # Gather pillar features for current batch
            pillars = pillar_features[batch_mask, :]  # Shape: (num_voxels_in_batch, C)
            pillars = pillars.t()  # Shape: (C, num_voxels_in_batch)

            # Fill features into spatial grid
            spatial_feature[:, indices] = pillars

            # Append to batch list
            batch_spatial_features.append(spatial_feature)

        # Stack all batches
        batch_spatial_features = torch.stack(batch_spatial_features, 0)

        # Reshape to (B, C, ny, nx) for 2D spatial features
        spatial_features_2d = batch_spatial_features.view(
            batch_size, self.num_bev_features, self.ny, self.nx
        )
        batch_dict['spatial_features'] = spatial_features_2d

        # If nz > 1 and occ_voxel_features exist, process 3D features
        if self.nz > 1 and 'occ_features' in batch_dict:
            occ_features = batch_dict['occ_features']
            occ_coords = batch_dict['occ_voxel_coords']
            batch_3d_spatial_features = []

            for batch_idx in range(batch_size):
                # Initialize spatial feature grid for 3D
                spatial_feature_3d = torch.zeros(
                    self.num_bev_features,
                    self.nz * self.nx * self.ny,  # Flattened grid for 3D
                    dtype=occ_features.dtype,
                    device=occ_features.device
                )

                # Mask for the current batch
                batch_mask = occ_coords[:, 0] == batch_idx
                this_coords = occ_coords[batch_mask, :]  # Shape: (num_voxels_in_batch, 4)

                # Calculate 3D indices (z, y, x)
                indices = (this_coords[:, 1] * self.ny * self.nx +  # z * ny * nx
                        this_coords[:, 2] * self.nx +            # y * nx
                        this_coords[:, 3])                      # x
                indices = indices.type(torch.long)

                # Gather occ features for current batch
                pillars = occ_features[batch_mask, :]  # Shape: (num_voxels_in_batch, C)
                pillars = pillars.t()  # Shape: (C, num_voxels_in_batch)

                # Fill features into 3D spatial grid
                spatial_feature_3d[:, indices] = pillars

                # Append to batch list
                batch_3d_spatial_features.append(spatial_feature_3d)

            # Stack all batches
            batch_3d_spatial_features = torch.stack(batch_3d_spatial_features, 0)

            # Reshape to (B, C, nz, ny, nx) for 3D spatial features
            spatial_features_3d = batch_3d_spatial_features.view(
                batch_size, self.num_bev_features, self.nz, self.ny, self.nx
            )
            batch_dict['3d_spatial_features'] = spatial_features_3d

        return batch_dict
