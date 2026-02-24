# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Transform points to voxels using sparse conv library
"""
import sys

import numpy as np
import torch
from icecream import ic

from opencood.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor


class SpVoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(SpVoxelPreprocessor, self).__init__(preprocess_params, train)
        self.spconv = 1
        try:
            # spconv v1.x
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
        except:
            # spconv v2.x
            from cumm import tensorview as tv
            from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
            self.tv = tv
            self.spconv = 2

        self.lidar_range = self.params['cav_lidar_range']
        self.voxel_size = self.params['args']['voxel_size']
        self.max_points_per_voxel = self.params['args']['max_points_per_voxel']

        if train:
            self.max_voxels = self.params['args']['max_voxel_train']
        else:
            self.max_voxels = self.params['args']['max_voxel_test']

        grid_size = (np.array(self.lidar_range[3:6]) - np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        self.voxel_size_bev = [self.voxel_size[0], self.voxel_size[1], self.lidar_range[5] - self.lidar_range[2]]
        
        self.use_occ = False
        if 'use_occ' in self.params['args']:
            self.use_occ = self.params['args']['use_occ']
        # self.use_occ = self.params['args']['use_occ']
        # Use sparse conv library to generate voxel
        if self.spconv == 1:
            self.voxel_generator = VoxelGenerator(
                voxel_size=self.voxel_size,
                point_cloud_range=self.lidar_range,
                max_num_points=self.max_points_per_voxel,
                max_voxels=self.max_voxels
            )
            if self.grid_size[2] > 1 and self.use_occ:
                self.voxel_generator_bev = VoxelGenerator(
                    voxel_size=[self.voxel_size[0], self.voxel_size[1], self.lidar_range[5] - self.lidar_range[2]],
                    point_cloud_range=self.lidar_range,
                    max_num_points=self.max_points_per_voxel,
                    max_voxels=self.max_voxels
                )
        else:
            self.voxel_generator = VoxelGenerator(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.lidar_range,
                max_num_points_per_voxel=self.max_points_per_voxel,
                num_point_features=4,
                max_num_voxels=self.max_voxels
            )
            if self.grid_size[2] > 1 and self.use_occ:
                self.voxel_generator_bev = VoxelGenerator(
                    vsize_xyz=self.voxel_size_bev,
                    coors_range_xyz=self.lidar_range,
                    max_num_points_per_voxel=self.max_points_per_voxel,
                    num_point_features=4,
                    max_num_voxels=self.max_voxels
                )

    def preprocess(self, pcd_np):
        data_dict = {}
        
        if self.grid_size[2] == 1:  # z=1: merge voxel and occ
            if self.spconv == 1:
                voxel_output = self.voxel_generator.generate(pcd_np)
            else:
                pcd_tv = self.tv.from_numpy(pcd_np)
                voxel_output = self.voxel_generator.point_to_voxel(pcd_tv)

            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], \
                    voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output

            if self.spconv == 2:
                voxels = voxels.numpy()
                coordinates = coordinates.numpy()
                num_points = num_points.numpy()

            # Merge voxel and occ as one set
            data_dict['voxel_features'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points

        elif self.use_occ:  # z>1: voxel and occ are separate
            if self.spconv == 1:
                voxel_output = self.voxel_generator_bev.generate(pcd_np)
                occ_output = self.voxel_generator.generate(pcd_np)
            else:
                pcd_tv = self.tv.from_numpy(pcd_np)
                voxel_output = self.voxel_generator_bev.point_to_voxel(pcd_tv)
                occ_output = self.voxel_generator.point_to_voxel(pcd_tv)

            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], \
                    voxel_output['num_points_per_voxel']
                occ_voxels, occ_coordinates, occ_num_points = \
                    occ_output['voxels'], occ_output['coordinates'], \
                    occ_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
                occ_voxels, occ_coordinates, occ_num_points = occ_output

            if self.spconv == 2:
                voxels = voxels.numpy()
                coordinates = coordinates.numpy()
                num_points = num_points.numpy()
                occ_voxels = occ_voxels.numpy()
                occ_coordinates = occ_coordinates.numpy()
                occ_num_points = occ_num_points.numpy()

            # Separate voxel and occ
            data_dict['voxel_features'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points

            data_dict['occ_voxel_features'] = occ_voxels
            data_dict['occ_voxel_coords'] = occ_coordinates
            data_dict['occ_voxel_num_points'] = occ_num_points
        else:
            assert self.grid_size[2] == 1, "not use occ! but z is not 1"

        return data_dict
    
    # def __init__(self, preprocess_params, train):
    #     super(SpVoxelPreprocessor, self).__init__(preprocess_params,
    #                                               train)
    #     self.spconv = 1
    #     try:
    #         # spconv v1.x
    #         from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
    #     except:
    #         # spconv v2.x
    #         from cumm import tensorview as tv
    #         from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
    #         self.tv = tv
    #         self.spconv = 2
    #     self.lidar_range = self.params['cav_lidar_range']
    #     self.voxel_size = self.params['args']['voxel_size']
    #     self.max_points_per_voxel = self.params['args']['max_points_per_voxel']

    #     if train:
    #         self.max_voxels = self.params['args']['max_voxel_train']
    #     else:
    #         self.max_voxels = self.params['args']['max_voxel_test']

    #     grid_size = (np.array(self.lidar_range[3:6]) -
    #                  np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
    #     self.grid_size = np.round(grid_size).astype(np.int64)


    #     # use sparse conv library to generate voxel
    #     if self.spconv == 1:
    #         self.voxel_generator = VoxelGenerator(
    #             voxel_size=self.voxel_size,
    #             point_cloud_range=self.lidar_range,
    #             max_num_points=self.max_points_per_voxel,
    #             max_voxels=self.max_voxels
    #         )
    #     else:
    #         self.voxel_generator = VoxelGenerator(
    #             vsize_xyz=self.voxel_size,
    #             coors_range_xyz=self.lidar_range,
    #             max_num_points_per_voxel=self.max_points_per_voxel,
    #             num_point_features=4,
    #             max_num_voxels=self.max_voxels
    #         )

    # def preprocess(self, pcd_np):
    #     data_dict = {}
    #     if self.spconv == 1:
    #         voxel_output = self.voxel_generator_bev.generate(pcd_np)
    #     else:
    #         pcd_tv = self.tv.from_numpy(pcd_np)
    #         voxel_output = self.voxel_generator_bev.point_to_voxel(pcd_tv)
    #     if isinstance(voxel_output, dict):
    #         voxels, coordinates, num_points = \
    #             voxel_output['voxels'], voxel_output['coordinates'], \
    #             voxel_output['num_points_per_voxel']
    #     else:
    #         voxels, coordinates, num_points = voxel_output

    #     if self.spconv == 2:
    #         voxels = voxels.numpy()
    #         coordinates = coordinates.numpy()
    #         num_points = num_points.numpy()

    #     data_dict['voxel_features'] = voxels
    #     data_dict['voxel_coords'] = coordinates # int 
    #     data_dict['voxel_num_points'] = num_points

    #     return data_dict

    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            List or dictionary.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """

        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit('Batch has too be a list or a dictionarn')

    @staticmethod
    def process_voxel_features(batch, key_prefix):
        """
        Process voxel features for either 'voxel' or 'occ' based on the key_prefix.

        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represents a single frame.
        key_prefix : str
            Prefix for the keys to process (e.g., 'voxel' or 'occ_voxel').

        Returns
        -------
        dict
            Processed tensor features, coordinates, and num_points.
        """
        features = []
        num_points = []
        coords = []

        for i in range(len(batch)):
            if f'{key_prefix}_features' in batch[i]:
                features.append(batch[i][f'{key_prefix}_features'])
                num_points.append(batch[i][f'{key_prefix}_num_points'])
                frame_coords = batch[i][f'{key_prefix}_coords']
                coords.append(
                    np.pad(frame_coords, ((0, 0), (1, 0)),
                        mode='constant', constant_values=i)
                )

        if features:
            return {
                f'{key_prefix}_features': torch.from_numpy(np.concatenate(features)),
                f'{key_prefix}_coords': torch.from_numpy(np.concatenate(coords)),
                f'{key_prefix}_num_points': torch.from_numpy(np.concatenate(num_points)),
            }
        return {}

    @staticmethod
    def collate_batch_list(batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represents a single frame.

        Returns
        -------
        dict
            Updated lidar batch.
        """
        # Process 'voxel' fields
        result = SpVoxelPreprocessor.process_voxel_features(batch, 'voxel')

        # Process 'occ' fields if they exist
        occ_result = SpVoxelPreprocessor.process_voxel_features(batch, 'occ_voxel')
        result.update(occ_result)

        return result

    @staticmethod
    def collate_batch_dict(batch: dict):
        """
        Collate batch if the batch is a dictionary,
        eg: {'voxel_features': [feature1, feature2...., feature n]}

        Parameters
        ----------
        batch : dict

        Returns
        -------
        dict
            Updated lidar batch.
        """
        def process_dict_features(batch, key_prefix):
            if f'{key_prefix}_features' in batch:
                features = torch.from_numpy(np.concatenate(batch[f'{key_prefix}_features']))
                num_points = torch.from_numpy(np.concatenate(batch[f'{key_prefix}_num_points']))
                coords = batch[f'{key_prefix}_coords']

                padded_coords = []
                for i in range(len(coords)):
                    padded_coords.append(
                        np.pad(coords[i], ((0, 0), (1, 0)),
                            mode='constant', constant_values=i)
                    )
                coords = torch.from_numpy(np.concatenate(padded_coords))
                return {
                    f'{key_prefix}_features': features,
                    f'{key_prefix}_coords': coords,
                    f'{key_prefix}_num_points': num_points,
                }
            return {}

        # Process 'voxel' fields
        result = process_dict_features(batch, 'voxel')

        # Process 'occ' fields if they exist
        occ_result = process_dict_features(batch, 'occ_voxel')
        result.update(occ_result)

        return result