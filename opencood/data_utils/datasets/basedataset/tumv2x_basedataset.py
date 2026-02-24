# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
from collections import OrderedDict
import cv2
import h5py
import torch
import numpy as np
from functools import partial
from torch.utils.data import Dataset
from PIL import Image
import random
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.camera_utils import load_camera_data, load_intrinsic_DAIR_V2X
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import tfm_to_pose, rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor

class TUMV2XBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.post_processor.generate_gt_bbx = self.post_processor.generate_gt_bbx_by_iou
        if 'data_augment' in params: # late and early
            self.data_augmentor = DataAugmentor(params['data_augment'], train)
        else: # intermediate
            self.data_augmentor = None

        if 'clip_pc' in params['fusion']['args'] and params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False

        if 'train_params' not in params or 'max_cav' not in params['train_params']:
            self.max_cav = 2
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        assert self.load_depth_file is False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                                    else self.generate_object_center_camera

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        if self.train:
            split_dir = params['root_dir']
            split = 'train'
        else:
            split_dir = params['validate_dir']
            split = 'val'
        self.split = split
        self.root_dir = params['data_dir']

        self.split_info = read_json(split_dir)
        samples_len = len(self.split_info['filenames'])
        self.co_data = OrderedDict()

        for i in range(samples_len):
            veh_frame_id = self.split_info['filenames'][i]
            self.co_data[veh_frame_id] = read_json(self.root_dir + '/' + split + 
                                                   '/labels_point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered/' + 
                                                   veh_frame_id + '_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered.json') 

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False
        
        self.dir_mapping = {
            's110_camera_basler_south1_8mm': f'/mnt/sdb/public/data/yangk/dataset/TUMV2X/{split}/images/s110_camera_basler_south1_8mm/',
            's110_camera_basler_south2_8mm': f'/mnt/sdb/public/data/yangk/dataset/TUMV2X/{split}/images/s110_camera_basler_south2_8mm/',
            's110_camera_basler_north_8mm': f'/mnt/sdb/public/data/yangk/dataset/TUMV2X/{split}/images/s110_camera_basler_north_8mm/',
            'vehicle_camera_basler_16mm': f'/mnt/sdb/public/data/yangk/dataset/TUMV2X/{split}/images/vehicle_camera_basler_16mm/',
        }
    
    def reinitialize(self):
        pass

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.
        NOTICE!
        It is different from Intermediate Fusion and Early Fusion
        Label is not cooperative and loaded for both veh side and inf side.
        Parameters
        ----------
        idx : int
            Index given by dataloader.
        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        veh_frame_id = self.split_info['filenames'][idx]
        frame_info = self.co_data[veh_frame_id]
        data = OrderedDict()

        # ego is Inf
        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[1] = OrderedDict()
        data[1]['ego'] = False

        data[0]['params'] = OrderedDict()
        data[1]['params'] = OrderedDict()

        loaded_images_0 = []
        keys_frame = list(frame_info['openlabel']["frames"].keys())
        name_images = frame_info['openlabel']["frames"][keys_frame[0]]['frame_properties']['image_file_names']  # order: s1,s2,north,east,veh
        for name in name_images:
            # Extract the camera directory from the file name
            key = "_".join(name.split("_")[2:])[:-4]  # Remove .jpg extension
            if key == "s110_camera_basler_east_8mm":  # Skip the fourth image
                continue
            # Get the directory from the mapping
            dir_path = self.dir_mapping.get(key)
            if dir_path:
                # Construct full file path
                camera_file = f"{dir_path}{name}"
                try:
                    # Load the image and append to the list
                    image = Image.open(camera_file)
                    loaded_images_0.append(image)
                except FileNotFoundError:
                    return None
                except Exception as e:
                    print(f"Error loading image {camera_file}: {e}")
                    return None
        data[0]['camera_data'] = loaded_images_0[:-1]
        data[1]['camera_data'] = loaded_images_0[-1:]

        data[0]['params']['s1'] = OrderedDict()
        data[0]['params']['s1']['intrinsic'] = np.asarray([[1400.3096617691212, 0.0, 967.7899705163408],
                                       [0.0, 1403.041082755918, 581.7195041357244],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        
        data[0]['params']['s1']['lidar_to_image'] = np.asarray(
            [[1279.275240545117, -862.9254609474538, -443.6558546306608, -16164.33175985643],
             [-57.00793327192514, -67.92432779187584, -1461.785310749125, -806.9258947569469],
             [0.7901272773742676, 0.3428181111812592, -0.508108913898468, 3.678680419921875]], dtype=np.float32)
        
        data[0]['params']['s1']['camera_to_lidar'] = np.asarray([[0.41204962, -0.45377758, 0.7901276, 2.158825],
                                        [-0.9107832, -0.23010845, 0.34281868, -15.5765505],
                                        [0.02625162, -0.86089253, -0.5081085, 0.08758777],
                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)   # inverse of the extrinsic 
        
        # Define parameters for 's2'
        data[0]['params']['s2'] = OrderedDict()
        data[0]['params']['s2']['intrinsic'] = np.asarray([[1029.2795655594014, 0.0, 982.0311857478633],
                                                        [0.0, 1122.2781391971948, 1129.1480997238505],
                                                        [0.0, 0.0, 1.0]], dtype=np.float32)
        
        data[0]['params']['s2']['lidar_to_image'] = np.asarray([[1546.63215008, -436.92407115, -295.58362676, 1319.79271737],
                                                                    [93.20805656, 47.90351592, -1482.13403199, 687.84781276],
                                                                    [0.73326062, 0.59708904, -0.32528854, -1.30114325]], dtype=np.float32)
        data[0]['params']['s2']['camera_to_lidar'] = np.asarray([[0.6353517, -0.24219051, 0.7332613, -0.03734626],
                                                                [-0.7720766, -0.217673, 0.5970893, 2.5209506],
                                                                [0.01500183, -0.9454958, -0.32528937, 0.543223],
                                                                [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        # Define parameters for 'north'
        data[0]['params']['north'] = OrderedDict()
        data[0]['params']['north']['intrinsic'] = np.asarray([[1315.158203125, 0.0, 962.7348338975571],
                                                            [0.0, 1362.7757568359375, 580.6482296623581],
                                                            [0.0, 0.0, 1.0]], dtype=np.float32)
        
        data[0]['params']['north']['lidar_to_image'] = np.asarray([[-185.2891049687059, -1504.063395597006, -525.9215327879701, -23336.12843138125],
                                                                    [-240.2665682659353, 220.6722195428702, -1567.287260600104, 6362.243306159624],
                                                                    [0.6863989233970642, -0.4493367969989777, -0.5717979669570923, -6.750176429748535]], dtype=np.float32)
        
        data[0]['params']['north']['camera_to_lidar'] = np.asarray([[-0.56460226, -0.4583457, 0.6863989, 0.64204305],
                                                                    [-0.8248329, 0.34314296, -0.4493365, -16.182753],
                                                                    [-0.02958117, -0.81986094, -0.57179797, 1.6824605],
                                                                    [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        
        data[1]['params']['vehicle'] = OrderedDict()
        data[1]['params']['vehicle']['intrinsic'] = np.asarray([[2726.55, 0.0, 685.235],
                                           [0.0, 2676.64, 262.745],
                                           [0.0, 0.0, 1.0]], dtype=np.float32)  # 内参
        
        data[1]['params']['vehicle']['lidar_to_image'] = np.asarray([[1019.929965441548, -2613.286262078907, 184.6794570200418, 370.7180273597151],
                                         [589.8963703919744, -24.09642935106967, -2623.908527352794,
                                          -139.3143336725661],
                                         [0.9841844439506531, 0.1303769648075104, 0.1199281811714172,
                                          -0.1664766669273376]], dtype=np.float32)
        
        data[1]['params']['vehicle']['camera_to_lidar'] = np.asarray([[0.12672871, 0.12377692, 0.9841849, 0.14573078],  # 外参的逆 就是这个
                                       [-0.9912245, -0.02180046, 0.13037732, 0.19717109],
                                       [0.03759337, -0.99207014, 0.11992808, -0.02214238],
                                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        
        # load point cloud 
        name_lidar_I = frame_info['openlabel']["frames"][keys_frame[0]]['frame_properties']['point_cloud_file_names'][1]
        name_lidar_V = frame_info['openlabel']["frames"][keys_frame[0]]['frame_properties']['point_cloud_file_names'][0]
        data[0]['lidar_np'],_ = pcd_utils.read_pcd(self.root_dir + '/' + self.split + '/point_clouds/s110_lidar_ouster_south/' + name_lidar_I)
        data[1]['lidar_np'],_ = pcd_utils.read_pcd(self.root_dir + '/' + self.split + '/point_clouds/vehicle_lidar_robosense/' + name_lidar_V)
        pairwise_t_matrix = np.asarray(frame_info['openlabel']["frames"]
                                       [keys_frame[0]]['frame_properties']['transforms']
                                       ['vehicle_lidar_robosense_to_s110_lidar_ouster_south']
                                       ['transform_src_to_dst']['matrix4x4'],dtype=np.float32)
        data[0]['params']['pairwise_t_matrix'] = pairwise_t_matrix # from v to I

        object_np, mask, object_ids = self.post_processor.generate_object_center_tumv2x(
            frame_info['openlabel']["frames"][keys_frame[0]])

        data[0]['params']['labels'] = OrderedDict()
        data[0]['params']['labels']['object_np'] = object_np
        data[0]['params']['labels']['mask'] = mask
        data[0]['params']['labels']['object_ids'] = object_ids
        return data


    def __len__(self):
        return len(self.split_info['filenames'])

    def __getitem__(self, idx):
        pass


    def generate_object_center_lidar(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_all']
        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)

    def generate_object_center_camera(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_front']
        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)
                                                        
    ### Add new func for single side
    def generate_object_center_single(self,
                               cav_contents,
                               reference_lidar_pose,
                               **kwargs):
        """
        veh or inf 's coordinate. 

        reference_lidar_pose is of no use.
        """
        suffix = "_single"
        for cav_content in cav_contents:
            cav_content['params']['vehicles_single'] = \
                    cav_content['params']['vehicles_single_front'] if self.label_type == 'camera' else \
                    cav_content['params']['vehicles_single_all']
        return self.post_processor.generate_object_center_dairv2x_single(cav_contents, suffix)

    ### Add for heterogeneous, transforming the single label from self coord. to ego coord.
    def generate_object_center_single_hetero(self,
                                            cav_contents,
                                            reference_lidar_pose, 
                                            modality):
        """
        loading the object from single agent. 
        
        The same as *generate_object_center_single*, but it will transform the object to reference(ego) coordinate,
        using reference_lidar_pose.
        """
        suffix = "_single"
        for cav_content in cav_contents:
            cav_content['params']['vehicles_single'] = \
                    cav_content['params']['vehicles_single_front'] if modality == 'camera' else \
                    cav_content['params']['vehicles_single_all']
        return self.post_processor.generate_object_center_dairv2x_single_hetero(cav_contents, reference_lidar_pose, suffix)


    def get_ext_int(self, params, camera_id):
        lidar_to_camera = params["camera%d" % camera_id]['extrinsic'].astype(np.float32) # R_cw
        camera_to_lidar = np.linalg.inv(lidar_to_camera) # R_wc
        camera_intrinsic = params["camera%d" % camera_id]['intrinsic'].astype(np.float32
        )
        return camera_to_lidar, camera_intrinsic

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.
        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape
        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw
        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask