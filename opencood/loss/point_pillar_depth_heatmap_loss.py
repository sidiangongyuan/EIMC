# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.loss.center_point_loss import GaussianFocalLoss, clip_sigmoid, draw_heatmap_gaussian, gaussian_2d, gaussian_radius
from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.models.DFusion_utils.Agent_Fusion import visualize_heatmap

class PointPillarDepthheatmapLoss(PointPillarLoss):
    def __init__(self, args):
        super().__init__(args)
        self.depth = args['depth']
        self.depth_weight = self.depth['weight']
        self.smooth_target = True if 'smooth_target' in self.depth and self.depth['smooth_target'] else False
        self.use_fg_mask = True if 'use_fg_mask' in self.depth and self.depth['use_fg_mask'] else False
        self.loss_cls = GaussianFocalLoss(reduction='mean')
        self.fg_weight = 3.25
        self.bg_weight = 0.25
        if self.smooth_target:
            self.depth_loss_func = FocalLoss(alpha=0.25, gamma=2.0, reduction="none", smooth_target=True)
        else:
            self.depth_loss_func = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        self.target_cfg = args['target_assigner_config']
        self.lidar_range = args['lidar_range']
        self.voxel_size = args['voxel_size']
        self.heatmap_weight = args['heatmap_weight']

    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """

        total_loss = super().forward(output_dict, target_dict, suffix)
        all_depth_loss = 0
        depth_items_list = [x for x in output_dict.keys() if x.startswith(f"depth_items{suffix}")]
        ######## heatmap supervision ######
        if 'heatmap' in output_dict:
            self.heatmap_loss(output_dict, target_dict)
        else:
            self.loss_dict['heatmap_loss'] = 0
        total_loss += self.loss_dict['heatmap_loss']
        ######## Depth Supervision ########
        for depth_item_name in depth_items_list:
            depth_item = output_dict[depth_item_name]
            if depth_item is None:
                continue
            # depth logdit: [N, D, H, W]
            # depth gt indices: [N, H, W]
            # fg_mask: [N, H, W]
            depth_logit, depth_gt_indices = depth_item[0], depth_item[1]
            depth_loss = self.depth_loss_func(depth_logit, depth_gt_indices) 
            if self.use_fg_mask:
                fg_mask = depth_item[-1]
                weight_mask = (fg_mask > 0) * self.fg_weight + (fg_mask == 0) * self.bg_weight
                depth_loss *= weight_mask

            depth_loss = depth_loss.mean() * self.depth_weight 
            all_depth_loss += depth_loss

        total_loss += all_depth_loss
        self.loss_dict.update({'depth_loss': all_depth_loss}) # no update the total loss in dict
        
        self.loss_dict.update({'total_loss':total_loss})
        
        return total_loss

    def heatmap_loss(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        # Predictions 
        box_preds = output_dict['reg_preds{}'.format(suffix)].permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

        # GTs
        bbox_center = target_dict['object_bbx_center{}'.format(suffix)].cpu().numpy()
        bbox_mask = target_dict['object_bbx_mask{}'.format(suffix)].cpu().numpy()
        batch_size = bbox_mask.shape[0]

        max_gt = int(max(bbox_mask.sum(axis=1)))
        gt_boxes3d = np.zeros((batch_size, max_gt, bbox_center[0].shape[-1]), dtype=np.float32)  # [B, max_anchor_num, 7]
        for k in range(batch_size):
            gt_boxes3d[k, :int(bbox_mask[k].sum()), :] = bbox_center[k, :int(bbox_mask[k].sum()), :]
        gt_boxes3d = torch.from_numpy(gt_boxes3d).to(box_preds.device)
        heatmap_loss_total = 0
        for i in range(len(output_dict['heatmap'])):
            targets_dict = self.assign_targets(
                gt_boxes=gt_boxes3d, out_size_factor=self.target_cfg['out_size_factor'][i]   #    [B, max_anchor_num, 7 + C ]      heatmap [2,1,h,w]  anno_boxes [2,100,8] inds [2, 100]
            )

            cls_gt =  targets_dict['heatmaps']

            pred_heatmap = output_dict['heatmap'][0][i:i+1].sigmoid()

            # visualize_heatmap(pred_heatmap[0], cmap='viridis',path='/mnt/sdb/public/data/yangk/result/heal/vis/pred_heatmap.png')
            # visualize_heatmap(cls_gt[0], cmap='viridis',path='/mnt/sdb/public/data/yangk/result/heal/vis/gt_heatmap.png')

            heatmap_loss = self.get_heatmap_loss(pred_heatmap, cls_gt)
            
            heatmap_loss_total += heatmap_loss
        self.loss_dict.update({'heatmap_loss': heatmap_loss_total})


    def get_heatmap_loss(self, pred_heatmaps, gt_heatmaps):
        num_pos = gt_heatmaps.eq(1).float().sum().item()
        cls_loss = self.loss_cls(
            pred_heatmaps,
            gt_heatmaps,
            avg_factor=max(num_pos, 1))
        cls_loss = cls_loss * self.heatmap_weight
        return cls_loss



    def assign_targets(self, gt_boxes, out_size_factor=2):
        """Generate targets.

        Args:
            gt_boxes: ( M, 7+c) box + cls   ## 这个地方函数和centerpoint-kitti 那个不太一样，这里是分开进行计算的 

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including \
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the \
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which \
                        boxes are valid.
        """
        if gt_boxes.shape[-1] == 8:
            gt_bboxes_3d, gt_labels_3d = gt_boxes[..., :-1], gt_boxes[..., -1]    # gt_box [2,14,8] batch_size * bbox_num * 8
            heatmaps, anno_boxes, inds, masks = self.get_targets_single(gt_bboxes_3d, gt_labels_3d)
        elif gt_boxes.shape[-1] == 7:
            gt_bboxes_3d = gt_boxes
            heatmaps, anno_boxes, inds, masks = self.get_targets_single(gt_bboxes_3d)

        # transpose heatmaps, because the dimension of tensors in each task is
        # different, we have to use numpy instead of torch to do the transpose.
        # heatmaps = np.array(heatmaps).transpose(1, 0).tolist()
        # heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # # heatmaps = torch.from_numpy(np.array(heatmaps))
        # # transpose anno_boxes
        # anno_boxes = np.array(anno_boxes).transpose(1, 0).tolist()
        # anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # # transpose inds
        # inds = np.array(inds).transpose(1, 0).tolist()
        # inds = [torch.stack(inds_) for inds_ in inds]
        # # transpose inds
        # masks = np.array(masks).transpose(1, 0).tolist()
        # masks = [torch.stack(masks_) for masks_ in masks]

        all_targets_dict = {
            'heatmaps': heatmaps,
            'anno_boxes': anno_boxes,
            'inds': inds,
            'masks': masks
        }
        
        return all_targets_dict


    def get_targets_single(self, gt_bbox_3d, gt_labels_3d=None, out_size_factor=2):
        
        batch_size = gt_bbox_3d.shape[0]
        device = gt_bbox_3d.device
        max_objs = self.target_cfg['max_objs']
        pc_range = self.lidar_range
        voxel_size = self.voxel_size

        grid_size = (np.array(self.lidar_range[3:6]) -
                     np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)

        feature_map_size = grid_size[:2] // out_size_factor

        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for batch in range(batch_size):
            task_boxes = gt_bbox_3d[batch, :, :]
            if not gt_labels_3d is None:
                task_classes = gt_labels_3d[batch, :]

            heatmap = gt_bbox_3d.new_zeros(    # 辅助gt_bboxes_3d的属性
                (1, feature_map_size[1],feature_map_size[0])) 

            anno_box = gt_bbox_3d.new_zeros((max_objs, 8), 
                                            dtype = torch.float32)
            
            ind = gt_bbox_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bbox_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes.shape[0], max_objs)
        
            for k in range(num_objs):
                # 计算x的heatmap坐标
                coor_x = (task_boxes[k][0] - pc_range[0]) / voxel_size[0] / out_size_factor
                coor_y = (task_boxes[k][1] - pc_range[1]) / voxel_size[1] / out_size_factor
                coor_z = (task_boxes[k][2] - pc_range[2]) / voxel_size[2] / out_size_factor
                h = task_boxes[k][3] / voxel_size[0] / out_size_factor
                w = task_boxes[k][4] / voxel_size[1] / out_size_factor
                l = task_boxes[k][5] / voxel_size[2] / out_size_factor
                rot = task_boxes[k][6]

                if h > 0 and w > 0:
                    radius = gaussian_radius(
                        (h, w),
                        min_overlap=self.target_cfg['gaussian_overlap'])
                    radius = max(self.target_cfg['min_radius'], int(radius))

                    center = torch.tensor([coor_x, coor_y],
                                        dtype=torch.float32,
                                        device=device)
                    center_int = center.to(torch.int32)   ## bbox 的中心在heatmap 中的位置

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0].item()
                            and 0 <= center_int[1] < feature_map_size[1].item()):
                        continue

                    draw_gaussian(heatmap[0], center_int, radius) 
                    
                    x, y = center_int[0], center_int[1]
                    assert (center_int[1] * feature_map_size[0] + center_int[0] <
                                        feature_map_size[0] * feature_map_size[1])
                    ind[k] = y * feature_map_size[0] + x
                    mask[k] = 1
                    # box_dim = task_boxes[k][3:6]
                    # box_dim = box_dim.log()
                    box_dim = torch.cat([h.unsqueeze(0), w.unsqueeze(0), l.unsqueeze(0)], dim=0)
                    anno_box[k] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        coor_z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                    ])   # [x,y,z, w, h, l, sin(heading), cos(heading)]

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            inds.append(ind)
            masks.append(mask)
            # import cv2; cv2.imwrite('test_{}.png'.format(batch), heatmap.cpu().numpy()[0]*255)
        heatmaps = torch.stack(heatmaps)
        anno_boxes = torch.stack(anno_boxes)
        inds = torch.stack(inds)
        masks = torch.stack(masks)
        return heatmaps, anno_boxes, inds, masks  # [B, H, W]
    
     
    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=""):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        depth_loss = self.loss_dict.get('depth_loss', 0)
        heatmap_loss = self.loss_dict.get('heatmap_loss', 0)


        print("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
              " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f || Depth Loss: %.4f || Heatmap Loss: %.4f" % (
                  epoch, batch_id + 1, batch_len, suffix,
                  total_loss, cls_loss, reg_loss, dir_loss, iou_loss, depth_loss, heatmap_loss))

        if not writer is None:
            writer.add_scalar('Regression_loss' + suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss' + suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss' + suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss' + suffix, iou_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Depth_loss' + suffix, depth_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Heatmap_loss' + suffix, heatmap_loss,
                            epoch*batch_len + batch_id)


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stability. This is no longer
          used.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha, gamma = 2.0, reduction= 'none', smooth_target = False , eps = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth_target = smooth_target
        self.eps = eps
        if self.smooth_target:
            self.smooth_kernel = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
            self.smooth_kernel.weight = torch.nn.Parameter(torch.tensor([[[0.2, 0.9, 0.2]]]), requires_grad=False)
            self.smooth_kernel = self.smooth_kernel.to(torch.device("cuda"))

    def forward(self, input, target):
        n = input.shape[0]
        out_size = (n,) + input.shape[2:]

        # compute softmax over the classes axis
        input_soft = input.softmax(1)
        log_input_soft = input.log_softmax(1)

        # create the labels one hot tensor
        D = input.shape[1]
        if self.smooth_target:
            target_one_hot = F.one_hot(target, num_classes=D).to(input).view(-1, D) # [N*H*W, D]
            target_one_hot = self.smooth_kernel(target_one_hot.float().unsqueeze(1)).squeeze(1) # [N*H*W, D]
            target_one_hot = target_one_hot.view(*target.shape, D).permute(0, 3, 1, 2)
        else:
            target_one_hot = F.one_hot(target, num_classes=D).to(input).permute(0, 3, 1, 2)
        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1.0, self.gamma)

        focal = -self.alpha * weight * log_input_soft
        loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        return loss