import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from mmdet.models import weight_reduce_loss, l1_loss
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss

from opencood.loss.ciassd_loss import softmax_cross_entropy_with_logits

from ..models.QumCo_utils.decoder import Box3DDecoder
from ..models.QumCo_utils.target import Box3DTarget

def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A warpper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if torch.cuda.is_available() and pred.is_cuda:
                calculate_loss_func = sigmoid_focal_loss
            else:
                num_classes = pred.size(1)
                target = F.one_hot(target, num_classes=num_classes + 1)
                target = target[:, :num_classes]
                calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls

class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

            #anchors = H * W * anchor_num

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = torch.mul(loss , weights)

        return loss

class L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


class TwopartDetrLoss(nn.Module):
    def __init__(self,args):
        super(TwopartDetrLoss,self).__init__()
        self.ego_loss = MultiDetrLoss(args['ego'])
        self.num_query = args['num_query']
    
    def forward(self, result, gt_dict):
        # output_dict = {
        #     'cls_scores': predictions_ego,
        #     'reg_preds': boxes_ego,
        #     "depth_loss": depth_loss
        # }

        # split ego and neighbor
        ego_res_dict = {}
        neighbor_res_dict= {}
        
        pred = result['cls_scores'] # list,  [1, N , 1]
        
        layers = len(pred)

        for layer in range(layers):
            ego_res_dict['cls_scores'] = pred[layer][:, :self.num_query, :]
            ego_res_dict['reg_preds'] = result['reg_preds'][layer][:, :self.num_query, :]
            ego_res_dict['depth_loss'] = result['depth_loss']

            neighbor_res_dict['cls_scores'] = pred[layer][:, self.num_query:, :]
            neighbor_res_dict['reg_preds'] = result['reg_preds'][layer][:, self.num_query:, :]
        
        ego_loss = self.ego_loss(ego_res_dict, gt_dict)
        neighbor_loss = self.ego_loss(neighbor_res_dict, gt_dict)

        return 0.8*ego_loss + 0.2*neighbor_loss
    
    
class MultiDetrLoss(nn.Module):
    def __init__(self,args):
        super(MultiDetrLoss,self).__init__()
        self.reg_weights = args['reg_weights']
        self.reg_weight = args['reg_weight']
        self.sampler = Box3DTarget(**args['sampler'])
        self.decoder = Box3DDecoder()
        self.loss_cls = FocalLoss(**args['focalloss'])
        self.loss_reg = L1Loss(loss_weight = args['reg_weight'])
        self.cls_threshold_to_reg = args['cls_threshold_to_reg']
        self.dir_weight = args['dir_weights']
        self.loss_dict = {}
        self.use_dir = args['use_dir']

    def forward(self, result, gt_dict):
        # here, cls and reg is list, which contain len(self.layer) results.
        # We only need ego . So before this , we need to fuse agent
        '''
        Args:
            cls_scores: [list] : layer of op.(6)    cls_scores[0] :  B,900,1
            reg_preds:  [list] :  ---               reg_preds[0] : B,900,7
            data: include gt_box_3d, gt_label_3d
            feature_maps:

        Returns:
        '''
        # Todo: direction loss
        cls_scores = result['cls_scores']
        reg_preds = result['reg_preds']
        B,query_num,_ = reg_preds[0].shape
        cls_target_list = []
        obj_target = gt_dict['object_target']
        num_total_ = B*query_num
        if 'dirs' in result:
            dirs = result['dirs']
        else:
            dirs = None
        for i in range(B):
            obj_num = len(obj_target[i])
            cls_target = torch.zeros(obj_num).to(dtype=torch.long, device=cls_scores[0].device) # CREATE label
            cls_target_list.append(cls_target)
            if obj_num == 0:
                obj_target[i] = torch.empty((0, reg_preds[0].shape[-1]), device=reg_preds[0].device, dtype=reg_preds[0].dtype)
            else:
                obj_target[i] = torch.stack(obj_target[i]).to(device=reg_preds[0].device, dtype=reg_preds[0].dtype)
        output = {}
        total_loss = 0
            

        for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):
            if cls is None:
                continue
            reg = reg[..., : len(self.reg_weights)]  # self.reg_weights = 2,2,2,1,...,1    10 length the same as classification number.
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                cls_target_list,
                obj_target
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            avg_factor = num_pos * 1.0 + (num_total_-num_pos) * 0.1
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=avg_factor)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            if torch.any(torch.isnan(reg_target)):
                print("reg_target is NAN")
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )

            box_preds_sin, reg_targets_sin = add_sin_difference(reg, reg_target)

            reg_loss = self.loss_reg(box_preds_sin, reg_targets_sin, weight=reg_weights, avg_factor=num_pos)

            if dirs is not None:
                dir = dirs[decoder_idx]
                dir = dir.flatten(end_dim=1)[mask]
                dir_loss = calculate_direction_loss(reg, dir, reg_target, 0.8)
                dir_loss = dir_loss.sum() * self.dir_weight
                output.update(
                    {
                        f"loss_dir_{decoder_idx}": dir_loss,
                    }
                )
            else:
                dir_loss = 0
            
            total_loss = total_loss + cls_loss + reg_loss + dir_loss

            output.update(
                {
                    f"loss_cls_{decoder_idx}": cls_loss,
                    f"loss_reg_{decoder_idx}": reg_loss,
                }
            )
        self.loss_dict = output
        if "depth_loss" in result:
            total_loss = total_loss + result["depth_loss"]
            self.loss_dict.update(
                {
                    "depth_loss": result["depth_loss"]
                }
            )
        self.loss_dict.update(
            {
                'total_loss': total_loss,
            }
        )
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None):
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
        total_loss = self.loss_dict['total_loss']
        conf_loss = self.loss_dict['loss_cls_5']
        reg_loss  = torch.sum(self.loss_dict['loss_reg_5'])

        print_msg = ("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                     " || Loc Loss: %.4f" % (
                         epoch, batch_id + 1, batch_len,
                         total_loss.item(), conf_loss.item(), reg_loss.item()))

        if self.use_dir:
            dir_loss = self.loss_dict['loss_dir_5']
            print_msg += " || Dir Loss: %.4f" % dir_loss.item()

        if "depth_loss" in self.loss_dict:
            depth_loss = self.loss_dict["depth_loss"]
            print_msg += "|| Depth Loss: % 4f" % depth_loss.item()
        print(print_msg)

        if not writer is None:
            writer.add_scalar('Regression_loss', reg_loss.item(),
                              epoch * batch_len + batch_id)
            writer.add_scalar('Confidence_loss', conf_loss.item(),
                              epoch * batch_len + batch_id)

            if self.use_dir:
                writer.add_scalar('dir_loss', dir_loss.item(),
                                  epoch * batch_len + batch_id)


def get_direction_classification_targets(reg_targets):
    """
    Args:
        reg_targets: [N, 7]
            The last term is θ_gt (实际的目标角度)
        
    Returns:
        dir_cls_targets: [N, 2]
            0或1的one-hot编码
    """
    N = reg_targets.shape[0]
    # 提取实际角度
    rot_gt = reg_targets[:, -1]  # [N]
    # 方向分类: 如果角度>0, 则为1类，否则为0类
    dir_cls_targets = (rot_gt > 0).long()  # [N]
    # 转换为one-hot编码
    dir_cls_targets = torch.nn.functional.one_hot(dir_cls_targets, num_classes=2)  # [N, 2]
    return dir_cls_targets


def angle_loss(pred_angle, true_angle):
    """
    Args:
        pred_angle: [N] 预测的角度值（弧度制）
        true_angle: [N] 真实的角度值（弧度制）
    
    Returns:
        角度损失值
    """
    return torch.mean(1 - torch.cos(pred_angle - true_angle))


def calculate_direction_loss(pred, dir, reg_targets, dir_weight=0.5):
    """
    Args:
        pred: [N, 7] 预测的bbox信息，包括角度（最后一个维度是角度）
        reg_targets: [N, 7] 真实的bbox信息，包括角度（最后一个维度是角度）
        dir_weight: 方向损失的权重
    
    Returns:
        方向的综合损失值
    """
    N = reg_targets.shape[0]

    # 提取预测的角度值和真实角度值
    pred_angle = pred[:, -1]  # [N]
    true_angle = reg_targets[:, -1]  # [N]
    
    # 计算角度回归损失
    dir_reg_loss = angle_loss(pred_angle, true_angle)  # 角度回归损失
    
    # 生成角度分类目标
    dir_cls_targets = get_direction_classification_targets(reg_targets)  # [N, 2]
    
    # 计算角度分类损失
    # 假设 pred_cls 包含预测的角度分类概率 [N, 2]
    pred_cls = dir
    dir_cls_loss = F.cross_entropy(pred_cls, dir_cls_targets.argmax(dim=1))  # 方向分类损失
    
    # 综合损失
    total_loss = dir_reg_loss * dir_weight + dir_cls_loss * (1 - dir_weight)
    
    return total_loss



import torch.distributed as dist
def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


def add_sin_difference(boxes1, boxes2, dim=6):
    assert dim != -1
    rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                        torch.cos(boxes2[..., dim:dim + 1])
    rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                      torch.sin(boxes2[..., dim:dim + 1])

    boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                        boxes1[..., dim + 1:]], dim=-1)
    boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                        boxes2[..., dim + 1:]], dim=-1)
    return boxes1, boxes2