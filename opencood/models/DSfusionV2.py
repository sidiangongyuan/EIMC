import cv2
import torch
import torch.nn as nn

from mmdet3d.models.backbones import ResNet
from mmdet3d.models.necks import FPN
from opencood.models.DFusion_utils.Agent_Fusion import AgentFusion
from opencood.models.DFusion_utils.Mocc import Mocc
from opencood.models.QumCo_utils.utils import DenseDepthNet
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.QumCo_utils.dcn_net import DCNNet
from opencood.models.QumCo_utils.grid_mask import GridMask
# from opencood.models.CoMM_utils.backbone import CustomFPN,CustomResNet
from opencood.models.QumCo_utils.query_head import queryfusion

from opencood.models.QumCo_utils.decoder import Box3DDecoder
from opencood.models.DFusion_utils.sensor_fusion import SensorFusion
import torch.nn.functional as F

from opencood.utils.camera_utils import QuickCumsum, bin_depths, cumsum_trick, depth_discretization, gen_dx_bx
from opencood.utils.transformation_utils import normalize_pairwise_tfm, normalize_pairwise_tfm_3d


class DSFusionv2(nn.Module):
    def __init__(self, args): 
        super(DSFusionv2, self).__init__()


        self.embed_dims = args['embed_dims']
        # Lidar backbone
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        self.voxel_size = args['voxel_size']
        self.lidar_range = args['lidar_range']
        self.shrink_flag = True

        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            
        self.use_occ = args['use_occ']
        self.num_classes = args['num_class']
        self.modality = args['modality']
        self.occ_dim = args['occ_dims']
        if "C" in self.modality:
            args_img = args['Image_branch']
            self.use_depth = args_img['use_depth_gt']
            self.grid_conf = args_img['grid_conf']   # 网格配置参数
            self.data_aug_conf = args_img['data_aug_conf']   # 数据增强配置参数
            dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                                self.grid_conf['ybound'],
                                                self.grid_conf['zbound'],
                                                )  # 划分网格

            self.dx = dx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [0.4,0.4,20]
            self.bx = bx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [-49.8,-49.8,0]
            self.nx = nx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [250,250,1]
            
            self.d_min = self.grid_conf['ddiscr'][0]
            self.d_max = self.grid_conf['ddiscr'][1]
            self.num_bins = self.grid_conf['ddiscr'][2]
            self.mode = self.grid_conf['mode']

            self.downsample = args_img['img_downsample']  # 下采样倍数
            self.camC = args_img['img_features']  # 图像特征维度
            self.frustum = self.create_frustum().clone().detach().requires_grad_(False).to(torch.device("cuda"))  # frustum: DxfHxfWx3(41x8x16x3)

            self.D, _, _, _ = self.frustum.shape  # D: 41
            
            self.img_backbone = ResNet(
                depth=50,
                num_stages=4,
                frozen_stages=-1,
                norm_eval=True,
                style='pytorch',
                with_cp=False,
                out_indices=(0, 1, 2, 3),
                norm_cfg=dict(type="BN", requires_grad=True),
                # strides=(1, 1, 1, 1),
                pretrained="/mnt/sdb/public/data/yangk/pretrained/sparse4D/resnet50-19c8e357.pth",
            )
            self.img_neck = FPN(
                num_outs= 4,
                start_level= 1,
                out_channels= self.embed_dims,
                add_extra_convs="on_output",
                relu_before_extra_convs=True,
                in_channels=[256, 512, 1024, 2048],
            )

            self.grid_mask = GridMask(
                    True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
            self.depth_branch = nn.Conv2d(self.embed_dims, self.D, kernel_size=1, padding=0)  # 1x1卷积，变换维度

            self.use_quickcumsum = True

        self.predicter = nn.Sequential(
            nn.Linear(self.occ_dim, self.embed_dims),
            nn.Softplus(),
            nn.Linear(self.embed_dims, self.num_classes),
        )

        self.align = nn.Sequential(
            nn.Linear(self.occ_dim, self.embed_dims),
        )

        self.voxel_conv3d = nn.Conv3d(
            in_channels=self.embed_dims,       # 输入通道数
            out_channels=self.embed_dims,      # 输出通道数（保持不变）
            kernel_size=(2, 2, 1), # 卷积核大小：在H/W方向为4，D方向为1
            stride=(2, 2, 1),    # 步长：在H/W方向为4，D方向为1
            padding=(0, 0, 0)    # 不填充，输出尺寸只受卷积和步长影响
        )

        self.occ_conv3d = nn.Conv3d(
            in_channels=64,   
            out_channels=64,    
            kernel_size=(2, 2, 1), 
            stride=(2, 2, 1),    
            padding=(0, 0, 0)    
        )

        self.mocc = Mocc(args)
        self.sensor_fusion = SensorFusion(args['sensor_fusion'])
        self.agent_fusion = AgentFusion(args['agent_fusion'])
        self.cls_head = nn.Conv2d(self.embed_dims, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.embed_dims, 7 * args['anchor_number'],
                                  kernel_size=1)
    
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.embed_dims, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
            
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']  # 原始图片大小  ogfH:128  ogfW:288
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # 下采样16倍后图像大小  fH: 12  fW: 22
        # ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)  # 在深度方向上划分网格 ds: DxfHxfW(41x12x22)
        ds = torch.tensor(depth_discretization(*self.grid_conf['ddiscr'], self.grid_conf['mode']), dtype=torch.float).view(-1,1,1).expand(-1, fH, fW)

        D, _, _ = ds.shape # D: 41 表示深度方向上网格的数量
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  # 在0到288上划分18个格子 xs: DxfHxfW(41x12x22)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)  # 在0到127上划分8个格子 ys: DxfHxfW(41x12x22)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)  # 堆积起来形成网格坐标, frustum[i,j,k,0]就是(i,j)位置，深度为k的像素的宽度方向上的栅格坐标   frustum: DxfHxfWx3
        return frustum


    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  # B:4(batchsize)    N: 4(相机数目)

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],  # points[:, :, :, :, :, 2:3] ranges from [4, 45) meters
                            points[:, :, :, :, :, 2:3]
                            ), 5)  # 将像素坐标(u,v,d)变成齐次坐标(du,dv,d)
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)  # 将像素坐标d[u,v,1]^T转换到车体坐标系下的[x,y,z]
        
        return points  # B x N x D x H x W x 3 (4 x 4 x 41 x 16 x 22 x 3) 
    
    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B*N*D*H*W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)  # 将图像展平，一共有 B*N*D*H*W 个点

        # flatten indices

        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  # 将[-48,48] [-10 10]的范围平移到 [0, 240), [0, 1) 计算栅格坐标并取整
        geom_feats = geom_feats.view(Nprime, 3)  # 将像素映射关系同样展平  geom_feats: B*N*D*H*W x 3 
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: B*N*D*H*W x 4, geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~240  y: 0~240  z: 0
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept] 
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)  # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # modify griddify (B x C x Z x Y x X) by Yifan Lu 2022.10.7
        # ------> x
        # |
        # |
        # y
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)  # final: 4 x 64 x Z x Y x X
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x  # 将x按照栅格坐标放到final中

        # collapse Z
        #final = torch.max(final.unbind(dim=2), 1)[0]  # 消除掉z维

        # final = torch.max(final, 2)[0]  # 消除掉z维

        return final  # final: 4 x 64 x 240 x 240  # B, C, H, W
    
    
    def get_gt_depth_dist(self, x):  # 对深度维进行onehot，得到每个像素不同深度的概率
        """
        Args:
            x: [B*N, H, W]
        Returns:
            x: [B*N, D, fH, fW]
        """
        # if len(x.shape) == 4:
        #     x = x.squeeze(1)
        target = self.training
        torch.clamp_max_(x, self.d_max) # save memory
        # [B*N, H, W], indices (float), value: [0, num_bins)
        depth_indices, mask = bin_depths(x, self.mode, self.d_min, self.d_max, self.num_bins, target=target)
        depth_indices = depth_indices[:, self.downsample//2::self.downsample, self.downsample//2::self.downsample]
        onehot_dist = F.one_hot(depth_indices.long()).permute(0,3,1,2) # [B*N, num_bins, fH, fW]

        if not target:
            mask = mask[:, self.downsample//2::self.downsample, self.downsample//2::self.downsample].unsqueeze(1)
            onehot_dist *= mask

        return onehot_dist, depth_indices
    

    def get_depth_dist(self, x, eps=1e-5):  # 对深度维进行softmax，得到每个像素不同深度的概率
        return F.softmax(x, dim=1)
    

    def extract_img_feat(self,data_dict, gome):
        
        image_inputs_dict = data_dict['image_inputs']
        img = image_inputs_dict['imgs']

        # extract img features.
        # img.shape: bs ~ num_agent,4,3,H,W
        assert len(img.shape) == 5
        bs,num_cam,C,H,W = img.shape
        img = img.reshape(bs*num_cam,C,H,W)
        # 分离图片和深度信息
        if self.use_depth:
            gt_depths = []
            gt_depth = img[:,-1,:,:]
            depth_gt, depth_gt_indices = self.get_gt_depth_dist(gt_depth)
            img = img[:,:-1,:,:]
        # grid mask
        img = self.grid_mask(img)
        feature_maps = self.img_backbone(img)
        feature_maps =list(self.img_neck(feature_maps))

        # only need 0 
        # I_feature = torch.reshape(feature_maps[0], (bs,num_cam) + feature_maps[0].shape[1:])
        I_feature = feature_maps[0]  # B,C,H,W
     
        # depth loss
        depth_logit = self.depth_branch(I_feature) # C -> D
        depth = self.get_depth_dist(depth_logit)
        new_x = depth.unsqueeze(1) * I_feature.unsqueeze(2) # new_x: B,C,D,H,W

        cam_C = new_x.shape[1]

        x = new_x.view(bs, num_cam, cam_C, self.D, H//self.downsample, W//self.downsample)  #将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 16 x 22)

        x = x.permute(0, 1, 3, 4, 5, 2) # x: B x N x D x fH x fW x C

        # TODO: voxel_features
        voxel_features = self.voxel_pooling(gome, x)  # x: 4 x 64 x 240 x 240

        return voxel_features, (depth_logit, depth_gt_indices)
    
    def extract_pts_feat(self, data_dict):
        # we also need pts, original points. List
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']


        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      'pairwise_t_matrix': pairwise_t_matrix,}

        if 'occ_voxel_features' in data_dict['processed_lidar']:
            batch_dict['occ_voxel_features'] = data_dict['processed_lidar']['occ_voxel_features']
            batch_dict['occ_voxel_coords'] = data_dict['processed_lidar']['occ_voxel_coords']
            batch_dict['occ_voxel_num_points'] = data_dict['processed_lidar']['occ_voxel_num_points']

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)  # output_c=64

        batch_dict = self.backbone(batch_dict) # output_c = 128

        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
            
        return spatial_features_2d, batch_dict
     

    def forward(self, data_dict): 
        # x: [4,4,3,256, 352]
        # rots: [4,4,3,3]
        # trans: [4,4,3]
        # intrins: [4,4,3,3]
        # post_rots: [4,4,3,3]
        # post_trans: [4,4,3]
        image_inputs_dict = data_dict['image_inputs']
        x, rots, trans, intrins, post_rots, post_trans = \
            image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
        
        # s1: extract features from image and pts 

        if "C" in self.modality:
            gome = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
            voxel_feature, depth_items = self.extract_img_feat(data_dict, gome)  # B,C1,10,256,256 ,  c1=128
        else:
            voxel_feature = None
            depth_items = None
        if "L" in self.modality: 
            pts_feats, batch_dict = self.extract_pts_feat(data_dict) # pts: B,C,H,W
            occ = batch_dict['3d_spatial_features'] # B,C2,10,256,256  ,  c2 = 64 
            occ = occ.permute(0,1,3,4,2)
            occ = self.occ_conv3d(occ).permute(0,2,3,4,1) # B,H,W,D,C
            _,H,W,D,C = occ.shape
            
            affine_matrix = normalize_pairwise_tfm_3d(data_dict['pairwise_t_matrix'], D, H, W, self.voxel_size[0]) # [B, L, L, 3, 4]
            # V2 
            fused_occ = self.mocc(occ,data_dict, affine_matrix) # B,H,W,D,C

            occ_pred = self.predicter(fused_occ).sigmoid()
            #TODO: multi-agent occ_fusion 

        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.lidar_range[4] - self.lidar_range[1], self.lidar_range[3] - self.lidar_range[0], 1)
            
        voxel_feature_ = self.voxel_conv3d(voxel_feature.permute(0,1,3,4,2)).permute(0,2,3,4,1) # B,H,W,D,C   camera  voxel 
        
        w_voxel_feature = voxel_feature_ * occ_pred  # Occ-guided voxel 

        #TODO: V3 cat image occ and LiDAR occ  
        fused_occ = self.align(fused_occ)
        fuse_voxel_features = w_voxel_feature + fused_occ + voxel_feature_# B,H,W,D,C
        
        cam_f = torch.sum(fuse_voxel_features, dim=-2) # B,H,W,C    
        #s2: multi-sensor fusion
        res = self.sensor_fusion(cam_f, pts_feats.permute(0,2,3,1))
        #s3: agent-fusion 

        # only support B==1
        res, heatmap = self.agent_fusion(res, data_dict, affine_matrix)

        #s4: decoder
        psm = self.cls_head(res[0:1])
        rm = self.reg_head(res[0:1])

        output_dict = {'psm': psm,
                       'rm': rm,
                       'depth_items': depth_items,
                       'heatmap': heatmap,
                       }
        
        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(res[0:1])})
        

        return output_dict


def compile_model(grid_conf, data_aug_conf, outC):
    return DSFusionv2(grid_conf, data_aug_conf, outC)

