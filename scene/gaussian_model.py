import numpy as np
import torch
from torch import nn

from arguments import OptimizationParams
from utils.general_utils import build_scaling_rotation, strip_symmetric, inverse_sigmoid, get_expon_lr_func
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2 # 未实现cuda

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling,rotation)
            actual_covarivance = L @ L.transpose(1,2) # 交换第二维度和第三维度的位置
            symm = strip_symmetric(actual_covarivance)
            return symm

        # 传入函数引用，而不是得到函数返回值

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree
        # torch.empty(0)创建一个一维空张量
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0) # 不考虑lighting effects的diffuse color,球谐函数的最低阶C0
        self._features_rest = torch.empty(0) # 剩余(rest)的其他球谐系数
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions() # 完成激活函数的初始化

    def capture(self):
        return(
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(), # state_dict() 获取优化器状态，返回一个字典
            self.spatial_lr_scale,
        )

    def restore(self,model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc,features_rest),dim=1) # 返回一维张量

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0],3,(self.max_sh_degree + 1) ** 2)).float().cuda() # shape[0]表示第一维度的size
        features[:,:3,0] = fused_color
        features[:,3:,1:] = 0.0 # emphasize,actually redundant

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),0.0000001) # clamp避免除0错误
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1,3) # 得到形状为(n,3)的张量
        rots = torch.zeros((fused_point_cloud.shape[0],4),device="cuda")
        rots[:,0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0],1),dtype=torch.float,device="cuda")) # 每个点的初始opacity设置为0.1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True)) # 启用梯度跟踪，表示为一个可学习的参数
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1,2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1,2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]),device="cuda")
        self.exposure_mapping = {cam_infos.image_name : idx for idx, cam_info in enumerate(cam_infos)} # 建立字典，并使用 enumerate 创造索引
        self.pretrained_exposures = None
        exposure = torch.eye(3,4,device="cuda")[None].repeat(len(cam_infos),1,1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args : OptimizationParams):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0],1),device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0],1),device="cuda")

        l = [
            {"params" : [self._xyz],"lr":training_args.position_lr_init * self.spatial_lr_scale, "name":"xyz"},
            {"params" : [self._features_dc], "lr":training_args.feature_lr, "name":"f_dc"},
            {"params" : [self._features_rest], "lr":training_args.feature_lr / 20.0, "name":"f_rest"},
            {"params" : [self._opacity], "lr":training_args.opacity_lr, "name":"opacity"},
            {"params" : [self._scaling], "lr":training_args.scaling_lr, "name":"scaling"},
            {"params" : [self._rotation], "lr":training_args.rotation_lr, "name":"rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l,lr=0.0,eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l,lr=0.0,eps=1e-15)
            except:
                self.optimizer = torch.optim.Adam(l,lr=0.0,eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure]) # 默认值 lr = 1e-3 eps = 1e-8

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.exposure_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps
                                                    )

        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init,training_args.exposure_lr_final,
                                                         lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                         lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                         max_steps=training_args.iterations)
