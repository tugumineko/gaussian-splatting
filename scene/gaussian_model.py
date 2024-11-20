import json
import os.path
from pickletools import optimize

import numpy as np
import torch
from plyfile import PlyElement, PlyData
from prompt_toolkit.input.typeahead import store_typeahead
from torch import nn

from arguments import OptimizationParams
from utils.general_utils import build_scaling_rotation, strip_symmetric, inverse_sigmoid, get_expon_lr_func, \
    build_rotation
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2 # 未实现cuda

from utils.system_utils import mkdir_p

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
        self.max_radii2D = torch.empty(0) # 3DGS分布半径
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

    def update_learning_rate(self,iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None: # 没有预训练曝光值，则
            for param_group in self.exposure_optimizer.param_groups:
                param_group["lr"] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x','y','z','nx','ny','nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i)) # 生成索引
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self,path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy() # 从cpu中分离出来并转化为numpy数组
        normals = np.zeros_like(xyz) # 创建与xyz形状相同的numpy张量
        f_dc = self._features_dc.detach().transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy() # 交换维度，展平，连续存储，转为numpy数组
        f_rest = self._features_rest.detach().transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy() # 交换维度，展平，连续存储，转为numpy数组
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0],dtype=dtype_full)
        attributes = np.concatenate((xyz,normals,f_dc,f_rest,opacities,scale,rotation),axis=1) # axis = 1，沿水平方向拼接，增加数组列数
        elements[:] = list(map(tuple,attributes)) # 转化为元组列表
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)* 0.01)) # opacity >= 0.01
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new,"opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self,path,use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path),os.pardir,os.pardir,"exposure.json") # 返回上两级路径与exposure.json拼接，os.pardir表示上一级目录
            if os.path.exists(exposure_file):
                with open(exposure_file,"r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name:torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures} # 预训练参数禁用梯度计算
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        # plydata.elements[0]保存顶点(vertex)数据
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[...,np.newaxis]

        features_dc = np.zeros((xyz.shape[0],3,1))
        features_dc[:,0,0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:,1,0] = np.asarray(plydata.elements[1]["f_dc_1"])
        features_dc[:,2,0] = np.asarray(plydata.elements[2]["f_dc_2"])

        # 按编号对f_rest_x进行排序
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x : int(x.split("_")[-1]))

        assert len(extra_f_names)==3*(self.max_sh_degree +1)**2 -3
        features_extra = np.zeros((xyz.shape[0],len(extra_f_names)))
        for idx,attr_name in enumerate(extra_f_names):
            features_extra[:,idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P,F,SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0],3,(self.max_sh_degree +1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properities if p.name.startwith("scale_")]
        scale_names = sorted(scale_names,key=lambda x : int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0],len(scale_names)))
        for idx,attr_name in enumerate(scale_names):
            scales[:,idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properities if p.name.startwith("rot")]
        rot_names = sorted(rot_names, key=lambda x : int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0],len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:,idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float,device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc,dtype=torch.float,device="cuda").transpose(1,2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra,dtype=torch.float,device="cuda").transpose(1,2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities,dtype=torch.float,device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales,dtype=torch.float,device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots,dtype=torch.float,device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    # 更新优化后的参数
    def replace_tensor_to_optimizer(self,tensor,name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                store_state = self.optimizer.state.get(group['params'][0],None) # 取第一个元素，没有为None
                store_state["exp_avg"] = torch.zeros_like(tensor)
                store_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]] # 删除旧状态
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"]] = store_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self,mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0])
            if stored_state is not None:
                # mask:bool 进行剪枝,只保留true的部分
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def prune_points(self,mask):
        valid_points_mask = ~mask # 取反
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    # cat拼接，拓展
    def cat_tensors_to_optimizer(self,tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0],None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"],torch.zeros_like(extension_tensor)),dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"],torch.zeros_like(extension_tensor)),dim=0)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"],extension_tensor),dim=0).requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0],extension_tensor),dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def densification_postfix(self,new_xyz,new_features_dc,new_features_rest,new_opacities,new_scaling,new_rotation,new_tmp_radii):
        d = {"xyz":new_xyz,
        "f_dc":new_features_dc,
        "f_rest":new_features_rest,
        "opacity":new_opacities,
        "scaling":new_scaling,
        "rotation":new_rotation,
        }

        optimize_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimize_tensors["xyz"]
        self._features_dc = optimize_tensors["f_dc"]
        self._features_rest = optimize_tensors["f_rest"]
        self._opacity = optimize_tensors["opacity"]
        self._scaling = optimize_tensors["scaling"]
        self._rotation = optimize_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii,new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0],1),device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0],1),device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0],1),device="cuda")

    # 点云优化
    def densify_and_split(self,grads,grad_threshold,scene_extent,N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points),device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze() # 去除大小为1的维度
        selected_pts_mask = torch.where(padded_grad >= grad_threshold,True,False)
        # self.get_scaling 获取最大值 (其实就是有效值，其余都是0)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0),3),device="cuda")
        samples = torch.normal(mean=means,std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots,samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N,1) # 进行旋转变换后进行点偏移
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8 * N) )
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeated(N)

        self.densification_postfix(new_xyz,new_features_dc,new_features_rest,new_opacity,new_scaling,new_rotation,new_tmp_radii)

        # split
        prune_filter = torch.cat((selected_pts_mask,torch.zeros(N * selected_pts_mask.sum(),device="cuda",dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self,grads,grad_threshold,scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads,dim=-1) >= grad_threshold,True,False)
        selected_pts_mask = torch.logical_and((selected_pts_mask,
                                               torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent))

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz,new_features_dc,new_features_rest,new_opacities,new_scaling,new_rotation,new_tmp_radii)

    def densify_and_prune(self,max_grad,min_opacity,extent,max_screen_size,radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0 # is NaN

        self.tmp_radii = radii
        self.densify_and_clone(grads,max_grad,extent)
        self.densify_and_split(grads,max_grad,extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze() # 小于最小透明度的部分需要被移除
        if max_screen_size: # 如果传入了max_screen_size
            # 存在屏幕最大尺寸限制
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask,big_points_vs),big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii # ???
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self,viewspace_point_tensor,update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2],dim=-1,keepdim=True) # 计算x,y方向的梯度，并保持维度不变
        self.denom[update_filter] += 1