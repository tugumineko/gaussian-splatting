from typing import NamedTuple

import numpy as np
import torch


# 继承自NamedTuple，一旦实例化则无法改变数据
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

'''
def geom_transform_points(points, transf_matrix):
    P, _ = points.shape # '_'表示占位符,P表示第一维数组的size
    ones = torch.ones(P,1,dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones],dim=1)
    points_out = torch.
'''
