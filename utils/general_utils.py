import random
import sys
from datetime import datetime

import numpy as np
import torch

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

# 下半对角
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0]+r[:,1]*r[:,1]+r[:,2]*r[:,2]+r[:,3]*r[:,3])

    q = r/norm[:,None]

    R = torch.zeros((q.size(0),3,3),device="cuda")

    r = q[:,0]
    x = q[:,1]
    y = q[:,2]
    z = q[:,3]

    R[:,0,0] = 1 - 2 * (y*y + z*z)
    R[:,0,1] = 2 * (x*y - r*z)
    R[:,0,2] = 2 * (x*z + r*y)
    R[:,1,0] = 2 * (x*y + r*z)
    R[:,1,1] = 1 - 2 * (x*x + z*z)
    R[:,1,2] = 2 * (y*z - r*x)
    R[:,2,0] = 2 * (x*z - r*y)
    R[:,2,1] = 2 * (y*z + r*x)
    R[:,2,2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s,r):
    L = torch.zeros((s.shape[0],3,3),dtype=torch.float,device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

# 重定义输出流，定义输出末尾是有’\n‘时,则输出带时间戳
# 若设置为静默模式，则print函数无效，不会输出任何内容
def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self,silent):
            self.silent = silent

        def write(self,x):
            if not self.silent:
                if x.endwith("\n"):
                    old_f.write(x.replace("\n", "[{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    # 设定随机数种子，使不同人的实验结果一致
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0")) # 指定cuda使用第一个GPU
