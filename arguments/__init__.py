import os
import sys
from argparse import ArgumentParser, Namespace


# 空类 用于拓展
class GroupParams:
    pass

class ParamGroup:
    def __init__(self,paser:ArgumentParser,name:str,fill_none = False):
        # group并不是ParamsGroup的成员,self.xxx才是
        group = paser.add_argument_group(name)
        # 命令行传参
        for key, value in vars(self).items():
            shorthand = False
            # 带下划线，代码会以argparse中以长格式和短格式添加命令行选项
            # 单下划线是保护成员，双下划线是私有成员
            # 不带下划线，参数只能以长格式的形式添加
            if key.startswith("_"):
                shorthand = True
                key = key[1:]

            t = type(value)
            # fill_none == True则value = None
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key,("-"+key[0:1]),default = value,action="store_true")
                else:
                    group.add_argument("--"+key,("-"+key[0:1]),default=value,type=t)

            else:
                if t == bool:
                    group.add_argument("--"+key,default=value,action="store_true")
                else:
                    group.add_argument("--"+key,default=value,type=t)

    # 为空类GroupParams设置属性值
    def extract(self,args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group,arg[0],arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self,parser,sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser,"Loading Parameters",sentinel)

    def extract(self,args):
        g = super().extract(args)
        # os.path.abspath(path)转换为绝对路径
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self,parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser,"Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self,parser):
        self.iterations = 30_000 # 优化算法要执行的总迭代次数
        self.position_lr_init = 0.00016 # 初始位置学习率
        self.position_lr_final = 0.0000016 # 最后位置学习率，优化过程中学习率通常会不断减小
        self.position_lr_delay_mult = 0.01 # 学习率延迟乘数
        self.position_lr_max_steps = 30_000 # 指定位置学习率最大步骤
        self.feature_lr = 0.0025
        self.opacity_lr = 0.005
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01 # 密集度的百分比
        self.lambda_dssim = 0.2 # DSIM函数的权重系数
        self.densification_interval = 100 # 执行密集化的间隔
        self.opacity_reset_interval = 3000 # 透明度参数每迭代3000次重置一次
        self.densify_from_iter = 500 # 500次之后开启密集化
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002 # 在更新模型参数时，密集化步骤的梯度阈值。只有当梯度超过此阈值时才会执行密集化操作
        self.depth_l1_weight_init = 1.0 # 深度L1权重的初始值，L1正则化可以防止过拟合
        self.depth_l1_weight_final = 0.01
        self.random_background = False # 随机背景通常用于训练图像模型，以增加数据集的多样性
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def get_command_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:] # 返回所有传递的参数的字符串列表
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path,"cfg_args")
        print("Looking for config file in",cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string) # 将cfgfile_string作为代码执行

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict) # 解析字典转换成带有若干(k,v)属性的命名空间

