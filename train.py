import os
import sys
import uuid
from argparse import ArgumentParser

import torch.autograd
from scipy.signal import step2

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not install, please install the correct rasterizer using pip install [3dgs_accel]")

    first_iter = 0
    tb_write = prepare_output_and_logger(dataset) # 设定输出路径
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str=os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/",unique_str[0:10])

if __name__ == "__main__":
    #set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip',type=str,default="127.0.0.1")
    parser.add_argument('--port',type=int,default=6009)
    parser.add_argument('--debug_from',type=int,default=-1)
    parser.add_argument('--detect_anomaly',action='store_true',default=False) # 如果用户不提供'--detect_anomaly'则值为false否则为true
    parser.add_argument('--test_iterations',nargs="+",type=int,default=[7_000,30_000])
    parser.add_argument('--save_iterations',nargs="+",type=int,default=[7_000,30_000])
    parser.add_argument('--quiet',action="store_true")
    parser.add_argument('--disable_viewer',action="store_true")
    parser.add_argument("--start_checkpoint",type=str,default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing" + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quite)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip,args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly) # 异常检测


    # All done
    print("\nTraining complete.")

