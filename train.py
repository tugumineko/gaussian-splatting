import sys
from argparse import ArgumentParser

from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state

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
    # ...

    # All done
    print("\nTraining complete.")
