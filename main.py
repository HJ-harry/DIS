import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb

from guided_diffusion.diffusion import Diffusion
from guided_diffusion.diffusion_adapt import Diffusion as Diffusion_adapt

from pathlib import Path

torch.set_printoptions(sci_mode=False)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--deg", type=str, required=True, help="Degradation"
    )
    parser.add_argument(
        "--path_y",
        type=str,
        required=True,
        help="Path of the test dataset.",
    )
    parser.add_argument(
        "--sigma_y", type=float, default=0., help="sigma_y"
    )
    parser.add_argument(
        "--eta", type=float, default=0.85, help="Eta"
    )    
    parser.add_argument(
        "--simplified",
        action="store_true",
        help="Use simplified DDNM, without SVD",
    )    
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--deg_scale", type=float, default=0., help="deg_scale"
    )    
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )
    parser.add_argument(
        "-n",
        "--noise_type",
        type=str,
        default="gaussian",
        help="gaussian | 3d_gaussian | poisson | speckle"
    )
    parser.add_argument(
        "--add_noise",
        action="store_true"
    )
    # adaptation parameters
    parser.add_argument(
        "--use_adapt",
        action="store_true"
    )
    parser.add_argument(
        "--use_additional_dc",
        action="store_true",
        help="when using adaptation, additionally use projection w.r.t. y?"
    )
    parser.add_argument(
        "--adapt_method", 
        type=str, 
        default="lora", 
        help="Method of adaptation"
    )
    parser.add_argument(
        "--lora_rank", 
        type=int, 
        default=4, 
        help="Rank used in LoRA parameter injection"
    )
    parser.add_argument(
        "--tv_penalty", 
        type=float, 
        default=0.0,
        help="Additional TV penalty when applying adaptation. \
            If you want to use this, set this to a very low value e.g. 1e-6."
    )
    parser.add_argument(
        "--adapt_lr", 
        type=float, 
        default=1e-3,
        help="5e-5 for decoder, 1e-3 for lora"
    )
    parser.add_argument(
        "--adapt_iter", 
        type=int, 
        default=10,
        help="5 for decoder, 10 for lora"
    )
    parser.add_argument(
        "--ddim_eps_choice", 
        type=str, 
        default="adapt",
        help="One of [adapt, old]. If adapt, use eps_pred from the online adapted model for DDIM noise addition. \
              If old, use the old non-adapted model for this. This would require an additional NFE per iteration."
    )
    parser.add_argument(
        "--use_dc_before_adapt", 
        action="store_true",
        help="If true, minimize ||A(DC(x0hat)) - y||^2, where DC is some gradient descent step"
    )
    parser.add_argument(
        "--dc_before_adapt_method", 
        type=str,
        default="dds",
        help="which type of gradient descent step to take before adaptation. one of [dps, dds]."
    )
    parser.add_argument(
        "--T_sampling", type=int, default=50, help="Total number of sampling steps"
    )
    parser.add_argument(
        "--method", 
        type=str,
        default="ddnm",
    )

    

    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    use_adapt = True if args.use_adapt else False
    use_additional_dc = True if args.use_additional_dc else False
    use_dc_before_adapt = True if args.use_dc_before_adapt else False
    am = args.adapt_method
    print(f"Use Adaptation? {use_adapt}")
    if use_adapt:
        args.image_folder = Path(args.image_folder) / f"{args.deg}" / f"x{args.deg_scale}" / f"{args.T_sampling}" \
                            / f"adapt_{use_adapt}" / f"iter_{args.adapt_iter}_lr{args.adapt_lr}_tv{args.tv_penalty}"
    else:
        args.image_folder = Path(args.image_folder) / f"{args.deg}" / f"x{args.deg_scale}" / f"{args.T_sampling}" \
                            / f"adapt_{use_adapt}"
    args.image_folder.mkdir(exist_ok=True, parents=True)
    
    
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    try:
        if args.use_adapt:
            print(f"Using adaptation...")
            runner = Diffusion_adapt(args, config)
        else:
            runner = Diffusion(args, config)
        runner.sample(args.simplified)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
