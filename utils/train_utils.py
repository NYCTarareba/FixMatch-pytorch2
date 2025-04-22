import math
import os
import random
import argparse
import shutil
import numpy as np
import yaml
from types import SimpleNamespace
import torch
from torch.optim.lr_scheduler import LambdaLR



def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def build_parser():
    """
    Parse command line arguments
    - --config: Path to config.yaml
    - --resume: Path to latest checkpoint
    - --epochs: Number of epochs to train
    - --label-ratio: Ratio of labeled data
    - --threshold: Threshold for pseudo-labeling
    """
    parser = argparse.ArgumentParser(description='FixMatch Training')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        help='path to latest checkpoint (default: none)')
    return parser

def load_config(parser_args):
    """
    Load config yaml file and override with command line arguments
    """
    # Load config file
    with open(parser_args.config, 'r') as f:
        args_dict = yaml.safe_load(f)
    args = SimpleNamespace(**args_dict)
    
    # save out based on config suffix
    args.out = parser_args.config.replace('.yaml', '').split('_')[-1]

    # Override with command line arguments
    for key, value in vars(parser_args).items():
        if value is not None:
            setattr(args, key, value)
    return args
