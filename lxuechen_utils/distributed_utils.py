"""Utilities for PyTorch's distributed training.

Last tweaks were made in Feb. 2021, so might be outdated.
"""
import os
import sys

import torch
import torch.distributed as dist

from lxuechen_utils import utils


def setup(rank, world_size):
    if sys.platform == 'win32':
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method = "file:///f:/libtmp/dist-tmp"
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prep(train_dir, task_folder, seed):
    utils.manual_seed(seed)
    utils.show_env()

    world_size = torch.cuda.device_count()

    ckpts_dir = os.path.join(train_dir, task_folder, "ckpts")
    results_dir = os.path.join(train_dir, task_folder, "results")
    os.makedirs(ckpts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    return world_size, ckpts_dir, results_dir
