"""Utilities for PyTorch's distributed training.

Last tweaks were made in Feb. 2021, so might be outdated.
"""
import os
import sys

import torch.distributed as dist


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


def should_save():
    """Return True if the current process is the main process.

    This function is compatible with torchrun / elastic and torch.distributed.launch.
    """
    local_rank = os.getenv("LOCAL_RANK", -1)
    return local_rank <= 0
