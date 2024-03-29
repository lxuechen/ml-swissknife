"""Utilities for PyTorch's distributed training.

Compatible with torchrun / elastic and torch.distributed.launch.
"""
import logging
import os
import sys
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup(rank: Optional[int] = None, world_size: Optional[int] = None):
    if rank is None:
        rank = get_local_rank()
    if world_size is None:
        world_size = get_world_size()

    if world_size <= 1:
        return rank, world_size

    if not dist.is_initialized():
        logger.warning(
            "Initializing torch.distributed by setting up the process group..."
        )

        if sys.platform == "win32":
            # Distributed package only covers collective communications with Gloo
            # backend and FileStore on Windows platform. Set init_method parameter
            # in init_process_group to a local file.
            # Example init_method="file:///f:/libtmp/some_file"
            init_method = "file:///f:/libtmp/dist-tmp"
            dist.init_process_group(
                backend="gloo",
                init_method=init_method,
                rank=rank,
                world_size=world_size,
            )
        elif torch.cuda.is_available():
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        else:
            dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    return rank, world_size


def cleanup():
    dist.destroy_process_group()


def get_local_rank():
    return int(os.getenv("LOCAL_RANK", -1))


def get_world_size():
    return int(os.getenv("WORLD_SIZE", 1))


def should_save():
    """Return True if the current process is the main process."""
    return get_local_rank() <= 0


is_main_process = should_save
