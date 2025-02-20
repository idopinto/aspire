import os
from datetime import timedelta
import torch
import torch.distributed as dist


def setup_ddp(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=3600)
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process(process_rank=0):
    return process_rank == 0