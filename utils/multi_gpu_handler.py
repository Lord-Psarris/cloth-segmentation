# for handling multi gpu/distributed training

import numpy as np
import random
import torch
import os

from torch import distributed


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)


def synchronize():
    if not distributed.is_available() or not distributed.is_initialized():
        return

    world_size = distributed.get_world_size()
    if world_size == 1:
        return

    distributed.barrier()


def cleanup(is_distributed: bool):
    # cleanup torch distributed process groups
    if is_distributed:
        distributed.destroy_process_group()
