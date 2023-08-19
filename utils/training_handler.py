import sys
import time
import traceback
from typing import Callable
from pprint import pprint

import torch
import torch.nn as nn
from torch import distributed
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from utils.multi_gpu_handler import cleanup, synchronize, set_seed
from utils.checkpoint_handler import save_checkpoints
from options import Parser


def run_training_batch(data_batch, u_net: nn.Module, device: torch.device, loss_function: Callable, start_time: float,
                       options: Parser, optimizer: torch.optim, local_rank: int, iteration: int,
                       writer: SummaryWriter = None):
    image, label = data_batch
    image = Variable(image.to(device))
    label = label.type(torch.long)
    label = Variable(label.to(device))

    d0, d1, d2, d3, d4, d5, d6 = u_net(image)

    loss0 = loss_function(d0, label)
    loss1 = loss_function(d1, label)
    loss2 = loss_function(d2, label)
    loss3 = loss_function(d3, label)
    loss4 = loss_function(d4, label)
    loss5 = loss_function(d5, label)
    loss6 = loss_function(d6, label)
    del d1, d2, d3, d4, d5, d6

    total_loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    for param in u_net.parameters():
        param.grad = None

    total_loss.backward()
    if options.clip_grad != 0:
        nn.utils.clip_grad_norm_(u_net.parameters(), options.clip_grad)
    optimizer.step()

    if local_rank == 0:
        # printing and saving work
        if iteration % options.print_freq == 0:
            pprint("[step-{:08d}] [time-{:.3f}] [total_loss-{:.6f}]  [loss0-{:.6f}]".format(
                    iteration, time.time() - start_time, total_loss, loss0
                ))

        # TODO: setup board add image to writer

        if iteration % options.save_freq == 0:
            save_checkpoints(model=u_net, iteration=iteration, options=options)


def run_training_loop(training_loop: Callable, options: Parser, seed: int = 1000):
    try:
        if options.distributed:
            print("Initialize Process Group...")
            distributed.init_process_group(backend="nccl", init_method="env://")
            synchronize()

        # run training process
        set_seed(seed)
        training_loop(options)

        # cleanup
        cleanup(options.distributed)
        print("Exiting..............")

    except KeyboardInterrupt:
        cleanup(options.distributed)

    except Exception:
        traceback.print_exc(file=sys.stdout)
        cleanup(options.distributed)
