import time
import warnings

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from networks import U2NET
from options.main import Parser
from dataset.load_dataset import DatasetLoader, sample_data
from utils.checkpoint_handler import load_checkpoint, save_checkpoints
from utils.training_handler import run_training_loop, run_training_batch
from options.utils import generate_options_directories, generate_yaml_file

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def args_parser(yaml_path: str = None):
    # TODO: set to process input arguments :- dataset_folder, device, checkpoints_path, continue_train
    options = Parser()

    # run options utils
    generate_options_directories(options)
    generate_yaml_file(options, yaml_path)

    return options


def training_loop(options: Parser):
    # local rank
    if options.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK"))
        # Unique only on individual node.
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cpu")

    # initialize u2net
    u_net = U2NET(in_ch=3, out_ch=4)
    if options.continue_train:
        u_net = load_checkpoint(u_net, options.unet_checkpoint)
    u_net = u_net.to(device)
    u_net.train()

    # save u2net network
    if local_rank == 0:
        with open(os.path.join(options.save_dir, "networks.txt"), "w") as outfile:
            print("<----U-2-Net---->", file=outfile)
            print(u_net, file=outfile)

    # speed up training process using DistributedDataParallel
    if options.distributed:
        u_net = nn.parallel.DistributedDataParallel(u_net,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank,
                                                    broadcast_buffers=False)
        print("---- improving training speed using DistributedDataParallel")

    # initialize optimizer
    optimizer = optim.Adam(u_net.parameters(),
                           lr=0.001, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0)

    # load dataset
    dataloader = DatasetLoader(options)
    dataloader.initialize()
    loader = dataloader.get_loader()

    # TODO: add summary writer
    writer = None

    # set loss function
    weights = np.array([1, 1.5, 1.5, 1.5], dtype=np.float32)
    weights = torch.from_numpy(weights).to(device)
    loss_ce = nn.CrossEntropyLoss(weight=weights).to(device)

    get_data = sample_data(loader)

    print("------ Starting Training Loop ------")
    start_time = time.time()

    # main loop
    for iteration in range(dataset_size // options.batchSize):
        data_batch = next(get_data)
        run_training_batch(data_batch, u_net,
                           device=device, start_time=start_time,
                           loss_function=loss_ce, options=options,
                           optimizer=optimizer, local_rank=local_rank,
                           iteration=iteration, writer=writer)

    if local_rank == 0:
        iteration += 1
        save_checkpoints(model=u_net, options=options, iteration=iteration)


if __name__ == "__main__":
    # process args
    parser_options = args_parser()

    # handle inference
    run_training_loop(training_loop, parser_options)

    print("..... Training Completed .....")
