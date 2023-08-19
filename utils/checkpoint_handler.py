import os
from collections import OrderedDict

import torch
import torch.nn as nn

from options import Parser


def load_checkpoint(model: nn.Module, checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
    print(f"----checkpoints loaded from path: {checkpoint_path}----")
    return model


def load_distributed_checkpoint(model: nn.Module, checkpoint_path: str):
    # for multi gpu training
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return

    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print(f"----checkpoints loaded from path: {checkpoint_path}----")
    return model


def save_checkpoint(model: nn.Module, save_path: str):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.state_dict(), save_path)
    print(f"----saved checkpoints to path: {save_path}----")


def save_checkpoints(model: nn.Module, options: Parser, iteration: int):
    save_path = os.path.join(options.save_dir, "checkpoints", "itr_{:08d}_u2net.pth".format(iteration))
    save_checkpoint(model, save_path)
