import os

from pathlib import Path
from typing import Union, Literal


class Parser:

    def __init__(self, name: str = "training_cloth_segm_u2net", dataset_folder: Union[str, Path] = "_dataset",
                 device: str = "cpu", continue_train: bool = False, is_train: bool = True,
                 checkpoints_path: Union[str, Path] = "_prev_checkpoints/cloth_segm_unet_surgery.pth"):

        if device not in ['cpu', "gpu", "cuda"]:
            raise ValueError("invalid device passed")

        self.name = name  # Experiment name
        self.image_folder = f"{dataset_folder}/train/"  # image folder path
        self.df_path = f"{dataset_folder}/train.csv"  # label csv path
        self.distributed = False if device == "cpu" else True  # True for multi gpu training

        self.is_train = is_train
        self.continue_train = continue_train

        # base width of trained files
        self.fine_width = 192 * 4
        self.fine_height = 192 * 4

        # Mean std params
        self.mean = 0.5
        self.std = 0.5

        self.batchSize = 2  # 12
        self.nThreads = 2  # 3
        self.max_dataset_size = float("inf")

        self.serial_batches = False
        if continue_train:
            self.unet_checkpoint = checkpoints_path

        self.save_freq = 1000
        self.print_freq = 10
        self.image_log_freq = 100

        self.iter = 100000
        self.lr = 0.0002
        self.clip_grad = 5

        self.logs_dir = os.path.join("_logs", self.name)
        self.save_dir = os.path.join("_results", self.name)
