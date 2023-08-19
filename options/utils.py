from typing import Union, Optional

from .main import Parser

import yaml
import os


def generate_options_directories(options: Parser):
    os.makedirs(options.logs_dir, exist_ok=True)
    os.makedirs(options.save_dir, exist_ok=True)
    os.makedirs(os.path.join(options.save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(options.save_dir, "checkpoints"), exist_ok=True)


def generate_yaml_file(options: Parser, path: Optional[str] = None):
    path = "training_options.yml" if not path else path
    option_dict = vars(options)
    with open(os.path.join(options.save_dir, path), "w") as outfile:
        yaml.dump(option_dict, outfile)
