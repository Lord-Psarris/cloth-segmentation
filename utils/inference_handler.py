import collections

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from networks import U2NET
from dataset.base_dataset import Normalize_image
from utils.checkpoint_handler import load_checkpoint, load_distributed_checkpoint


def get_palette(n: int):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        n: Number of classes
    Returns:
        The color map
    """
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0

        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3

    return palette


def setup_model(checkpoint_path: str, device: str = "cpu"):
    net = U2NET(in_ch=3, out_ch=4)
    if device == "cpu":
        net = load_checkpoint(net, checkpoint_path)
    else:
        net = load_distributed_checkpoint(net, checkpoint_path)

    net = net.to(device)
    net = net.eval()
    return net


def get_mask_array(img: Image.Image, model: nn.Module, device: str = "cpu"):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)

    image = img.convert("RGB")
    # img = img.resize((768, 768), resample=Image.BICUBIC)
    image.thumbnail((768, 768), Image.LANCZOS)
    image_tensor = transform_rgb(image)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output_tensor = model(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()
    return output_arr


def get_inference_results(image: Image.Image):
    """
    color map:
        black:- background
        red:- upper body
        green:- lower body
        yellow:- full body
    """
    # get the colors and count their occurrences
    colors = image.convert('RGB').getcolors(image.size[0] * image.size[1])
    color_counts = collections.Counter()

    for count, color in colors:
        # skip black colors
        if color == (0, 0, 0):
            continue

        color_counts[color] += count

    # getting only 3 instances as only 3 colors can be detected
    detected_colors = color_counts.most_common(3)

    # get colors with count above 1100 to ensure minor errors aren't counted
    colors = [i for i, j in detected_colors if j > 1100]

    # map to colour.
    # Note, this is based on what's generated by the palette in the script, so adjust accordingly
    colors_map = {
        (128, 0, 0): "upper body",  # red
        (0, 128, 0): "lower body",  # green
        (128, 128, 0): "full body",  # yellow
    }

    results = []
    for color in colors:
        results.append(colors_map.get(color))
        # results.append(color)

    return results
