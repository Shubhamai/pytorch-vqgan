# Importing Libraries
import glob
import os
import random
import shutil

import imageio
import numpy as np
import torch
from torchsummary import summary


def print_summary(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    col_names: list = ["input_size", "output_size", "num_params"],
    device: str = "cpu",
    depth: int = 2,
):
    """
    Prints a summary of the model.
    """
    return summary(
        model, input_data=input_data, col_names=col_names, device=device, depth=depth
    )


def weights_init(m):
    """Setting up the weights for the discriminator model.
    This is mentioned in the original PatchGAN paper, in page 16, section 6.2 - Training Details

    ```
    All networks were trained from scratch. Weights were initialized from a Gaussian distribution with mean 0 and
    standard deviation 0.02.
    ```

    Image-to-Image Translation with Conditional Adversarial Network - https://arxiv.org/pdf/1611.07004v3.pdf

    Args:
        m
    """

    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


def generate_gif(imgs_path: str, save_path: str):
    """Generates a gif from a directory of images.

    Args:
        imgs_path: Path to the directory of images.
        save_path: Path to save the gif.
    """

    with imageio.get_writer(save_path, mode="I") as writer:
        for filename in glob.glob(imgs_path + "/*.jpg"):
            image = imageio.imread(filename)
            writer.append_data(image)


def clean_directory(directory: str):
    """Cleans a directory.
    Args:
        directory: Path to the directory.
    """

    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)


def reproducibility(seed: int = 42):
    """Set the random seed.

    Args:
        seed (int): The seed to use.

    Returns:
        None
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    np.random.seed(seed)
    random.seed(seed)


def collate_fn(batch):
    """
    Collate function for the dataloader like mnist or cifar10.
    """

    imgs = torch.stack([img[0] for img in batch])

    return imgs
