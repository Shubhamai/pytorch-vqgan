# Importing Libraries
import torch

from dataloader import load_mnist, load_cifar10


def load_dataloader(
    name: str = "mnist",
    batch_size: int = 2,
    image_size: int = 256,
    num_workers: int = 4,
    save_path: str = "data",
) -> torch.utils.data.DataLoader:
    """Load the data loader for the given name.

    Args:
        name (str, optional): The name of the data loader. Defaults to "mnist".
        batch_size (int, optional): The batch size. Defaults to 2.
        image_size (int, optional): The image size. Defaults to 256.
        num_workers (int, optional): The number of workers to use for the dataloader. Defaults to 4.
        save_path (str, optional): The path to save the data to. Defaults to "data".

    Returns:
        torch.utils.data.DataLoader: The data loader.
    """

    if name == "mnist":
        return load_mnist(
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            save_path=save_path,
        )

    elif name == "cifar10":
        return load_cifar10(
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            save_path=save_path,
        )