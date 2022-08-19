# Importing Libraries
import torch
import torchvision

from utils import collate_fn


def load_mnist(
    batch_size: int = 16,
    image_size: int = 28,
    num_workers: int = 4,
    save_path: str = "data",
) -> torch.utils.data.DataLoader:
    """Load the MNIST data and returns the dataloaders (train ). The data is downloaded if it does not exist.

    Args:
        batch_size (int): The batch size.
        image_size (int): The image size.
        num_workers (int): The number of workers to use for the dataloader.
        save_path (str): The path to save the data to.

    Returns:
        torch.utils.data.DataLoader: The data loader.
    """

    # Load the data
    mnist_data = torchvision.datasets.MNIST(
        root=save_path,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    dataloader = torch.utils.data.DataLoader(
        mnist_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return dataloader
