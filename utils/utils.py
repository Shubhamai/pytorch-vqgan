# Importing Libraries
from torchsummary import summary
import torch


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
    """ Setting up the weights for the discriminator model. 
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
