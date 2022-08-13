"""
Contains the decoder implementation of VQGAN.

The decoder architecture is also highly inspired by the - Denoising Diffusion Probabilistic Models - https://arxiv.org/abs/2006.11239
According to the official implementation.  
"""

# Importing Libraries
import torch
import torch.nn as nn
from torchsummary import summary

from vqgan.common import (
    GroupNorm,
    NonLocalBlock,
    ResidualBlock,
    Swish,
    UpsampleBlock,
)


class Decoder(nn.Module):
    """
    The decoder part of the VQGAN.

    The implementation is similar to the encoder but inverse, to produce an image from a latent vector. 

    Args:
        img_channels (int): Number of channels in the output image.
        input_size (int): Size of the input tensor (height or width ).
        latent_channels (int): Number of channels in the latent vector.
        intermediate_channels (list): List of channels in the intermediate layers.
        num_residual_blocks (int): Number of residual blocks b/w each downsample block.
        dropout (float): Dropout probability for residual blocks.
        attention_resolution (list): tensor size ( height or width ) at which to add attention blocks
    """

    def __init__(
        self,
        img_channels: int = 3,
        input_size: int = 16,
        latent_channels: int = 256,
        intermediate_channels: list = [512, 256, 256, 128, 128],
        num_residual_blocks: int = 3,
        dropout: float = 0.0,
        attention_resolution: list = [16],
    ):
        super().__init__()

        # Appends all the layers to this list
        layers = []

        # Adding the first conv layer to increase the input channels to the first intermediate channels
        in_channels = intermediate_channels[0]
        layers.extend(
            [
                nn.Conv2d(
                    latent_channels,
                    intermediate_channels[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                ResidualBlock(
                    in_channels=in_channels, out_channels=in_channels, dropout=dropout
                ),
                NonLocalBlock(in_channels=in_channels),
                ResidualBlock(
                    in_channels=in_channels, out_channels=in_channels, dropout=dropout
                ),
            ]
        )

        # Loop over the intermediate channels
        for n in range(len(intermediate_channels)):
            out_channels = intermediate_channels[n]

            # adding the residual blocks
            for _ in range(num_residual_blocks):
                layers.append(ResidualBlock(in_channels, out_channels, dropout=dropout))
                in_channels = out_channels

                # adding the non local block
                if input_size in attention_resolution:
                    layers.append(NonLocalBlock(in_channels))

            # Due to conv in first layer, do not upsample
            if n != 0:
                layers.append(UpsampleBlock(in_channels=in_channels))
                input_size = input_size * 2 # Upsample by a factor of 2

        # Adding rest of the layers
        layers.extend(
            [
                GroupNorm(in_channels=in_channels),
                Swish(),
                nn.Conv2d(
                    in_channels, img_channels, kernel_size=3, stride=1, padding=1
                ),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":

    image = torch.randn(1, 256, 16, 16)
    model = Decoder()

    output = model(image)

    summary(
        model,
        input_data=image,
        col_names=["input_size", "output_size", "num_params"],
        device="cpu",
        depth=2,
    )
