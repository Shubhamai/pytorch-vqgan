"""
Contains the encoder implementation of VQGAN. 

The encoder is highly inspired by the - Denoising Diffusion Probabilistic Models - https://arxiv.org/abs/2006.11239
According to the official implementation. 
"""

# Importing Libraries
import torch
import torch.nn as nn

from vqgan.common import DownsampleBlock, GroupNorm, NonLocalBlock, ResidualBlock, Swish


class Encoder(nn.Module):
    """
    The encoder part of the VQGAN.

    Args:
        img_channels (int): Number of channels in the input image.
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
        input_size: int = 256,
        latent_channels: int = 256,
        intermediate_channels: list = [128, 128, 256, 256, 512],
        num_residual_blocks: int = 2,
        dropout: float = 0.0,
        attention_resolution: list = [16],
    ):
        super().__init__()

        # Inserting first intermediate channel to index 0
        intermediate_channels.insert(0, intermediate_channels[0])

        # Appends all the layers to this list
        layers = []

        # Addingt the first conv layer increase input channels to the first intermediate channels
        layers.append(
            nn.Conv2d(
                img_channels,
                intermediate_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        # Loop over the intermediate channels except the last one
        for n in range(len(intermediate_channels) - 1):
            in_channels = intermediate_channels[n]
            out_channels = intermediate_channels[n + 1]

            # Adding the residual blocks for each channel
            for _ in range(num_residual_blocks):
                layers.append(ResidualBlock(in_channels, out_channels, dropout=dropout))
                in_channels = out_channels

                # Once we have downsampled the image to the size in attention resolution, we add attention blocks
                if input_size in attention_resolution:
                    layers.append(NonLocalBlock(in_channels))

            # only downsample for the first n-2 layers, and decrease the input size by a factor of 2
            if n != len(intermediate_channels) - 2:
                layers.append(DownsampleBlock(in_channels=intermediate_channels[n + 1]))
                input_size = input_size // 2  # Downsample by a factor of 2

        in_channels = intermediate_channels[-1]
        layers.extend(
            [
                ResidualBlock(
                    in_channels=in_channels, out_channels=in_channels, dropout=dropout
                ),
                NonLocalBlock(in_channels=in_channels),
                ResidualBlock(
                    in_channels=in_channels, out_channels=in_channels, dropout=dropout
                ),
                GroupNorm(in_channels=in_channels),
                Swish(),
                # increase the channels upto the latent vector channels
                nn.Conv2d(
                    in_channels, latent_channels, kernel_size=3, stride=1, padding=1
                ),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
