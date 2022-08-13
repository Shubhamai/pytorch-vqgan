# Importing Libraries
import torch
import torch.nn as nn


class Trainer:
    """Trainer class for VQGAN, contains step, train, and test methods"""

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:

        self.encoder = encoder
        self.decoder = decoder

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """ Performs a single step of training on the input tensor x 

        Args:
            x (torch.Tensor): Input tensor to the encoder.

        Returns:
            torch.Tensor: Output tensor from the decoder.
        """

        x = self.encoder(x)
        x = self.decoder(x)

        return x
