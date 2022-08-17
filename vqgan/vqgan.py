"""
Implementing the main VQGAN
"""

# Importing Libraries
from os import stat
from turtle import forward
import torch
import torch.nn as nn

from vqgan import Encoder
from vqgan import Decoder
from vqgan import CodeBook


class VQGAN(nn.Module):
    """
    VQGAN class

    Args:
        img_channels (int, optional): Number of channels in the input image. Defaults to 3.
        img_size (int, optional): Size of the input image. Defaults to 256.
        latent_channels (int, optional): Number of channels in the latent vector. Defaults to 256.
        latent_size (int, optional): Size of the latent vector. Defaults to 16.
        intermediate_channels (list, optional): List of channels in the intermediate layers of encoder and decoder. Defaults to [512, 256, 256, 128, 128].
        num_residual_blocks_encoder (int, optional): Number of residual blocks in the encoder. Defaults to 3.
        num_residual_blocks_decoder (int, optional): Number of residual blocks in the decoder. Defaults to 3.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        attention_resolution (list, optional): Resolution of the attention mechanism. Defaults to [16].
    """

    def __init__(
        self,
        img_channels: int = 3,
        img_size: int = 256,
        latent_channels: int = 256,
        latent_size: int = 16,
        intermediate_channels: list = [512, 256, 256, 128, 128],
        num_residual_blocks_encoder: int = 3,
        num_residual_blocks_decoder: int = 3,
        dropout: float = 0.0,
        attention_resolution: list = [16],
    ):

        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.codebook = CodeBook()

        self.quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a single step of training on the input tensor x

        Args:
            x (torch.Tensor): Input tensor to the encoder.

        Returns:
            torch.Tensor: Output tensor from the decoder.
        """

        x = self.encoder(x)
        quant_x = self.quant_conv(x)

        codebook_mapping, codebook_indices, codebook_loss = self.codebook(quant_x)

        post_quant_x = self.post_quant_conv(codebook_mapping)
        x = self.decoder(post_quant_x)

        return x, codebook_indices, codebook_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:

        x = self.encoder(x)
        quant_x = self.quant_conv(x)

        z_q, min_distance_indices, q_loss = self.codebook(quant_x)

        return z_q, min_distance_indices, q_loss

    def decode(self, x: torch.Tensor) -> torch.Tensor:

        x = self.post_quant_conv(x)
        x = self.decoder(x)

    def calculate_lambda(self, perceptual_loss, gan_loss):
        """Calculating lambda shown in the eq. 7 of the paper

        Args:
            perceptual_loss (torch.Tensor): Perceptual reconstruction loss.
            gan_loss (torch.Tensor): loss from the GAN discriminator.
        """

        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight

        # Because we have multiple loss functions in the networks, retain graph helps to keep the computational graph for backpropagation
        # https://stackoverflow.com/a/47174709
        perceptual_loss_grads = torch.autograd.grad(
            perceptual_loss, last_layer_weight, retain_graph=True
        )[0]
        gan_loss_grads = torch.autograd.grad(
            gan_loss, last_layer_weight, retain_graph=True
        )[0]

        lmda = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lmda = torch.clamp(
            lmda, 0, 1e4
        ).detach()  # Here, we are constraining the value of lambda between 0 and 1e4,

        return 0.8 * lmda  # Note: not sure why we are multiplying it by 0.8... ?

    @staticmethod
    def adopt_weight(
        disc_factor: float, i: int, threshold: int, value: float = 0.0
    ) -> float:
        """Starting the discrimator later in training, so that our model has enough time to generate "good-enough" images to try to "fool the discrimator".

        To do that, we before eaching a certain global step, set the discriminator factor by `value` ( default 0.0 ) .
        This discriminator factor is then used to multiply the discriminator's loss.

        Args:
            disc_factor (float): This value is multiple to the discriminator's loss.
            i (int): The current global step
            threshold (int): The global step after which the `disc_factor` value is retured.
            value (float, optional): The value of discriminator factor before the threshold is reached. Defaults to 0.0.

        Returns:
            float: The discriminator factor.
        """

        if i < threshold:
            disc_factor = value

        return disc_factor

    def load_checkpoint(self, path):
        """Loads the checkpoint from the given path."""

        self.load_state_dict(torch.load(path))
