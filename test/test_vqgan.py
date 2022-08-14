"""
Contains test functions for the VQGAN model. 
"""

# Importing Libraries
import torch

from vqgan import Encoder, Decoder


def test_encoder():

    input_channels = 3
    input_size = 256
    latent_channels = 256
    attn_resolution = 16

    image = torch.randn(1, input_channels, input_size, input_size)

    model = Encoder(
        img_channels=input_channels,
        input_size=input_size,
        latent_channels=latent_channels,
        attention_resolution=[attn_resolution],
    )

    output = model(image)

    assert output.shape == (
        1,
        latent_channels,
        attn_resolution,
        attn_resolution,
    ), "Output of encoder does not match"


def test_decoder():

    img_channels = 3
    input_size = 16
    img_size = 256
    latent_channels = 256
    attn_resolution = 16

    latent = torch.randn(1, latent_channels, input_size, input_size)
    model = Decoder(
        img_channels=img_channels,
        input_size=input_size,
        latent_channels=latent_channels,
        attention_resolution=[attn_resolution],
    )

    output = model(latent)

    assert output.shape == (
        1,
        img_channels,
        img_size,
        img_size,
    ), "Output of decoder does not match"
