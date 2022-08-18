"""
Contains test functions for the VQGAN model. 
"""

# Importing Libraries
import torch

from vqgan import Encoder, Decoder, CodeBook, Discriminator


def test_encoder():

    image_channels = 3
    image_size = 256
    latent_channels = 256
    attn_resolution = 16

    image = torch.randn(1, image_channels, image_size, image_size)

    model = Encoder(
        img_channels=image_channels,
        image_size=image_size,
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
    img_size = 256
    latent_channels = 256
    latent_size = 16
    attn_resolution = 16

    latent = torch.randn(1, latent_channels, latent_size, latent_size)
    model = Decoder(
        img_channels=img_channels,
        latent_size=latent_size,
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


def test_codebook():

    z = torch.randn(1, 256, 16, 16)

    codebok = CodeBook(num_codebook_vectors=100, latent_dim=16)

    z_q, min_distance_indices, loss = codebok(z)

    assert z_q.shape == (1, 256, 16, 16), "Output of codebook does not match"


def test_discriminator():

    image = torch.randn(1, 3, 256, 256)

    model = Discriminator()


    output = model(image)

    assert output.shape == (1, 1, 30, 30), "Output of discriminator does not match"
