"""
https://github.com/dome272/VQGAN-pytorch/blob/main/training_vqgan.py
"""

# Importing Libraries
import os

import imageio
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from aim import Image, Run
from utils import clean_directory, weights_init
from vqgan import Discriminator


class VQGANTrainer:
    """Trainer class for VQGAN, contains step, train methods"""

    def __init__(
        self,
        model: torch.nn.Module,
        run: Run,
        # Training parameters
        device: str or torch.device = "cuda",
        learning_rate: float = 2.25e-05,
        beta1: float = 0.5,
        beta2: float = 0.9,
        # Loss parameters
        perceptual_loss_factor: float = 1.0,
        rec_loss_factor: float = 1.0,
        # Discriminator parameters
        disc_factor: float = 1.0,
        disc_start: int = 100,
        # Miscellaneous parameters
        experiment_dir: str = "./experiments",
        perceptual_model: str = "vgg",
        save_every: int = 10,
    ):

        self.run = run
        self.device = device

        # VQGAN parameters
        self.vqgan = model

        # Discriminator parameters
        self.discriminator = Discriminator(image_channels=self.vqgan.img_channels).to(self.device)
        self.discriminator.apply(weights_init)

        # Loss parameters
        self.perceptual_loss = lpips.LPIPS(net=perceptual_model).to(self.device)

        # Optimizers
        self.opt_vq, self.opt_disc = self.configure_optimizers(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2
        )

        # Hyperprameters
        self.disc_factor = disc_factor
        self.disc_start = disc_start
        self.perceptual_loss_factor = perceptual_loss_factor
        self.rec_loss_factor = rec_loss_factor

        # Save directory
        self.expriment_save_dir = experiment_dir

        # Miscellaneous
        self.global_step = 0
        self.sample_batch = None
        self.gif_images = []
        self.save_every = save_every

    def configure_optimizers(
        self, learning_rate: float = 2.25e-05, beta1: float = 0.5, beta2: float = 0.9
    ):
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters())
            + list(self.vqgan.decoder.parameters())
            + list(self.vqgan.codebook.parameters())
            + list(self.vqgan.quant_conv.parameters())
            + list(self.vqgan.post_quant_conv.parameters()),
            lr=learning_rate,
            eps=1e-08,
            betas=(beta1, beta2),
        )
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            eps=1e-08,
            betas=(beta1, beta2),
        )

        return opt_vq, opt_disc

    def step(self, imgs: torch.Tensor) -> torch.Tensor:
        """Performs a single training step from the dataloader images batch

        For the VQGAN, it calculates the perceptual loss, reconstruction loss, and the codebook loss and does the backward pass.

        For the discriminator, it calculates lambda for the discriminator loss and does the backward pass.

        Args:
            imgs: input tensor of shape (batch_size, channel, H, W)

        Returns:
            decoded_imgs: output tensor of shape (batch_size, channel, H, W)
        """

        # Getting decoder output
        decoded_images, _, q_loss = self.vqgan(imgs)

        """
        =======================================================================================================================
        VQ Loss
        """
        perceptual_loss = self.perceptual_loss(imgs, decoded_images)
        rec_loss = torch.abs(imgs - decoded_images)
        perceptual_rec_loss = (
            self.perceptual_loss_factor * perceptual_loss
            + self.rec_loss_factor * rec_loss
        )
        perceptual_rec_loss = perceptual_rec_loss.mean()

        """
        =======================================================================================================================
        Discriminator Loss
        """
        disc_real = self.discriminator(imgs)
        disc_fake = self.discriminator(decoded_images)

        disc_factor = self.vqgan.adopt_weight(
            self.disc_factor, self.global_step, threshold=self.disc_start
        )

        g_loss = -torch.mean(disc_fake)

        λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
        vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

        d_loss_real = torch.mean(F.relu(1.0 - disc_real))
        d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
        gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

        # ======================================================================================================================
        # Tracking metrics

        self.run.track(perceptual_rec_loss, name="Perceptual & Reconstruction loss", step=self.global_step)

        self.run.track(vq_loss, name="VQ Loss", step=self.global_step)
        self.run.track(gan_loss, name="GAN Loss", step=self.global_step)

        # =======================================================================================================================
        # Backpropagation

        self.opt_vq.zero_grad()
        vq_loss.backward(
            retain_graph=True
        )  # retain_graph is used to retain the computation graph for the discriminator loss

        self.opt_disc.zero_grad()
        gan_loss.backward()

        self.opt_vq.step()
        self.opt_disc.step()

        return decoded_images, vq_loss, gan_loss

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 1,
        limit_steps: int = -1,
    ):
        """Trains the VQGAN for the given number of epochs

        Args:
            dataloader (torch.utils.data.DataLoader): dataloader to use.
            epochs (int, optional): number of epochs to train for. Defaults to 100.
            limit_iterations (int, optional): training the model on a limited number of iterations. Defaults to -1, meaning all step from the dataloader.
        """

        for epoch in range(epochs):
            for index, imgs in enumerate(dataloader):

                # Training step
                imgs = imgs.to(self.device)

                decoded_images, vq_loss, gan_loss = self.step(imgs)

                # Updating global step
                self.global_step += 1

                if limit_steps != -1 and self.global_step >= limit_steps:
                    print("[INFO] Maximum number of steps reached, exiting")
                    break

                if index % self.save_every == 0:

                    print(
                        f"Epoch: {epoch+1}/{epochs} | Batch: {index}/{len(dataloader)} | VQ Loss : {vq_loss:.4f} | Discriminator Loss: {gan_loss:.4f}"
                    )

                    # Only saving the gif for the first 2000 save steps
                    if self.global_step // self.save_every <= 2000:
                        self.sample_batch = (
                            imgs[:] if self.sample_batch is None else self.sample_batch
                        )

                        with torch.no_grad():

                            gif_img = (
                                torchvision.utils.make_grid(
                                    torch.cat(
                                        (
                                            self.sample_batch,
                                            self.vqgan(self.sample_batch)[0],
                                        ),
                                    )
                                )
                                .detach()
                                .cpu()
                                .permute(1, 2, 0)
                                .numpy()
                            )

                            gif_img = (gif_img - gif_img.min()) * (
                                255 / (gif_img.max() - gif_img.min())
                            )
                            gif_img = gif_img.astype(np.uint8)

                            self.run.track(
                                Image(gif_img), name="VQGAN Reconstruction", step=self.global_step
                            )

                            self.gif_images.append(gif_img)

                        imageio.mimsave(
                            os.path.join(self.expriment_save_dir, "reconstruction.gif"),
                            self.gif_images,
                            fps=5,
                        )
