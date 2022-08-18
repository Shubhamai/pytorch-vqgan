"""
https://github.com/dome272/VQGAN-pytorch/blob/main/training_vqgan.py
"""

import os

# Importing Libraries
import lpips
import torch
import torch.nn.functional as F
import torchvision

from dataloader import load_dataloader
from utils import weights_init, generate_gif, clean_directory
from vqgan import VQGAN, Discriminator


class VQGANTrainer:
    """Trainer class for VQGAN, contains step, train, and test methods"""

    def __init__(self, device: str or torch.device = "cuda"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.vqgan = VQGAN(img_channels=1).to(self.device)
        self.discriminator = Discriminator(image_channels=1).to(self.device)
        self.discriminator.apply(weights_init)

        self.perceptual_loss = lpips.LPIPS(net="vgg").to(self.device)

        self.dataloader = load_dataloader(
            name="mnist", image_size=256, batch_size=1, save_path="data"
        )

        self.opt_vq, self.opt_disc = self.configure_optimizers()

        # Hyperprameters
        self.global_step = 0
        self.disc_factor = 1.0
        self.disc_start = 100
        self.perceptual_loss_factor = 1.0
        self.rec_loss_factor = 1.0

        # Save directory
        self.expriment_save_dir = "experiments"
        clean_directory(self.expriment_save_dir)


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

    def train(self, epochs: int = 100):
        """Trains the VQGAN for the given number of epochs

        Args:
            epochs (int, optional): number of epochs to train for. Defaults to 100.
        """

        for epoch in range(epochs):
            for index, (imgs, _) in enumerate(self.dataloader):

                # Training step
                imgs = imgs.to(self.device)

                decoded_images, vq_loss, gan_loss = self.step(imgs)

                # Updating global step
                self.global_step += 1

                if index % 20 == 0:

                    with torch.no_grad():
                        real_fake_images = torch.cat((imgs[:4], decoded_images[:4]))
                        torchvision.utils.save_image(
                            real_fake_images,
                            os.path.join(
                                self.expriment_save_dir,
                                "reconstructed_imgs",
                                f"epoch-{epoch}_step-{index}.jpg",
                            ),
                            nrow=4,
                        )

                    print(
                        f"Epoch: {epoch+1}/{epochs} | Batch: {index}/{len(self.dataloader)} | VQ Loss : {vq_loss:.4f} | Discriminator Loss: {gan_loss:.4f}"
                    )

                    if index % 100 == 0:

                        # Saving model
                        torch.save(
                            self.vqgan.state_dict(),
                            os.path.join(
                                self.expriment_save_dir, f"vqgan_epoch_{epoch}.pt"
                            ),
                        )
                        generate_gif(
                            os.path.join(self.expriment_save_dir, "reconstructed_imgs"),
                            os.path.join(
                                self.expriment_save_dir, "reconstructed_imgs.gif"
                            ),
                        )
