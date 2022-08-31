# Importing Libraries
import os

import torch
import torchvision
from aim import Run
from utils import reproducibility

from trainer import TransformerTrainer, VQGANTrainer


class Trainer:
    def __init__(
        self,
        vqgan: torch.nn.Module,
        transformer: torch.nn.Module,
        run: Run,
        config: dict,
        experiment_dir: str = "experiments",
        seed: int = 42,
        device: str = "cuda",
    ) -> None:

        self.vqgan = vqgan
        self.transformer = transformer

        self.run = run
        self.config = config
        self.experiment_dir = experiment_dir
        self.seed = seed
        self.device = device

        print(f"[INFO] Setting seed to {seed}")
        reproducibility(seed)

        print(f"[INFO] Results will be saved in {experiment_dir}")
        self.experiment_dir = experiment_dir

    def train_vqgan(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["vqgan"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan.pt")
        )

    def train_transformers(
        self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "transformer.pt")
        )

    def generate_images(self, n_images: int = 5):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.transformer = self.transformer.to(self.device)


        for i in range(n_images):
            start_indices = torch.zeros((4, 0)).long().to(self.device)
            sos_tokens = torch.ones(start_indices.shape[0], 1) * 0

            sos_tokens = sos_tokens.long().to(self.device)
            sample_indices = self.transformer.sample(
                start_indices, sos_tokens, steps=256
            )
            sampled_imgs = self.transformer.z_to_image(sample_indices)
            torchvision.utils.save_image(
                sampled_imgs,
                os.path.join(self.experiment_dir, f"generated_{i}.jpg"),
                nrow=4,
            )

