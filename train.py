# Importing Libraries
import torch
from torchsummary import summary

from trainer import VQGANTrainer

# trainer = VQGANTrainer()

from vqgan import Encoder

image = torch.randn(1, 3, 256, 256)

summary(
        Encoder(), input_data=image, col_names=["input_size", "num_params"], device="cpu", depth=2
    )