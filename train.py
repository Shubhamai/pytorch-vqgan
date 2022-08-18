# Importing Libraries
import torch
from torchsummary import summary

from trainer import VQGANTrainer

trainer = VQGANTrainer()

# summary(trainer.vqgan.encoder,  input_data=torch.rand((1, 1, 256, 256)), col_names = ["input_size", "output_size", "num_params"],
#     device = "cuda",
#     depth = 2,)

# summary(trainer.vqgan.decoder,  input_data=torch.rand((1, 256, 16, 16)), col_names = ["input_size", "output_size", "num_params"],
#     device = "cuda",
#     depth = 2,)

trainer.train()