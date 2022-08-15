# Importing Libraries
import torch
from torchsummary import summary

from trainer import VQGANTrainer
from vqgan import Decoder, Encoder

encoder = Encoder()
decoder = Decoder()

trainer = VQGANTrainer(encoder, decoder)

# sample input
x = torch.randn(1, 3, 256, 256)

# encoder summary
summary(
    encoder,
    input_data=x,
    col_names=["input_size", "output_size", "num_params"],
    device="cpu",
    depth=2,
)


x = trainer.step(x)

print(x.shape)
