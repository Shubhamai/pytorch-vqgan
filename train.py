# Importing Libraries
import torch
from trainer import Trainer

from vqgan import Encoder, Decoder

encoder = Encoder()
decoder = Decoder()

trainer = Trainer(encoder, decoder)

# sample input
x = torch.randn(1, 3, 256, 256)
x = trainer.step(x)

print(x.shape)
