from trainer import Trainer
from vqgan import VQGAN
from transformer import VQGANTransformer
from dataloader import load_mnist
from aim import Run


vqgan = VQGAN(img_channels=1)
transformer = VQGANTransformer(vqgan)
dataloader = load_mnist()

run = Run(experiment="mnist")


trainer = Trainer(vqgan, transformer, run=run, config={"vqgan":{}, "transformer":{}})

trainer.train_vqgan(dataloader)

trainer.train_transformers(dataloader)

trainer.generate_images()