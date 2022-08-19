# Importing Libraries
from trainer import VQGANTrainer
from dataloader import load_dataloader
from utils import reproducibility
from vqgan import VQGAN

# Reproducibility
reproducibility(42)

model = VQGAN(
            img_channels=1,
            # img_size=img_size,
            # latent_channels=latent_channels,
            # latent_size=latent_size,
            # intermediate_channels=intermediate_channels,
            # num_residual_blocks_encoder=num_residual_blocks_encoder,
            # num_residual_blocks_decoder=num_residual_blocks_decoder,
            # dropout=dropout,
            # attention_resolution=attention_resolution,
            # num_codebook_vectors=num_codebook_vectors,
        ).to("cuda")

# Setting up the trainer
trainer = VQGANTrainer(model=model, img_channels=1)

# Training the model 
dataloader = load_dataloader(name="mnist", save_path="data", batch_size=1, image_size=256, num_workers=4)
trainer.train(epochs=100, dataloader=dataloader)