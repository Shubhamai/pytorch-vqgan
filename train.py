# Importing Libraries
import argparse
import os
from unicodedata import name

from aim import Run

import yaml

from dataloader import load_dataloader
from trainer import VQGANTrainer
from utils import reproducibility
from vqgan import VQGAN


def main(config: dict):
    """
    Main function for training the VQGAN ( Stage 1 )
    """

    # Reproducibility
    reproducibility(config["seed"])

    model = VQGAN(**config["model"]).to(config["device"])

    # Experiment tracker
    run = Run(experiment=config['name'])
    run["hparams"] = config

    # Setting up the trainer
    trainer = VQGANTrainer(
        model=model, run=run, device=config["device"], **config["trainer"], 
    )

    # Training the model
    dataloader = load_dataloader(
        **config["dataloader"],
        image_size=config["model"]["img_size"],
    )
    trainer.train(
        epochs=config["epochs"],
        dataloader=dataloader,
        limit_steps=config["limit_steps"],
    )

    # Saving the model
    model.save_checkpoint(os.path.join(config["trainer"]["experiment_dir"], "model.pt"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/mnist.yml",
        help="path to config file",
    )

    args = parser.parse_args()
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
