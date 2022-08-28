# Importing Libraries
import argparse

import yaml

from trainer import Trainer
from transformer import VQGANTransformer
from vqgan import VQGAN


def main(args, config):

    vqgan = VQGAN(**config["architecture"]["vqgan"])
    vqgan.load_checkpoint("./experiments/checkpoints/vqgan.pt")

    transformer = VQGANTransformer(
        vqgan, device=args.device, **config["architecture"]["transformer"]
    )
    transformer.load_checkpoint("./experiments/checkpoints/transformer.pt")
    
    trainer = Trainer(
        vqgan,
        transformer,
        run=None,
        config=config["trainer"],
        seed=args.seed,
        device=args.device,
    )

    trainer.generate_images()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/default.yml",
        help="path to config file",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["mnist", "cifar", "custom"],
        default="mnist",
        help="Dataset for the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to train the model on",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=42,
        help="Seed for Reproducibility",
    )

    args = parser.parse_args()

    args = parser.parse_args()
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(args, config)
