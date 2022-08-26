# Importing Libraries
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from aim import Run


class TransformerTrainer:
    def __init__(
        self,
        model: nn.Module,
        run: Run,
        experiment_dir: str = "experiments",
        device: str = "cuda",
    ):

        self.experiment_dir = experiment_dir

        self.model = model
        self.device = device
        self.optim = self.configure_optimizers()

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": 0.01,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=4.5e-06, betas=(0.9, 0.95))
        return optimizer

    def train(self, dataloader: torch.utils.data.DataLoader, epochs: int):
        for epoch in range(epochs):

            for index, imgs in enumerate(dataloader):
                self.optim.zero_grad()
                imgs = imgs.to(device=self.device)
                logits, targets = self.model(imgs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
                loss.backward()
                self.optim.step()

                print(
                    f"Epoch: {epoch+1}/{epochs} | Batch: {index}/{len(dataloader)} | Cross Entropy Loss : {loss:.4f}"
                )
