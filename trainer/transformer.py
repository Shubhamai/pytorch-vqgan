# Importing Libraries
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from transformer import VQGANTransformer


class TransformerTrainer:
    def __init__(
        self,
        vqgan_checkpoint_path: str,
        experiment_dir: str = "experiments",
        device: str = "cuda",
    ):

        self.experiment_dir = experiment_dir

        self.model = VQGANTransformer(
            vqgan_checkpoint_path,
            num_codebook_vectors=512,
            sos_token=0,
            pkeep=0.5,
            device=device,
        ).to(device)
        self.optim = self.configure_optimizers()

        self.train()

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

    def train(self, epochs: int, device: str, dataloader: torch.utils.data.DataLoader):
        for epoch in range(epochs):
            with tqdm(range(len(dataloader))) as pbar:
                for i, imgs in zip(pbar, dataloader):
                    self.optim.zero_grad()
                    imgs = imgs.to(device=device)
                    logits, targets = self.model(imgs)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                    )
                    loss.backward()
                    self.optim.step()
                    pbar.set_postfix(
                        Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4)
                    )
                    pbar.update(0)
            log, sampled_imgs = self.model.log_images(imgs[0][None])
            torchvision.utils.save_image(
                sampled_imgs,
                os.path.join(self.experiment_dir, f"transformer_{epoch}.jpg"),
                nrow=4,
            )
            # plot_images(log)
            torch.save(
                self.model.state_dict(),
                os.path.join(self.experiment_dir, "checkpoints", f"transformer_{epoch}.pt"),
            )
