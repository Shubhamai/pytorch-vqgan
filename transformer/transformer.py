"""
https://github.com/dome272/VQGAN-pytorch/blob/main/transformer.py
"""

# Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.mingpt import GPT


class VQGANTransformer(nn.Module):
    def __init__(
        self,
        vqgan: nn.Module,
        device: str = "cuda",
        sos_token: int = 0,
        pkeep: float = 0.5,
        block_size: int = 512,
        n_layer: int = 12,
        n_head: int = 16,
        n_embd: int = 1024,
    ):
        super().__init__()

        self.sos_token = sos_token
        self.device = device

        self.vqgan = vqgan

        self.transformer = GPT(
            vocab_size=self.vqgan.num_codebook_vectors,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )

        self.pkeep = pkeep

    @torch.no_grad()
    def encode_to_z(self, x: torch.tensor) -> torch.tensor:
        """Processes the input batch ( containing images ) to encoder and returning flattened quantized encodings

        Args:
            x (torch.tensor): the input batch b*c*h*w

        Returns:
            torch.tensor: the flattened quantized encodings
        """
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(
        self, indices: torch.tensor, p1: int = 16, p2: int = 16
    ) -> torch.Tensor:
        """Returns the decoded image from the indices for the codebook embeddings

        Args:
            indices (torch.tensor): the indices of the vectors in codebook to use for generating the decoder output
            p1 (int, optional): encoding size. Defaults to 16.
            p2 (int, optional): encoding size. Defaults to 16.

        Returns:
            torch.tensor: generated image from decoder
        """

        ix_to_vectors = self.vqgan.codebook.codebook(indices).reshape(
            indices.shape[0], p1, p2, 256
        )
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        transformer model forward pass 

        Args:
            x (torch.tensor): Batch of images
        """

        # Getting the codebook indices of the image
        _, indices = self.encode_to_z(x)

        # sos tokens, this will be needed when we will generate new and unseen images
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(self.device)

        # Generating a matrix of shape indices with 1s and 0s
        mask = torch.bernoulli(
            self.pkeep * torch.ones(indices.shape, device=indices.device)
        )  # torch.bernoulli([0.5 ... 0.5]) -> [1, 0, 1, 1, 0, 0] ; p(1) - 0.5
        mask = mask.round().to(dtype=torch.int64)

        # Generate a vector containing randomlly indices
        random_indices = torch.randint_like(
            indices, high=self.transformer.config.vocab_size
        )  # generating indices from the distribution

        """
        indices - [3, 56, 72, 67, 45, 53, 78, 90]
        mask - [1, 1, 0, 0, 1, 1, 1, 0]
        random_indices - 15, 67, 27, 89, 92, 40, 91, 10]

        new_indices - [ 3, 56,  0,  0, 45, 53, 78,  0] + [ 0,  0, 27, 89,  0,  0,  0, 10] => [ 3, 56, 27, 89, 45, 53, 78, 10]
        """
        new_indices = mask * indices + (1 - mask) * random_indices

        # Adding sos ( start of sentence ) token
        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        logits, _ = self.transformer(new_indices[:, :-1])

        return logits, target

    def top_k_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """

        Args:
            logits (torch.Tensor): predictions from the transformer
            k (int): returning k highest values

        Returns:
            torch.Tensor: retuning tensor of same dimension as input keeping the top k entries
        """
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float(
            "inf"
        )  # Setting all values except in topk to inf
        return out

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        steps: int = 256,
        temperature: float = 1.0,
        top_k: int = 100,
    ) -> torch.Tensor:
        """Generating sample indices from the transformer

        Args:
            x (torch.Tensor): the batch of images
            c (torch.Tensor): sos token 
            steps (int, optional): the lenght of indices to generate. Defaults to 256.
            temperature (float, optional): hyperparameter for minGPT model. Defaults to 1.0.
            top_k (int, optional): keeping top k entries. Defaults to 100.

        Returns:
            torch.Tensor: _description_
        """

        self.transformer.eval()

        x = torch.cat((c, x), dim=1)  # Appending sos token
        for k in range(steps):
            logits, _ = self.transformer(x)  # Getting the predicted indices
            logits = (
                logits[:, -1, :] / temperature
            )  # Getting the last prediction and scaling it by temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(
                probs, num_samples=1
            )  # Note : not sure what's happening here

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1] :]  # Removing the sos token
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x:torch.Tensor):
        """ Generating images using the transformer and decoder. Also uses encoder to complete partial images.   

        Args:
            x (torch.Tensor): batch of images

        Returns:
            Retures the input and generated image in dictionary and in a simple concatenated image
        """
        log = dict()

        _, indices = self.encode_to_z(x) # Getting the indices of the quantized encoding
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(self.device)

        start_indices = indices[:, : indices.shape[1] // 2]
        sample_indices = self.sample(
            start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1]
        )
        half_sample = self.z_to_image(sample_indices)

        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log, torch.concat((x, x_rec, half_sample, full_sample))

    def load_checkpoint(self, path):
        """Loads the checkpoint from the given path."""

        self.load_state_dict(torch.load(path))

    def save_checkpoint(self, path):
        """Saves the checkpoint to the given path."""

        torch.save(self.state_dict(), path)
