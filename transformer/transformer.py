# Importing Libraries
import torch
import torch.nn as nn
import torch.functional as F
from transformer.mingpt import GPT
from vqgan.vqgan import VQGAN

class VQGANTransformer(nn.Module):

    def __init__(self, vqgan_checkpoint_path:str, num_codebook_vectors:int=512, sos_token:int=0, pkeep:float=0.5):
        super().__init__()

        self.sos_token = sos_token

        self.vqgan = VQGAN().load_checkpoint(vqgan_checkpoint_path)
        self.vqgan.eval()

        self.transformer = GPT(vocab_size=num_codebook_vectors, block_size=512, n_layer=24, n_head=16, n_embd=1024)

        self.pkeep = pkeep

    @torch.no_grad()
    def encode_to_z(self, x:torch.tensor) -> torch.tensor:
        """ Processes the input batch ( containing images ) to encoder and returning flattened quantized encodings 

        Args:
            x (torch.tensor): the input batch b*c*h*w

        Returns:
            torch.tensor: the flattened quantized encodings
        """
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices:torch.tensor, p1:int=16, p2:int=16) -> torch.Tensor:
        """ Returns the decoded image from the indices for the codebook embeddings 

        Args:
            indices (torch.tensor): the indices of the vectors in codebook to use for generating the decoder output
            p1 (int, optional): encoding size. Defaults to 16.
            p2 (int, optional): encoding size. Defaults to 16.

        Returns:
            torch.tensor: generated image from decoder 
        """

        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 256)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image

    def forward(self, x):
        
        # Getting the codebook indices of the image
        _, indices = self.encode_to_z(x)

        # sos tokens, this will be needed when we will generate new and unseen images
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        # Generating a matrix of shape indices with 1s and 0s
        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device)) # torch.bernoulli([0.5 ... 0.5]) -> [1, 0, 1, 1, 0, 0] ; p(1) - 0.5
        mask = mask.round().to(dtype=torch.int64)

        # Generate a vector containing randomlly indices
        random_indices = torch.randint_like(indices, high=self.transformer.config.vocab_size) # generating indices from the distribution
        
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

    def top_k_logits(self, logits:torch.Tensor, k:int) -> torch.Tensor:
        """ 

        Args:
            logits (torch.Tensor): _description_
            k (int): _description_

        Returns:
            torch.Tensor: _description_
        """
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        for k in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x):
        log = dict()

        _, indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        start_indices = indices[:, :indices.shape[1] // 2]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1])
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

