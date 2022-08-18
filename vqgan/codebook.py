"""
https://github.com/dome272/VQGAN-pytorch/blob/main/codebook.py

Contains the implementation of the codebook for the VQGAN. 
With each forward pass, it returns the loss, indices of min distance latent vectors between codebook and encoder output and latent vector with minimim distance. 
"""

# Importing Libraries
import torch
import torch.nn as nn


class CodeBook(nn.Module):
    """
    This is class, we are mostly implemented para 3.1 from the paper,

    We generate the codebook from nn.Embeddings of given size and randomly initialize the weights in uniform distribution.

    The `forward` method is mostly to calculates
    1. the nearest vector in the codebook from the given latent vector by the encoder.
    2. The index of the nearest vector in the codebook.
    3. loss ( from eq. 4 ) ( except reconstruction loss )

    Args:
        num_codebook_vectors (int): Number of codebook vectors.
        latent_dim (int): Latent dimension of individual vectors.
        beta (int): Beta value for the commitment loss.
    """

    def __init__(
        self, num_codebook_vectors: int = 1024, latent_dim: int = 256, beta: int = 0.25
    ):
        super().__init__()

        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim
        self.beta = beta

        # creating the codebook, nn.Embedding here is simply a 2D array mainly for storing our embeddings, it's also learnable
        self.codebook = nn.Embedding(num_codebook_vectors, latent_dim)

        # Initializing the weights in codebook in uniform distribution
        self.codebook.weight.data.uniform_(
            -1 / num_codebook_vectors, 1 / num_codebook_vectors
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss and nearest vector in the codebook from the given latent vector.

        We are mostly implementing the eq 2 and 4 ( except reconstruction loss ) from the paper.

        Args:
            z (torch.Tensor): Latent vector.
        Returns:
            torch.Tensor: Nearest vector in the codebook.
            torch.Tensor: Index of the nearest vector in the codebook.
            torch.Tensor: Loss ( except reconstruction loss ).
        """

        # Channel to last dimension and copying the tensor to store it in a contiguous ( in a sequence ) way
        z = z.permute(0, 2, 3, 1).contiguous()

        z_flattened = z.view(
            -1, self.latent_dim
        )  # b*h*w * latent_dim, will look similar to codebook in fig 2 of the paper

        # calculating the distance between the z to the vectors in flattened codebook, from eq. 2
        # (a - b)^2 = a^2 + b^2 - 2ab
        distance = (
            torch.sum(
                z_flattened**2, dim=1, keepdim=True
            )  # keepdim = True to keep the same original shape after the sum
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2
            * torch.matmul(
                z_flattened, self.codebook.weight.t()
            )  # 2*dot(z, codebook.T)
        )

        # getting indices of vectors with minimum distance from the codebook
        min_distance_indices = torch.argmin(distance, dim=1)

        # getting the corresponding vector from the codebook
        z_q = self.codebook(min_distance_indices).view(z.shape)

        """
        this represent the equation 4 from the paper ( except the reconstruction loss ) . Thia loss will then be added
        to GAN loss to create the final loss function for VQGAN, eq. 6 in the paper.


        Note : In the first para of A. Changlog section of the paper,
        they found a bug which resulted in beta equal to 1. here https://github.com/CompVis/taming-transformers/issues/57
        just a note :)
        """
        loss = torch.mean(
            (z_q.detach() - z) ** 2
            # detach() to avoid calculating gradient while backpropagating
            + self.beta
            * torch.mean(
                (z_q - z.detach()) ** 2
            )  # commitment loss, detach() to avoid calculating gradient while backpropagating
        )

        # Not sure why we need this, but it's in the original implementation and mentions for "preserving gradients"
        z_q = z + (z_q - z).detach()

        # reshapring to the original shape
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_distance_indices, loss
