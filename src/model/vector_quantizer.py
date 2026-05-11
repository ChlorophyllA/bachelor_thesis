import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """Simple non-EMA vector quantizer. Operates on feature maps of shape
    [B, 1, C, T] or [B, C, T]. Quantizes along channel dimension C for each time-step.
    Returns quantized tensor with same shape and a scalar loss.
    """
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        # z: [B, 1, C, T] or [B, C, T]
        orig_shape = z.shape
        if z.dim() == 4:
            # support either [B,1,C,T] (singleton channel axis) or [B,C,n_chann,T]
            b, d1, d2, T = z.shape
            if d1 == 1:
                # [B,1,C,T] -> flatten to [B*T, C]
                C = d2
                z_flat = z.permute(0, 3, 2, 1).contiguous().view(-1, C)
            else:
                # assume [B,C,n_chann,T] -> quantize across C for each (n_chann, T)
                C = d1
                z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B, n_chann, T, C] -> [B*n_chann*T, C]
        elif z.dim() == 3:
            b, C, T = z.shape
            z_flat = z.permute(0, 2, 1).contiguous().view(-1, C)
        else:
            raise ValueError(f"Unsupported z.dim()={z.dim()}")

        if C != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch: input C={C}, quantizer emb_dim={self.embedding_dim}")

        # compute distances
        # z_flat: [N, D], embedding.weight: [M, D]
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
            + self.embedding.weight.pow(2).sum(1)
        )  # [N, M]

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(z_flat.shape)

        # reshape back
        if z.dim() == 4:
            if d1 == 1:
                # original [B,1,C,T]
                quantized = quantized.view(b, T, C).permute(0, 2, 1).unsqueeze(1)  # [B,1,C,T]
            else:
                # original [B,C,n_chann,T]
                quantized = quantized.view(b, d2, T, C).permute(0, 3, 1, 2)  # [B,C,n_chann,T]
        else:
            quantized = quantized.view(b, T, C).permute(0, 2, 1)  # [B,C,T]

        # losses
        q_latent_loss = F.mse_loss(quantized.detach(), z)
        e_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, loss
