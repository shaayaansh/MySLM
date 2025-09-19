import torch
import torch.nn as nn
from math import sqrt
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):     
        super().__init__()   
        
        
        self.weights = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        std = sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x, self.weights,
            "... d_in, d_out d_in -> ... d_out"
        )
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        embeddings = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.embeddings = nn.Parameter(embeddings)
        nn.init.trunc_normal_(self.embeddings, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids should be of type long.")
        if torch.any((token_ids < 0) | (token_ids >= self.num_embeddings)):
            raise IndexError("token_ids out of range for embedding matrix")
        
        return self.embeddings[token_ids]