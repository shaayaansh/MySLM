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