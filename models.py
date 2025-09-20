import torch
import math
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
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.d_model = d_model
        self.eps = eps
        gain_tensor = torch.ones(d_model, device=device, dtype=dtype)
        self.gain = nn.Parameter(gain_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_sq = einsum(x**2, "... d_model -> ...") / self.d_model
        rms_x = torch.sqrt(mean_sq + self.eps)
        result = einsum((x / rms_x), self.gain, "... d_model, d_model -> ... d_model")
        result = result.to(in_dtype)

        return result
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        dff_raw = (8 * self.d_model) / 3
        out_features = int(math.ceil(dff_raw/64) * 64)
        self.linear_1 = Linear(d_model, out_features)
        self.linear_2 = Linear(out_features, d_model)
        self.linear_3 = Linear(d_model, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.linear_1(x)
        silu_w1x = einsum(
            w1x, 
            torch.sigmoid(w1x), 
            "... features_out, ... features_out -> ... features_out"
        )
        w3x = self.linear_3(x)
        x = einsum(silu_w1x, w3x, "... d_ff, ... d_ff -> ... d_ff")
        x = self.linear_2(x)

        return x