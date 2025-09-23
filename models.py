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
    def __init__(self, d_model: int, eps: float = 1e-5, weights=None, device=None, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.d_model = d_model
        self.eps = eps

        if weights is not None:
        gain_tensor = weights
        else:
        gain_tensor = torch.ones(self.d_model, device=device, dtype=dtype)
        
        self.gain = nn.Parameter(gain_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_sq = (x**2).mean(dim=-1, keepdim=True)
        rms_x = torch.sqrt(mean_sq + self.eps)
        result = (x / rms_x) * self.gain
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
    


class Rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        k = torch.arange(d_k // 2, device=device)
        freqs = self.theta ** (-2.0 * k / float(self.d_k))

        pos = torch.arange(max_seq_len, device=device)
        phases = einsum(pos, freqs, "max_len, d_k_2 -> max_len d_k_2")
        self.sins = torch.sin(phases)
        self.cosines = torch.cos(phases)
        
        self.register_buffer("rope_cosines", self.cosines, persistent=False)
        self.register_buffer("rope_sins", self.sins, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sins = self.rope_sins[token_positions] # (seq_len, d_k/2)
        cosines = self.rope_cosines[token_positions] # (seq_len, d_k/2)

        x_even = x[..., 0::2] # (b, seq_len, d_k/2)
        x_odd = x[..., 1::2] # (b, seq_len, d_k/2)

        x_even_rot = einsum(x_even, cosines, "... seq_len d_k_2, ... seq_len d_k_2 -> ... seq_len d_k_2") - \
            einsum(x_odd, sins, "... seq_len d_k_2, ... seq_len d_k_2 -> ... seq_len d_k_2") # (b, seq_len, d_k/2)
        x_odd_rot = einsum(x_even, sins, "... seq_len d_k_2, ... seq_len d_k_2 -> ... seq_len d_k_2") + \
            einsum(x_odd, cosines, "... seq_len d_k_2, ... seq_len d_k_2 -> ... seq_len d_k_2") # (b, seq_len, d_k/2)
        
        combined = torch.cat([x_even_rot[..., None], x_odd_rot[..., None]], dim=-1)
        x_rot = rearrange(combined, "... d_k two -> ... (d_k two)")

        return x_rot


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // num_heads

        self.weights_q = nn.Linear(d_model, num_heads * self.d_k)
        self.weights_k = nn.Linear(d_model, num_heads * self.d_k)
        self.weights_v = nn.Linear(d_model, num_heads * self.d_k)
        self.output_layer = nn.Linear(num_heads * self.d_k, d_model)

    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        qk = einsum(q, k, "... seq d_k, ... seq2 d_k -> ... seq seq2")
        d_k = self.d_k
        scaled_qk = qk / math.sqrt(d_k)

        if mask is not None:
            scaled_qk = scaled_qk.masked_fill(mask, float("-inf"))

        attn_weights = softmax(scaled_qk, dim=-1)
        output = einsum(attn_weights, v, "... seq seq2, ... seq2 d_v -> ... seq d_v")

        return output, attn_weights

    def forward(self, x):
        seq_len = x.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        q = self.weights_q(x) # (b, seq, nheads*d_k)
        k = self.weights_k(x) # (b, seq, nheads*d_k)
        v = self.weights_v(x) # (b, seq, nheads*d_k)

        q = rearrange(q, "... seq_len (nheads d_k) -> ... nheads seq_len d_k", nheads=self.num_heads) # (b, nheads, seq, d_k)
        k = rearrange(k, "... seq_len (nheads d_k) -> ... nheads seq_len d_k", nheads=self.num_heads) # (b, nheads, seq, d_k)
        v = rearrange(v, "... seq_len (nheads d_k) -> ... nheads seq_len d_k", nheads=self.num_heads) # (b, nheads, seq, d_k)

        outputs, attn_weights = self._scaled_dot_product_attention(q, k, v, mask=mask)

        outputs = rearrange(outputs, "... nheads seq_len d_k -> ... seq_len (nheads d_k)")
        outputs = self.output_layer(outputs)

        return outputs


def softmax(x: torch.Tensor, dim: int=-1) -> torch.Tensor:
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - max_vals)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)