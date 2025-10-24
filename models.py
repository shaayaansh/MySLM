import torch
import math
import torch.nn as nn
from math import sqrt
from einops import rearrange, einsum
import re
from typing import Iterable

class Linear(nn.Module):
    def __init__(self, in_features, out_features, weights=None, device=None, dtype=None):     
        super().__init__()   
        if weights is not None:
            self.weights = weights
        else:
            self.weights = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
            std = sqrt(2 / (in_features + out_features))
            nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x, self.weights,
            "... d_in, d_out d_in -> ... d_out"
        )
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, embeddings=None, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings

        if embeddings is not None:
            self.embeddings = nn.Parameter(embeddings)
        else:
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
    def __init__(self, d_model, dff=None, w1_weight=None, w2_weight=None, w3_weight=None):
        super().__init__()
        self.d_model = d_model

        if dff is not None:
            dff_raw = dff
        else:
            dff_raw = (8 * self.d_model) / 3

        out_features = int(math.ceil(dff_raw/64) * 64)
        self.linear_1 = Linear(d_model, out_features, weights=w1_weight)
        self.linear_2 = Linear(out_features, d_model, weights=w2_weight)
        self.linear_3 = Linear(d_model, out_features, weights=w3_weight)

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
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        theta: float,
        max_seq_len: int,
        q_proj_weight=None, 
        k_proj_weight=None, 
        v_proj_weight=None, 
        o_proj_weight=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // num_heads

        self.theta = theta
        self.max_seq_len = max_seq_len

        self.weights_q = Linear(d_model, num_heads * self.d_k, weights=q_proj_weight)
        self.weights_k = Linear(d_model, num_heads * self.d_k, weights=k_proj_weight)
        self.weights_v = Linear(d_model, num_heads * self.d_k, weights=v_proj_weight)
        self.output_layer = Linear(num_heads * self.d_k, d_model, weights= o_proj_weight)
        self.rope = Rope(theta=self.theta, d_k=self.d_k, max_seq_len=self.max_seq_len)

    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        k_rearr = rearrange(k, "... seq d_k -> ... d_k seq")
        qk = einsum(q, k_rearr, "... seq d_k, ... d_k seq2 -> ... seq seq2")
        d_k = self.d_k
        scaled_qk = qk / math.sqrt(d_k)

        if mask is not None:
            mask = mask.to(q.device)
            scaled_qk = scaled_qk.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scaled_qk, dim=-1)
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

        q = self.rope(q, torch.arange(seq_len))
        k = self.rope(k, torch.arange(seq_len))

        outputs, attn_weights = self._scaled_dot_product_attention(q, k, v, mask=mask)

        outputs = rearrange(outputs, "... nheads seq_len d_k -> ... seq_len (nheads d_k)")
        outputs = self.output_layer(outputs)

        return outputs


def softmax(x: torch.Tensor, dim: int=-1) -> torch.Tensor:
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - max_vals)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float=None, weights: dict[str, torch.Tensor]=None):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        weights_q = weights.get("attn.q_proj.weight") if weights is not None else None
        weights_k = weights.get("attn.k_proj.weight") if weights is not None else None
        weights_v = weights.get("attn.v_proj.weight") if weights is not None else None
        weights_o = weights.get("attn.output_proj.weight") if weights is not None else None
        weights_ff1 = weights.get("ffn.w1.weight") if weights is not None else None
        weights_ff2 = weights.get("ffn.w2.weight") if weights is not None else None
        weights_ff3 = weights.get("ffn.w3.weight") if weights is not None else None
        weights_ln1 = weights.get("ln1.weight") if weights is not None else None
        weights_ln2 = weights.get("ln2.weight") if weights is not None else None

        self.mh_attn = MultiheadSelfAttention(
            self.d_model, 
            self.num_heads, 
            max_seq_len=self.max_seq_len,
            theta=theta,
            q_proj_weight=weights_q, 
            k_proj_weight=weights_k, 
            v_proj_weight=weights_v, 
            o_proj_weight= weights_o
        )
        self.rms_norm1 = RMSNorm(self.d_model, weights=weights_ln1)
        self.rms_norm2 = RMSNorm(self.d_model, weights=weights_ln2)
        self.ff = PositionWiseFeedForward(
            self.d_model, self.d_ff, 
            w1_weight=weights_ff1, 
            w2_weight=weights_ff2, 
            w3_weight=weights_ff3
        )

    def forward(self, x):
        x_norm = self.rms_norm1(x)
        x_attn = self.mh_attn(x_norm)
        x_res = x + x_attn
        x_norm_norm = self.rms_norm2(x_res)
        x_ff = self.ff(x_norm_norm)
        x_out = x_res + x_ff

        return x_out
        

class TransformerLM(nn.Module):
    def __init__(
            self, 
            vocab_size: int, 
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            weights: dict[str, torch.Tensor] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.embeddings = Embedding(self.vocab_size, self.d_model) if weights is None else weights["token_embeddings"]

        weights_ln_final = weights.get("ln_final.weight") if weights is not None else None
        weights_lm_head = weights.get("lm_head.weight") if weights is not None else None

        weights_dict = self._reorganize_weights(weights=weights) if weights is not None else [None] * self.num_layers
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model, 
                    self.num_heads, 
                    self.d_ff, 
                    self.context_length, 
                    self.rope_theta,
                    weights_dict[i]
                ) for i in range(self.num_layers)
            ]
        )

        self.rms_norm = RMSNorm(self.d_model, weights=weights_ln_final)
        self.lm_head = Linear(self.d_model, self.vocab_size, weights=weights_lm_head)

    def forward(self, x):
        embeddings = self.embeddings(x)
        transformer_output = embeddings
        for block in self.transformer_blocks:
            transformer_output = block(transformer_output)

        normalized = self.rms_norm(transformer_output)
        linear_output = self.lm_head(normalized)
        probs = torch.softmax(linear_output, dim=-1)

        return probs


    def _reorganize_weights(self, weights):
        pattern = r"^layers\.(\d+)\."
        weights_dictionary = {}
        for k, v in weights.items():
            if k.startswith("layers."):
                match = re.match(pattern, k)
                if match:
                    layer_num = int(match.group(1))  # e.g., "3"
                    new_key = k.replace(f"layers.{layer_num}.", "")  # strip prefix

                    if layer_num not in weights_dictionary:
                        weights_dictionary[layer_num] = {}

                    weights_dictionary[layer_num][new_key] = v
        return weights_dictionary



def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor):
    """
    logits: (B, V) for classification OR (B, T, V) for seq2seq/language modeling
    targets: (B,) or (B, T) with integer class indices
    """
    if logits.ndim == 3:
        # (B, T, V) → (B*T, V)
        flat_logits = rearrange(logits, 'b t v -> (b t) v')
        flat_targets = rearrange(targets, 'b t -> (b t)')
    elif logits.ndim == 2:
        # (B, V)
        flat_logits, flat_targets = logits, targets
    else:
        raise ValueError(f"Unsupported logits shape {logits.shape}")

    max_values = flat_logits.max(dim=1, keepdim=True).values
    reduced_logits = flat_logits - max_values 
    log_sum_exp = torch.log(torch.sum(torch.exp(reduced_logits), dim=1))

    
    idx = torch.arange(flat_logits.size(0), device=logits.device)
    target_logits = reduced_logits[idx, flat_targets]  

    loss = (log_sum_exp - target_logits).mean()
    return loss


def cosine_annealing_lr(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    """
    Compute the learning rate at iteration t using cosine annealing with warm-up.
    """
    if t < T_w:
        return (t / T_w) * alpha_max
    elif t <= T_c:
        cos_term = math.cos(math.pi * (t - T_w) / (T_c - T_w))
        return alpha_min + 0.5 * (1 + cos_term) * (alpha_max - alpha_min)
    else:
        return alpha_min
    

def clip_gradients(params: Iterable[torch.nn.Parameter], max_norm: float, eps=1e-6):
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = math.sqrt(total_norm)

    clip_coef = max_norm / (total_norm + eps)
    clip_coef = min(clip_coef, 1.0)

    for p in params:
        if p.grad is not None:
            p.grad.data.mul_(clip_coef)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure = None):
        loss = None if closure is None else closure
        
        for group in self.param_groups:
            lr = group['lr'] # get learning rate for param groups
            lr_initial = lr
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.grad.data) # first moment
                    state["v"] = torch.zeros_like(p.grad.data) # second moment
                
                t = state.get("t", 0)
                grad = p.grad.data

                state["m"] = b1 * state["m"] + (1 - b1) * grad
                state["v"] = b2 * state["v"] + (1 - b2) * (grad ** 2)
                lr_t = lr * math.sqrt(1 - b2**(t+1)) / (1 - b1**(t+1))
                
                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps)
                p.data -= lr * wd * p
                state["t"] = t + 1
                
        return loss