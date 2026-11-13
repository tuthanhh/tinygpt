import torch
import math
from einops import einsum


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.amax(x, dim=dim, keepdim=True)
    exps = torch.exp(x - x_max)

    return exps / exps.sum(dim=dim, keepdim=True)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... queries d_k , ... keys d_k -> ... queries keys") * 1 / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = softmax(scores, -1)

    return einsum(attn, V, "... queries keys, ... keys d_v -> ... queries d_v")
