import torch
import torch.nn as nn
from einops import rearrange

from tinygpt.layers.functional import scaled_dot_product_attention
from tinygpt.layers.linear import Linear


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_head = d_model // num_heads

        self.rope = rope

        factory_kwargs = {"device": device, "dtype": dtype}

        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)

        self.output_proj = Linear(d_model, d_model, **factory_kwargs)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        n = x.shape[-2]

        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 2. Split heads: (..., seq, d_model) -> (..., heads, seq, d_head)
        Q = rearrange(Q, "... n (heads d_head) -> ... heads n d_head", heads=self.num_heads)
        K = rearrange(K, "... n (heads d_head) -> ... heads n d_head", heads=self.num_heads)
        V = rearrange(V, "... n (heads d_head) -> ... heads n d_head", heads=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(n, device=x.device)
            elif token_positions.dim() > 1:
                token_positions = token_positions.unsqueeze(-2)  # (batch, seq) -> (batch, 1, seq)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=x.device))

        out = scaled_dot_product_attention(Q, K, V, mask)

        out = rearrange(out, "... heads n d_head -> ... n (heads d_head)")
        return self.output_proj(out)
