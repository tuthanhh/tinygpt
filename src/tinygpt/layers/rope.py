import torch
import torch.nn as nn
from einops import einsum, rearrange


class RotaryPositionalEmbedding(nn.Module):
    """RoPE, handout 3.4.3. Rotates neighbor pairs (0,1), (2,3), ... of the last dim.
    No learnable parameters: cos/sin tables are non-persistent buffers.
    """

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert d_k % 2 == 0, f"RoPE rotates pairs of dims, so d_k must be even (got {d_k})"
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 1.1 One frequency per pair: theta^(-2k/d_k), k = 0 .. d_k/2 - 1  -> (d_k/2,)
        two_k = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = theta ** (-two_k / d_k)

        # 1.2 Positions 0 .. max_seq_len-1  -> (max_seq_len,)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # 1.3 Outer product: angle[i, k] = i * inv_freq[k]  -> (max_seq_len, d_k/2)
        angles = einsum(positions, inv_freq, "i, k -> i k")

        # 1.4 Derived from config, not learned: keep out of state_dict
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq, d_k)    token_positions: (..., seq) integer

        # 2.1 Look up each position's row  -> (..., seq, d_k/2)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # 2.2 Pair halves, each (..., seq, d_k/2); element j of each = pair j
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 2.3 Rotate every pair by its angle (eq. 8)
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        # 2.4 Interleave back to [e0, o0, e1, o1, ...]
        stacked = torch.stack([rot_even, rot_odd], dim=-1)  # (..., seq, d_k/2, 2)
        out = rearrange(stacked, "... half two -> ... (half two)")  # (..., seq, d_k)

        # float32 tables may have promoted a bf16/fp16 input; return in x's dtype
        return out.to(x.dtype)
