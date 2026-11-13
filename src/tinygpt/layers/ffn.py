from tinygpt.layers.functional import silu
import torch
import torch.nn as nn
from tinygpt.layers.linear import Linear
import math


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 64 * math.ceil(8 * d_model / 3 / 64)

        self.d_model = d_model
        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}

        self.w1 = Linear(d_model, d_ff, **factory_kwargs)  # gate branch
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)  # value branch
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)  # back down to d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
