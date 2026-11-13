from einops import reduce
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        factory_kwargs = {"device": device, "dtype": dtype}

        # This is the "learnable parameters" in the description
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_dtype = x.dtype

        x = x.to(torch.float32)

        inv_rms = torch.rsqrt(reduce(x**2, "... d_model -> ... 1", "mean") + self.eps)
        result = x * inv_rms * self.weight
        return result.to(inp_dtype)
