import math
import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    """Bias-free linear map y = x W^T.

    Weight is stored as (out_features, in_features), matching torch convention and
    the test adapters. Initialized from a truncated normal N(0, 2 / (d_in + d_out)),
    clipped to +-3 std, so activation variance stays roughly constant across layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device": device, "dtype": dtype}

        # Parameter (not a plain tensor) so it is registered: gradient to the optimizer,
        # saved in state_dict, and moved by .to(device).
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    # x: (..., d_in) -> (..., d_out). Leading dims are arbitrary (batch, seq, heads).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """Token ID -> vector lookup table.

    Weight is (num_embeddings, embedding_dim): row i is the vector for token i.
    Initialized from a truncated normal N(0, 1), clipped to +-3.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        torch.nn.init.trunc_normal_(self.weight, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
