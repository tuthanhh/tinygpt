import torch
import torch.nn as nn

from tinygpt.layers.attention import MultiheadAttention
from tinygpt.layers.ffn import SwiGLU
from tinygpt.layers.linear import Embedding, Linear
from tinygpt.layers.norm import RMSNorm
from tinygpt.layers.rope import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ln1 = RMSNorm(d_model, 1e-5, **factory_kwargs)
        self.attn = MultiheadAttention(d_model, num_heads, rope, **factory_kwargs)
        self.ln2 = RMSNorm(d_model, 1e-5, **factory_kwargs)
        self.ffn = SwiGLU(d_model, d_ff, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(self.ln1(x)) + x
        out = self.ffn(self.ln2(h)) + h

        return out


class TransformerLM(nn.Module):
    """token IDs (batch, seq) -> logits (batch, seq, vocab_size)."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)

        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device=device)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    rope=self.rope,
                    **factory_kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model, **factory_kwargs)
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq), integer
        # TODO (later): assert seq <= self.context_length

        x = self.token_embeddings(token_ids)

        for block in self.layers:
            x = block(x)

        x = self.ln_final(x)
        return self.lm_head(x)
