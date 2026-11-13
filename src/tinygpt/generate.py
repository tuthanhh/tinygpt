from collections.abc import Iterator

import torch.nn as nn
import torch

def sample_next(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> int:
    # 3. T = 0 means greedy: dividing by 0 is undefined, argmax is its limit.
    if temperature == 0:
        return int(logits.argmax())

    # 3-4. Temperature, then probabilities. float32 so a bf16 model still samples cleanly.
    probs = torch.softmax(logits.float() / temperature, dim=-1)

    # 5. Top-p: smallest set of most likely tokens whose total reaches top_p.
    sorted_probs, sorted_ids = probs.sort(descending=True)
    if top_p < 1.0:
        cumulative = sorted_probs.cumsum(dim=-1)
        # Mass BEFORE each token = cumulative - own prob. Drop a token once the tokens
        # above it already reach top_p. The top token has 0 before it, so it is never
        # dropped: no all-masked row, no NaN (the classic `cumsum > p` off-by-one).
        drop = (cumulative - sorted_probs) >= top_p
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum()

    # 6. Sample on the CPU (V numbers, cheap): a CPU generator then works whatever
    #    device the model is on, and a seed reproduces the same output on GPU or CPU.
    choice = torch.multinomial(sorted_probs.cpu(), num_samples=1, generator=generator)
    return int(sorted_ids.cpu()[choice])  # sorted position -> token ID


@torch.no_grad
def generate(
    model:nn.Module,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eot_id: int,
    context_length: int,
    generator: torch.Generator | None = None,
) -> Iterator[int]:
    """Yields new token IDs one at a time (not the prompt, not EOT).

    Stream:   for tid in generate(...): ...
    Collect:  ids = list(generate(...))
    """
    assert temperature >= 0, "temperature must be >= 0"
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"

    device = next(model.parameters()).device

    ids = list(prompt_ids) or [eot_id]

    for _ in range(max_new_tokens):
        # 1. RoPE tables cover context_length positions: feed at most that many.
        x = torch.tensor([ids[-context_length:]], dtype=torch.long, device=device)  # (1, seq)

        # 2. Only the last position predicts a token we don't have yet.
        logits = model(x)[0, -1, :]  # (V,)

        next_id = sample_next(logits, temperature, top_p, generator)

        # 7. End of story.
        if next_id == eot_id:
            return
        ids.append(next_id)

        yield next_id
