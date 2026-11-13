import torch


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # logits: (..., vocab)   target: (...) int64
    row_max = torch.amax(logits, dim=-1, keepdim=True)  # (..., 1)
    shifted = logits - row_max  # (..., vocab), max entry is 0

    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))  # (...)
    target_logit = shifted.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (...)

    return (log_sum_exp - target_logit).mean()
