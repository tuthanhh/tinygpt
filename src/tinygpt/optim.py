import math
from collections.abc import Callable, Iterable
from typing import overload

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # state = self.state[p]

                # t = state.get("t", 0)
                grad = p.grad.data

                p.data -= lr * grad

                # state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:  # first step for this parameter
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)  # same shape, device, dtype as p
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1  # steps count from 1
                t, m, v = state["t"], state["m"], state["v"]

                p.mul_(1 - lr * wd)  # θ = θ·(1 − α·λ)

                m.mul_(b1).add_(grad, alpha=1 - b1)  # m = β1·m + (1−β1)·g
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)  # v = β2·v + (1−β2)·g²

                lr_t = lr * math.sqrt(1 - b2**t) / (1 - b1**t)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)  # θ = θ − α_t·m/(√v + ε)
        return loss


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    device = grads[0].device
    total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, 2).to(device) for g in grads]), 2)

    # Branchless: clip_coef is exactly 1.0 when no clipping is needed.
    clip_coef = torch.clamp(max_l2_norm / (total_norm + eps), max=1.0)
    for g in grads:
        g.mul_(clip_coef.to(g.device, g.dtype))

    return total_norm


def learning_rate_schedule(it: int, max_lr: float, min_lr: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    assert cosine_cycle_iters > warmup_iters, "cosine phase needs at least one step"

    if it < warmup_iters:
        return (it / warmup_iters) * max_lr
    elif it <= cosine_cycle_iters:
        return min_lr + 1 / 2 * (max_lr - min_lr) * (
            1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        )
    else:
        return min_lr
