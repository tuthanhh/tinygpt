from collections.abc import Callable
import torch
from typing import overload


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
