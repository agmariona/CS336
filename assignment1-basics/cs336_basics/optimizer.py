from collections.abc import Callable, Iterable
from typing import Optional
import torch
from math import sqrt, cos, pi


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ) -> None:
        if lr < 0:
            raise ValueError("Learning rate lr must be nonnegative")
        if betas[0] < 0 or betas[0] >= 1 or betas[1] < 0 or betas[1] >= 1:
            raise ValueError("Hyperparameters beta must be in [0,1)")
        if eps <= 0:
            raise ValueError("Hyperparameter eps must be positive")
        if weight_decay <= 0:
            raise ValueError("Weight decay rate must be positive")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            bta1 = betas[0]
            bta2 = betas[1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 1)
                grad = p.grad.data

                # adjusted lr
                lr_t = lr * sqrt(1 - bta2**t) / (1 - bta1**t)

                # weight decay
                p.data -= lr * weight_decay * p.data

                # update moment estimates
                m = bta1 * m + (1 - bta1) * grad
                v = bta2 * v + (1 - bta2) * grad**2

                # apply moment-adjusted weight update
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss


def lr_cosine_schedule(
    it: int,
    max_lr: float,
    min_lr: float,
    warmup_its: int,
    cos_cycle_its: int
) -> float:
    if warmup_its <= 0:
        raise ValueError("Warm-up iteration count must be positive")
    if warmup_its == cos_cycle_its:
        raise ValueError(
            "Final cosine annealing iteration must be greater than "
            "warm-up iteration count"
        )

    if it < warmup_its:
        return (it / warmup_its) * max_lr
    elif it <= cos_cycle_its:
        return min_lr + \
            (1/2)*(1 + cos(
                pi * (it - warmup_its) / (cos_cycle_its - warmup_its)
            )) * (max_lr - min_lr)
    else:
        return min_lr
