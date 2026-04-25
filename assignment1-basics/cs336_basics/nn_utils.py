from collections.abc import Iterable
import torch
from einops import rearrange
from math import sqrt

def softmax(
    x: torch.Tensor,
    dim: int
) -> torch.Tensor:
    max_val, _ = x.max(dim=dim, keepdim=True)
    v_exp = torch.exp(x - max_val)
    out = v_exp / torch.sum(v_exp, dim=dim, keepdim=True)

    return out


def cross_entropy(
    pred: torch.Tensor,             # [batch_size ... vocab_size]
    targets: torch.Tensor           # [batch_size ...]
) -> torch.Tensor:
    max_val, _ = pred.max(dim=-1, keepdim=True)
    shifted = pred - max_val

    log_sum_exp = torch.log(
        torch.sum(
            torch.exp(shifted),
            dim = -1
        )
    )

    shifted_targets = rearrange(
        shifted.gather(
            dim = -1,
            index = rearrange(targets, '... -> ... 1')
        ),
        '... 1 -> ...'
    )

    out_mean = torch.mean(
        log_sum_exp - shifted_targets
    )

    return out_mean


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float | None = 1e-6
) -> float:
    parameters = list(parameters)
    norm = gradient_norm(parameters)
    if norm > max_l2_norm:
        for p in parameters:
            if p.grad is None:
                continue

            p.grad.data = p.grad.data * max_l2_norm / (norm + eps)

    # return pre-clipped norm
    return norm

def gradient_norm(
    parameters: Iterable[torch.nn.Parameter]
) -> float:
    norm_accum = 0
    for p in parameters:
        if p.grad is None:
            continue

        norm_accum += torch.linalg.vector_norm(p.grad.data, ord=2)**2
    norm = sqrt(norm_accum)

    return norm
