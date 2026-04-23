import torch
from einops import rearrange

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

