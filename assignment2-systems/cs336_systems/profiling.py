import torch
from contextlib import nullcontext
from einops import einsum
from math import sqrt

from cs336_basics.nn_utils import softmax

if torch.cuda.is_available():
    import torch.cuda.nvtx as nvtx
else:
    nvtx = None

def nvtx_range(name):
    if nvtx is None:
        return nullcontext()
    return nvtx.range(name)


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,                    # [batch_size ... q_len d_k]
    K: torch.Tensor,                    # [batch_size ... k_len d_k]
    V: torch.Tensor,                    # [batch_size ... k_len d_v]
    mask: torch.Tensor | None = None    # [seq_len seq_len]
) -> torch.Tensor:                      # [batch_size ... q_len d_v]
    with nvtx_range("scaled dot product attention"):
        if mask is not None and mask.dtype != torch.bool:
            raise ValueError('mask must be a boolean tensor')

        d_k = Q.size(-1)

        with nvtx_range("attention score matmul"):
            pre_mask = einsum(Q, K,
                '... q_len d_k, ... k_len d_k -> ... q_len k_len')

        with nvtx_range("scale and mask"):
            pre_mask = pre_mask / sqrt(d_k)

            if mask is not None:
                post_mask = pre_mask.masked_fill(~mask, -torch.inf)
            else:
                post_mask = pre_mask

        with nvtx_range("softmax"):
            S = softmax(post_mask, dim=-1)

        with nvtx_range("final matmul"):
            out = einsum(S, V,
                '... q_len k_len, ... k_len d_v -> ... q_len d_v')

    return out
