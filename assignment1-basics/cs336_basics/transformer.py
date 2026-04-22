import torch
import torch.nn as nn
from einops import reduce, einsum, rearrange
from math import sqrt


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,                           # d_model
        eps: float = 1e-5,                      # for numerical stability
        device: torch.device | None = None,     # device to store params
        dtype: torch.dtype | None = None        # param dtype
    ) -> None:
        super().__init__()

        G = torch.ones(
                [d_model],
                device=device,
                dtype=dtype
            )

        self.G = nn.Parameter(G)
        self.d_model = d_model
        self.eps = eps


    def forward(
        self,
        x: torch.Tensor                 # [batch_size seq_length d_model]
    ) -> torch.Tensor:                  # [batch_size seq_length d_model]
        in_type = x.dtype
        x = x.to(torch.float32)

        mean_sq = reduce(x.pow(2), '... d_model -> ... 1', 'mean')
        rrms = torch.rsqrt(mean_sq + self.eps)

        rmsnorm = x * rrms * self.G

        return rmsnorm.to(in_type)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        if d_ff is None:
            # (8/3)*d_model, to the nearest multiple of 64
            d_ff = round(((8/3)*d_model) / 64) * 64

        W1 = torch.empty(
                [d_ff, d_model],
                device=device,
                dtype=dtype
            )
        W3 = torch.empty(
                [d_ff, d_model],
                device=device,
                dtype=dtype
            )
        W2 = torch.empty(
                [d_model, d_ff],
                device=device,
                dtype=dtype
            )

        init_std = sqrt(2 / (d_ff + d_model))
        nn.init.trunc_normal_(W1, std=init_std, a=-3*init_std, b=3*init_std)
        nn.init.trunc_normal_(W2, std=init_std, a=-3*init_std, b=3*init_std)
        nn.init.trunc_normal_(W3, std=init_std, a=-3*init_std, b=3*init_std)

        self.W1 = nn.Parameter(W1)
        self.W2 = nn.Parameter(W2)
        self.W3 = nn.Parameter(W3)
        self.d_model = d_model
        self.d_ff = d_ff


    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        W1x = einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        W3x = einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        GLU = self._silu(W1x) * W3x
        out = einsum(GLU, self.W2, "... d_ff, d_model d_ff -> ... d_model")

        return out


    def _silu(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return x * torch.special.expit(x)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,               # dim of query and key vectors
        max_seq_len: int,
        device: torch.device | None = None
    ) -> None:
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError('d_k must be even')

        positions = torch.arange(max_seq_len, device=device)
        positions = rearrange(positions, 'i -> i 1')
        pairs = torch.arange(d_k // 2, device=device) + 1
        angle_rates = theta ** ((2*pairs - 2) / d_k)
        angle_rates = rearrange(angle_rates, 'k -> 1 k')
        angles = positions / angle_rates

        sin_table = torch.sin(angles)
        cos_table = torch.cos(angles)

        self.register_buffer("sin_table", sin_table, persistent=False)
        self.register_buffer("cos_table", cos_table, persistent=False)

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

    def forward(
        self,
        x: torch.Tensor,                    # [... seq_len d_k]
        token_positions: torch.Tensor       # [... seq_len]
    ) -> torch.Tensor:
        x = rearrange(x,
                '... seq_len (n_pairs p) -> ... seq_len n_pairs p',
                p=2
            )
        sin_th = self.sin_table[token_positions]
        cos_th = self.cos_table[token_positions]

        out_a = cos_th * x[..., 0] - sin_th * x[..., 1]
        out_b = sin_th * x[..., 0] + cos_th * x[..., 1]

        out = torch.stack([out_a, out_b], dim=-1)
        out = rearrange(out,
                '... seq_len n_pairs p -> ... seq_len (n_pairs p)',
                p=2
            )

        return out


def softmax(
    x: torch.Tensor,
    dim: int
) -> torch.Tensor:
    max_val, _ = x.max(dim=dim, keepdim=True)
    v_exp = torch.exp(x - max_val)
    out = v_exp / torch.sum(v_exp, dim=dim, keepdim=True)

    return out

def scaled_dot_product_attention(
    Q: torch.Tensor,                    # [batch_size ... q_len d_k]
    K: torch.Tensor,                    # [batch_size ... k_len d_k]
    V: torch.Tensor,                    # [batch_size ... k_len d_v]
    mask: torch.Tensor | None = None    # [seq_len seq_len]
) -> torch.Tensor:                      # [batch_size ... q_len d_v]
    if mask is not None and mask.dtype != torch.bool:
        raise ValueError('mask must be a boolean tensor')

    d_k = Q.size(-1)
    pre_mask = einsum(Q, K,
        '... q_len d_k, ... k_len d_k -> ... q_len k_len')
    pre_mask = pre_mask / sqrt(d_k)

    if mask is not None:
        post_mask = pre_mask.masked_fill(~mask, -torch.inf)
    else:
        post_mask = pre_mask

    out = einsum(softmax(post_mask, dim=-1), V,
        '... q_len k_len, ... k_len d_v -> ... q_len d_v')

    return out
