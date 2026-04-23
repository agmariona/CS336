import torch
import torch.nn as nn
from einops import reduce, einsum, rearrange
from math import sqrt
from .basic_layers import *
from .nn_utils import softmax


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,                           # d_model
        eps: float = 1e-5,                      # for numerical stability
        device: torch.device | None = None,     # device to store params
        dtype: torch.dtype | None = None        # param dtype
    ) -> None:
        super().__init__()

        weight = torch.ones(
                [d_model],
                device=device,
                dtype=dtype
            )

        self.weight = nn.Parameter(weight)
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

        rmsnorm = x * rrms * self.weight

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

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

        self.d_model = d_model
        self.d_ff = d_ff

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.w2(self._silu(self.w1(x)) * self.w3(x))

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


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError('num_heads must divide d_model')

        d_k = d_model // num_heads                  # we assume d_k = d_v

        if rope is not None and rope.d_k != d_k:
            raise ValueError('RoPE and MHSA must use same d_k')

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(
            d_model, d_model, device=device, dtype=dtype)

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k

        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,                                # [... seq_len d_model]
        token_positions: torch.Tensor | None = None     # [... seq_len]
    ) -> torch.Tensor:                                  # [... seq_len d_model]
        if token_positions is None and self.rope is not None:
            raise ValueError('token_positions required for MHSA with RoPE')

        seq_len = x.size(-2)
        all_true = torch.ones(
            [seq_len, seq_len],
            dtype=torch.bool,
            device=x.device
        )
        mask = ~torch.triu(all_true, diagonal=1)

        Qx = rearrange(self.q_proj(x),
                '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k',
                num_heads = self.num_heads, d_k = self.d_k
            )
        Kx = rearrange(self.k_proj(x),
                '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k',
                num_heads = self.num_heads, d_k = self.d_k
            )
        Vx = rearrange(self.v_proj(x),
                '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k',
                num_heads = self.num_heads, d_k = self.d_k
            )

        if self.rope is not None:
            Qx = self.rope(Qx, token_positions)
            Kx = self.rope(Kx, token_positions)

        mhsa = scaled_dot_product_attention(Qx, Kx, Vx, mask)
        mhsa = rearrange(mhsa,
                '... num_heads seq_len d_k -> ... seq_len (num_heads d_k)')

        out = self.output_proj(mhsa)

        return out

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, rope, device=device, dtype=dtype)
        self.ln1= RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,        # [... seq_length d_model]
    ) -> torch.Tensor:          # [... seq_length d_model]
        seq_length = x.size(-2)
        token_positions = rearrange(
            torch.arange(seq_length, device=x.device),
            'seq_length -> 1 seq_length'
        ).expand(*x.shape[:-1])
        y = x + self.attn(self.ln1(x), token_positions)
        z = y + self.ffn(self.ln2(y))

        return z


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        rope = RotaryPositionalEmbedding(
            rope_theta, d_model // num_heads, context_length,
            device=device)
        self.rope = rope
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(vocab_size, d_model,
            device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope,
                device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size,
            device=device, dtype=dtype)

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

    def forward(
        self,
        x: torch.Tensor,    # [batch_size seq_len]
    ) -> torch.Tensor:      # [batch_size seq_len vocab_size]
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

