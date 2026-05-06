import torch
import triton
import triton.language as tl
from einops import rearrange, einsum
from math import ceil, sqrt, prod

class FlashAttention2(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,            # [B, Nq, d]
        K: torch.Tensor,            # [B, Nk, d]
        V: torch.Tensor,            # [B, Nk, d]
        is_causal: bool = False
    ) -> torch.Tensor:
        if not (Q.device == K.device == V.device):
            raise ValueError(
                "Q/K/V should live on the same device: "
                f"{Q.device=} / {K.device=} / {V.device=}"
            )

        Nq, dq = Q.shape[-2:]
        Nk, dk = K.shape[-2:]
        Nv, dv = V.shape[-2:]

        if not (dq == dk == dv):
            raise ValueError(
                "Q/K/V must have the same feature dim: "
                f"{dq=} / {dk=}"
            )
        d = dq

        if Nk != Nv:
            raise ValueError(
                "K/V must have the same sequence dim: "
                f"{Nk=} / {Nv=}"
            )

        if not (Q.shape[:-2] == K.shape[:-2] == V.shape[:-2]):
            raise ValueError(
                "Q/K/V must have the same batch dims"
            )

        if (Nq % 16 != 0) or (Nk % 16 != 0):
            raise ValueError(
                "Q/K sequence dim must be divisible by 16. "
                f"{Nq=} / {Nk=}"
            )

        Bq, Bk = 16, 16

        Tq = Nq // Bq
        Tk = Nk // Bk

        bsz = Q.shape[:-2]

        O = torch.zeros_like(Q)
        L = Q.new_zeros((*bsz, Nq))

        # computation is batched; annotated are the final two dimensions
        # outer loop over query tiles
        for i in range(Tq):
            Qtile = Q[..., i*Bq : (i+1)*Bq, :]                  # [Bq d]

            Otile = torch.zeros_like(Qtile)                     # [Bq d]
            l = Q.new_zeros(*bsz, Bq)                           # [Bq]
            m = Q.new_full((*bsz, Bq), -torch.inf)              # [Bq]

            # inner loop over key tiles
            for j in range(Tk):
                Ktile = K[..., j*Bk : (j+1)*Bk, :]              # [Bk d]
                Vtile = V[..., j*Bk : (j+1)*Bk, :]              # [Bk d]

                KtileT = rearrange(                             # [d  Bk]
                    Ktile,
                    "... Bk d  -> ... d Bk"
                )
                Stile = (Qtile @ KtileT) / sqrt(d)              # [Bq Bk]

                S_rowmax, _ = torch.max(Stile, -1)              # [Bq]
                m_prev = m                                      # [Bq]
                m, _ = torch.max(                               # [Bq]
                    torch.stack((m, S_rowmax), -1),
                    -1
                )

                Ptile = torch.exp(Stile - m[..., None])        # [Bq Bk]

                P_rowsum = torch.sum(Ptile, -1)                 # [Bq]
                l = torch.exp(m_prev - m) * l + P_rowsum        # [Bq]

                exp_m = torch.exp(m_prev - m)                   # [Bq]
                Otile = Otile * exp_m[..., None] \
                    + Ptile @ Vtile                             # [Bq d]

            Otile = Otile / l[...,  None]                       # [Bq d]
            O[..., i*Bq : (i+1)*Bq, :] = Otile                  # [Bq d]

            Ltile = m + torch.log(l)                            # [Bq]
            L[..., i*Bq : (i+1)*Bq] = Ltile                     # [Bq]

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = flash_attn_backward_compiled(dO, Q, K, V, O, L, is_causal)

        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # identify thread block
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # block_ptrs for query-like tiles
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    # block_ptrs for key-like tiles
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    O_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    m_tile = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    l_tile = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    Q_tile = tl.load(
        Q_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero"
    )

    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_tile = tl.load(
            K_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero"
        )
        V_tile = tl.load(
            V_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero"
        )

        S_tile = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        if is_causal:
            global_q = query_tile_index * Q_TILE_SIZE + \
                tl.arange(0, Q_TILE_SIZE)
            global_k = key_tile_index * K_TILE_SIZE + \
                tl.arange(0, K_TILE_SIZE)
            mask_matrix = global_k[None, :] > global_q[:, None]
            S_tile = tl.where(mask_matrix, -1e6, S_tile)

        S_rowmax = tl.max(S_tile, axis=-1)
        m_tile_prev = m_tile
        m_tile = tl.maximum(m_tile, S_rowmax)

        P_tile = tl.exp(S_tile - m_tile[:, None])
        P_rowsum = tl.sum(P_tile, axis=-1)

        l_tile = tl.exp(m_tile_prev - m_tile) * l_tile + P_rowsum

        exp_m_tile = tl.exp(m_tile_prev - m_tile)
        O_tile = O_tile * exp_m_tile[:, None] + \
            tl.dot(P_tile.to(V_tile.dtype), V_tile)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_tile = O_tile / l_tile[:, None]
    L_tile = m_tile + tl.log(l_tile)

    tl.store(O_block_ptr, O_tile, boundary_check=(0, 1))
    tl.store(L_block_ptr, L_tile, boundary_check=(0,))

def flash_attn_backward(dO, Q, K, V, O, L, is_causal):
    Nq, d = Q.shape[-2:]
    Nk = K.shape[-2]
    scale = 1 / sqrt(d)

    D = torch.sum(O * dO, axis=-1)
    S = einsum(Q, K, "... Nq d, ... Nk d -> ... Nq Nk") * scale

    if is_causal:
        q_idx = torch.arange(Nq, device=Q.device)
        k_idx = torch.arange(Nk, device=K.device)
        mask = k_idx[None, :] > q_idx[:, None]
        S = torch.where(mask, -1e6, S)

    P = torch.exp(S - L[..., None])
    dV = einsum(P, dO, "... Nq Nk, ... Nq d -> ... Nk d")
    dP = einsum(dO, V, "... Nq d, ... Nk d -> ... Nq Nk")
    dS = P * (dP - D[..., None])
    dQ = einsum(dS, K, "... Nq Nk, ... Nk d -> ... Nq d") * scale
    dK = einsum(dS, Q, "... Nq Nk, ... Nq d -> ... Nk d") * scale

    return dQ, dK, dV

flash_attn_backward_compiled = torch.compile(flash_attn_backward)

class FlashAttention2_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        if not (Q.is_cuda and K.is_cuda and V.is_cuda):
            raise ValueError("Expected CUDA tensors")

        if not (Q.shape[:-2] == K.shape[:-2] == V.shape[:-2]):
            raise ValueError("Batch dimension mismatch")

        Nq, dq = Q.shape[-2:]
        Nk, dk = K.shape[-2:]
        Nv, dv = V.shape[-2:]
        batch_dims = Q.shape[:-2]
        n_batch = prod(batch_dims)

        if not (dq == dk == dv):
            raise ValueError("Feature dimension mismatch")

        if Nk != Nv:
            raise ValueError("Sequence dimension mismatch")

        if not (Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()):
            raise ValueError("Expected contiguous tensors")

        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        ctx.N_QUERIES = Nq
        ctx.N_KEYS = Nk
        ctx.D = dq
        ctx.BATCH_DIMS = batch_dims
        ctx.is_causal = is_causal

        O = torch.empty_like(Q)
        L = Q.new_empty((*batch_dims, Nq))

        Q_b = rearrange(Q, "... Nq d -> (...) Nq d")
        K_b = rearrange(K, "... Nk d -> (...) Nk d")
        V_b = rearrange(V, "... Nk d -> (...) Nk d")
        O_b = rearrange(O, "... Nq d -> (...) Nq d")
        L_b = rearrange(L, "... Nq -> (...) Nq")

        flash_fwd_kernel[
            (triton.cdiv(Nq, ctx.Q_TILE_SIZE), n_batch)
        ](
            Q_b, K_b, V_b, O_b, L_b,
            Q_b.stride(0), Q_b.stride(1), Q_b.stride(2),
            K_b.stride(0), K_b.stride(1), K_b.stride(2),
            V_b.stride(0), V_b.stride(1), V_b.stride(2),
            O_b.stride(0), O_b.stride(1), O_b.stride(2),
            L_b.stride(0), L_b.stride(1),
            ctx.N_QUERIES, ctx.N_KEYS,
            1 / sqrt(ctx.D),
            ctx.D,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
            ctx.is_causal
        )

        ctx.save_for_backward(L, Q, K, V, O)

        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV = flash_attn_backward_compiled(dO, Q, K, V, O, L, is_causal)

        return dQ, dK, dV, None

