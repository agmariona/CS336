import torch
from einops import rearrange
from math import ceil, sqrt

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

        # computation is batched; final two dimensions are annotated
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

                Ptile = torch.exp(Stile - m[..., :, None])      # [Bq Bk]

                P_rowsum = torch.sum(Ptile, -1)                 # [Bq]
                l = torch.exp(m_prev - m) * l + P_rowsum        # [Bq]

                exp_m = torch.exp(m_prev - m)                   # [Bq]
                Otile = Otile * exp_m[..., :, None] \
                    + Ptile @ Vtile                             # [Bq d]

            Otile = Otile / l[..., :, None]                     # [Bq d]
            O[..., i*Bq : (i+1)*Bq, :] = Otile                  # [Bq d]

            Ltile = m + torch.log(l)                            # [Bq]
            L[..., i*Bq : (i+1)*Bq] = Ltile                     # [Bq]

        ctx.save_for_backward(L, Q, K, V, O)
        return O


    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
