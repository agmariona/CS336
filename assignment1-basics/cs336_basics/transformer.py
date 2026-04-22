import torch
import torch.nn as nn
from einops import reduce

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
        x: torch.Tensor                 # [batch_size, seq_length, d_model]
    ) -> torch.Tensor:                  # [batch_size, seq_length, d_model]
        in_type = x.dtype
        x = x.to(torch.float32)

        mean_sq = reduce(x.pow(2), '... d_model -> ... 1', 'mean')
        rrms = torch.rsqrt(mean_sq + self.eps)

        rmsnorm = x * rrms * self.G

        return rmsnorm.to(in_type)
