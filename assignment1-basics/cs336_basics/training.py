import torch
import numpy as np

def data_loader(
    x: np.typing.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.reshape(-1)

    if len(x) <= context_length:
        raise ValueError(
            f"context length {context_length} must be less than "
            f"data length {len(x)}"
        )

    starts = np.random.choice(
        len(x) - context_length,
        batch_size
    )
    offsets = np.arange(context_length)
    indices = starts[:, None] + offsets[None,:]

    inputs = x[indices]
    targets = x[indices+1]

    inputs_tensor = torch.as_tensor(inputs, device=device, dtype=torch.long)
    targets_tensor = torch.as_tensor(targets, device=device, dtype=torch.long)

    return(inputs_tensor, targets_tensor)
