import torch
import numpy as np
import typing
import os
import wandb
from typing import Mapping, Protocol, Any

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

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(obj, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]

class Logger(Protocol):
    def log(
        self,
        metrics: Mapping[str, Any],
        iteration: int | None = None
    ) -> None:
        ...

class StdoutLogger:
    def log(
        self,
        metrics: Mapping[str, Any],
        iteration: int | None = None
    ) -> None:
        print(iteration, metrics)
