import torch
from pathlib import Path
import numpy as np
import typing
import os
import wandb
from typing import Mapping, Protocol, Any

from .tokenizer import Tokenizer
from .transformer import TransformerLM

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
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    metadata: Mapping[str, Any] | None = None
) -> None:
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "metadata": metadata
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

def load_gen_bundle_from_checkpoint(
    src: str | Path,
    load_device: str | torch.device = "cpu",
) -> (torch.nn.Module, Tokenizer):
    obj = torch.load(src, map_location=load_device)
    model_cfg = obj["metadata"]["model_config"]
    tokenizer_cfg = obj["metadata"]["tokenizer_config"]

    model = TransformerLM(
        vocab_size = model_cfg["vocab_size"],
        context_length = model_cfg["max_seq_length"],
        num_layers = model_cfg["num_layers"],
        d_model = model_cfg["d_model"],
        num_heads = model_cfg["num_heads"],
        d_ff = model_cfg["d_ff"],
        rope_theta = model_cfg["rope_theta"],
        device = load_device
    )
    model.load_state_dict(obj["model"])
    model.eval()

    tokenizer = Tokenizer.from_files(
        vocab_filepath = tokenizer_cfg["vocab_path"],
        merges_filepath = tokenizer_cfg["merges_path"],
        special_tokens = tokenizer_cfg.get("special_tokens")
    )

    return model, tokenizer

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

class WandbLogger:
    def __init__(self, run):
        self.run = run

    def log(self, metrics, step=None):
        self.run.log(metrics, step=step)
