import argparse
import yaml
from pathlib import Path
from typing import Any
import numpy as np
import torch

from .transformer import TransformerLM
from .optimizer import AdamW
from .training_utils import load_checkpoint, StdoutLogger
from .training import train

def load_cfg(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(
            "YAML config file must contain a top-level mapping."
        )
    return config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_cfg(args.config)

    training_cfg = config["training"]
    data_cfg = config["data"]
    optimizer_cfg = config["optimizer"]
    runtime_cfg = config["runtime"]
    model_cfg = config["model"]

    if training_cfg["context_length"] > model_cfg["max_seq_length"]:
        raise ValueError(
            f'Training context length {training_cfg["context_length"]} '
            'must be less than model maximum sequence length '
            f'{model_cfg["max_seq_length"]}'
        )

    # set seed
    seed = runtime_cfg.get("seed")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # map training / validation data
    train_memmap = np.load(data_cfg["train_path"], mmap_mode = 'r')
    val_memmap = np.load(data_cfg["val_path"], mmap_mode = 'r')

    # construct model
    model = TransformerLM(
        vocab_size = model_cfg["vocab_size"],
        context_length = model_cfg["max_seq_length"],
        num_layers = model_cfg["num_layers"],
        d_model = model_cfg["d_model"],
        num_heads = model_cfg["num_heads"],
        d_ff = model_cfg["d_ff"],
        rope_theta = model_cfg["rope_theta"],
        device = runtime_cfg["device"]
    )

    # construct optimizer
    optimizer = AdamW(
        params = model.parameters(),
        lr = optimizer_cfg["lr"],
        betas = optimizer_cfg["betas"],
        eps = optimizer_cfg["eps"],
        weight_decay = optimizer_cfg["weight_decay"]
    )

    # construct logger
    logger_type = runtime_cfg.get("logger")
    if logger_type is None or logger_type == 'stdout':
        logger = StdoutLogger()
    else:
        raise ValueError(
            f'Unsupported logger: {logger_type}'
        )

    # load from checkpoint if given
    resume_from = runtime_cfg.get("resume_from")
    if resume_from is not None:
        start_iteration = load_checkpoint(resume_from, model, optimizer)
    else:
        start_iteration = 0

    train(
        model = model,
        optimizer = optimizer,
        train_data = train_memmap,
        val_data = val_memmap,
        config = training_cfg,
        device = runtime_cfg["device"],
        logger = logger,
        start_iteration = start_iteration
    )


if __name__ == "__main__":
    main()
