import argparse
from pathlib import Path
from typing import Any
import numpy as np
import torch
import wandb
import json

from .transformer import TransformerLM
from .optimizer import AdamW
from .training_utils import *
from .training import train

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
    tokenizer_cfg = config["tokenizer"]

    if training_cfg["context_length"] > model_cfg["max_seq_length"]:
        raise ValueError(
            f'Training context length {training_cfg["context_length"]} '
            'must be less than model maximum sequence length '
            f'{model_cfg["max_seq_length"]}'
        )

    with open(tokenizer_cfg["vocab_path"], 'r', encoding='utf-8') as f:
        vocab_len = len(json.load(f))

    if vocab_len != model_cfg["vocab_size"]:
        raise ValueError(
            f'Model vocab size {model_cfg["vocab_size"]} does not match ',
            f'tokenizer vocab size {vocab_len}'
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

    # load from checkpoint if given
    resume_from = runtime_cfg.get("resume_from")
    if resume_from is not None:
        # TODO: check that config matches checkpoint model
        start_iteration = load_checkpoint(resume_from, model, optimizer)
    else:
        start_iteration = 0

    # construct logger
    if runtime_cfg.get("logger") == "stdout":
        logger = StdoutLogger()
    elif runtime_cfg.get("logger") == "wandb":
        wandb_cfg = config["wandb"]
        run = wandb.init(
            project = wandb_cfg["project"],
            entity = wandb_cfg.get("entity"),
            name = wandb_cfg.get("name"),
            mode = wandb_cfg.get("mode", "online"),
            group = wandb_cfg.get("group"),
            tags = wandb_cfg.get("tags")
        )
        logger = WandbLogger(run)
    else:
        logger = None

    train(
        model = model,
        optimizer = optimizer,
        train_data = train_memmap,
        val_data = val_memmap,
        train_cfg = training_cfg,
        optim_cfg = optimizer_cfg,
        metadata = {
            "model_config": model_cfg,
            "tokenizer_config": tokenizer_cfg
        },
        device = runtime_cfg["device"],
        logger = logger,
        start_iteration = start_iteration
    )

    if runtime_cfg.get("logger") == "wandb":
        run.finish()


if __name__ == "__main__":
    main()
