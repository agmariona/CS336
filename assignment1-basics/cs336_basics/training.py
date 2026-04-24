import torch
import numpy as np
from typing import Mapping, Any

from .training_utils import data_loader, save_checkpoint, Logger
from .nn_utils import cross_entropy, gradient_clipping

def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.typing.NDArray,
    val_data: np.typing.NDArray,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    device: str | torch.device,
    logger: Logger | None = None,
    start_iteration: int = 0
) -> int:
    config_must_be_positive = [
        "batch_size",
        "context_length",
        "iterations",
        "eval_batches",
        "log_every",
        "eval_every",
        "checkpoint_every"
    ]
    config_must_be_positive_or_none = [
        "max_norm"
    ]
    for key in config_must_be_positive:
        if config[key] <= 0:
            raise ValueError(
                f'config["{key}"] must be positive; got {config[key]}.'
            )
    for key in config_must_be_positive_or_none:
        if config[key] is not None and config[key] <= 0:
            raise ValueError(
                f'config["{key}"] must be positive or None; got {config[key]}.'
            )

    if start_iteration >= config["iterations"]:
        return start_iteration

    # validate and log initial model
    val_loss = evaluate(
        model = model,
        val_data = val_data,
        eval_batches = config["eval_batches"],
        batch_size = config["batch_size"],
        context_length = config["context_length"],
        device = device
    )
    model.train()

    if logger:
        metrics = {'val/loss': val_loss}
        logger.log(metrics, start_iteration)

    # training loop
    for step in range(start_iteration, config["iterations"]):
        iteration = step + 1
        optimizer.zero_grad()

        # get batch
        (train_inputs, train_targets) = data_loader(
            train_data,
            config["batch_size"],
            config["context_length"],
            device
        )

        # forward pass
        logits = model(train_inputs)

        # backward pass
        train_loss = cross_entropy(logits, train_targets)
        train_loss.backward()

        # gradient clipping
        if config["max_norm"] is not None:
            gradient_clipping(model.parameters(), config["max_norm"])

        # optimize
        optimizer.step()

        # evaluate
        if iteration % config["eval_every"] == 0:
            val_loss = evaluate(
                model = model,
                val_data = val_data,
                eval_batches = config["eval_batches"],
                batch_size = config["batch_size"],
                context_length = config["context_length"],
                device = device
            )

            if logger:
                metrics = {'val/loss': val_loss}
                logger.log(metrics, iteration)

        # log
        if logger and iteration % config["log_every"] == 0:
            metrics = {'train/loss': train_loss.item()}
            logger.log(metrics, iteration)

        # checkpoint
        if iteration % config["checkpoint_every"] == 0:
            save_checkpoint(
                model = model,
                optimizer = optimizer,
                iteration = iteration,
                metadata = metadata,
                out = config["checkpoint_path"]
            )

    return iteration


def evaluate(
    model: torch.nn.Module,
    val_data: np.typing.NDArray,
    eval_batches: int,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> float:
    was_training = model.training
    model.eval()

    try:
        val_loss = 0.0
        with torch.no_grad():
            for _ in range(eval_batches):
                (val_inputs, val_targets) = data_loader(
                    val_data,
                    batch_size,
                    context_length,
                    device
                )

                logits = model(val_inputs)
                val_loss += cross_entropy(logits, val_targets).item()
        val_loss /= eval_batches

        return val_loss

    finally:
        model.train(was_training)
