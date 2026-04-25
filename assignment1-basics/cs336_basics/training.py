import torch
import numpy as np
from typing import Mapping, Any
import time
import math

from .training_utils import data_loader, save_checkpoint, Logger
from .nn_utils import cross_entropy, gradient_clipping, gradient_norm
from .optimizer import lr_cosine_schedule


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.typing.NDArray,
    val_data: np.typing.NDArray,
    train_cfg: Mapping[str, Any],
    optim_cfg: Mapping[str, Any],
    metadata: Mapping[str, Any],
    device: str | torch.device,
    logger: Logger | None = None,
    start_iteration: int = 0
) -> int:
    # train_cfg validation
    train_cfg_must_be_positive = [
        "batch_size",
        "context_length",
        "iterations",
        "eval_batches",
        "log_every",
        "eval_every",
        "checkpoint_every"
    ]

    train_cfg_must_be_positive_or_none = [
        "max_norm",
        "divergence_threshold"
    ]

    for key in train_cfg_must_be_positive:
        if train_cfg[key] <= 0:
            raise ValueError(
                f'train_cfg["{key}"] must be positive; got {train_cfg[key]}.'
            )

    for key in train_cfg_must_be_positive_or_none:
        value = train_cfg.get(key)
        if value is not None and value <= 0:
            raise ValueError(
                f'train_cfg["{key}"] must be positive or None; got {value}.'
            )

    batch_size              = train_cfg["batch_size"]
    context_length          = train_cfg["context_length"]
    eval_batches            = train_cfg["eval_batches"]
    max_iterations          = train_cfg["iterations"]
    checkpoint_path         = train_cfg["checkpoint_path"]
    log_every               = train_cfg["log_every"]
    eval_every              = train_cfg["eval_every"]
    checkpoint_every        = train_cfg["checkpoint_every"]
    divergence_threshold    = train_cfg.get("divergence_threshold")
    max_norm                = train_cfg.get("max_norm")

    lr                      = optim_cfg["lr"]
    lr_sched_cfg            = optim_cfg.get("lr_schedule")
    if lr_sched_cfg is not None:
        lr_sched            = lr_sched_cfg["type"]
    else:
        lr_sched            = None

    if lr_sched not in [None, "constant", "cosine"]:
        raise ValueError(
            f"Unknown learning rate schedule {lr_sched}"
        )

    if lr_sched == "cosine":
        max_lr              = lr_sched_cfg["max_lr"]
        min_lr              = lr_sched_cfg["min_lr"]
        warmup_iters        = lr_sched_cfg["warmup_iters"]
        cosine_cycle_iters  = lr_sched_cfg["cosine_cycle_iters"]

    if start_iteration >= max_iterations:
        return start_iteration

    if start_iteration > 0:
        # assumes previous run used same batch_size and context_length
        tokens_processed = start_iteration * batch_size * context_length
    else:
        tokens_processed = 0

    # validate and log initial model
    val_loss = evaluate(
        model = model,
        val_data = val_data,
        eval_batches = eval_batches,
        batch_size = batch_size,
        context_length = context_length,
        device = device
    )
    model.train()

    if logger:
        metrics = {
            "val/loss": val_loss,
            "val/perplexity": perplexity(val_loss),
            "time/elapsed_sec": 0.0,
            "train/tokens_processed": tokens_processed
        }
        logger.log(metrics, start_iteration)

    # training loop
    train_start_time = time.perf_counter()
    tokens_this_run = 0
    for step in range(start_iteration, max_iterations):
        iteration = step + 1
        optimizer.zero_grad()
        metrics = {}

        # get batch
        (train_inputs, train_targets) = data_loader(
            train_data,
            batch_size,
            context_length,
            device
        )

        # set learning rate
        for group in optimizer.param_groups:
            if lr_sched == "cosine":
                curr_lr = lr_cosine_schedule(
                    it = iteration,
                    max_lr = max_lr,
                    min_lr = min_lr,
                    warmup_its = warmup_iters,
                    cos_cycle_its = cosine_cycle_iters
                )
            else:
                curr_lr = lr
            group["lr"] = curr_lr

        # forward pass
        logits = model(train_inputs)
        train_loss = cross_entropy(logits, train_targets)
        loss_item = train_loss.item()

        # check for divergence
        if (
            not math.isfinite(loss_item)
            or (
                divergence_threshold is not None
                and loss_item > divergence_threshold
            )
        ):
            elapsed_sec = time.perf_counter() - train_start_time

            metrics["run/diverged"] = 1
            metrics["train/loss"] = loss_item
            metrics["train/perplexity"] = perplexity(loss_item)
            metrics["train/tokens_processed"] = tokens_processed
            metrics["time/elapsed_sec"] = elapsed_sec
            metrics["optim/lr"] = curr_lr

            if logger:
                logger.log(metrics, iteration)

            return iteration

        # backward pass
        train_loss.backward()

        # gradient clipping
        clipped = 0
        if max_norm is not None:
            grad_norm = gradient_clipping(model.parameters(), max_norm)
            if grad_norm > max_norm:
                clipped = 1
        else:
            grad_norm = gradient_norm(model.parameters())

        # optimize
        optimizer.step()

        elapsed_sec = time.perf_counter() - train_start_time
        tokens_processed += batch_size * context_length
        tokens_this_run += batch_size * context_length

        # training metrics
        if iteration % log_every == 0:
            metrics["time/elapsed_sec"] = elapsed_sec
            metrics["train/loss"] = loss_item
            metrics["train/perplexity"] = perplexity(loss_item)
            metrics["train/tokens_processed"] = tokens_processed
            metrics["perf/tokens_per_sec"] = tokens_this_run / elapsed_sec
            metrics["optim/grad_norm"] = grad_norm
            metrics["optim/grad_clipped"] = clipped
            metrics["optim/lr"] = curr_lr

        # evaluation metrics
        if iteration % eval_every == 0:
            eval_start = time.perf_counter()
            val_loss = evaluate(
                model = model,
                val_data = val_data,
                eval_batches = eval_batches,
                batch_size = batch_size,
                context_length = context_length,
                device = device
            )
            eval_sec = time.perf_counter() - eval_start
            elapsed_sec = time.perf_counter() - train_start_time

            metrics["val/loss"] = val_loss
            metrics["val/perplexity"] = perplexity(val_loss)
            metrics["time/eval_sec"] = eval_sec
            metrics["time/elapsed_sec"] = elapsed_sec
            metrics["train/tokens_processed"] = tokens_processed
            metrics["perf/tokens_per_sec"] = tokens_this_run / elapsed_sec
            metrics["optim/lr"] = curr_lr

        if logger and metrics:
            logger.log(metrics, iteration)

        # checkpoint
        if iteration % checkpoint_every == 0:
            save_checkpoint(
                model = model,
                optimizer = optimizer,
                iteration = iteration,
                metadata = metadata,
                out = checkpoint_path
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

def perplexity(loss: float) -> float:
    if math.isfinite(loss) and loss < 100:
        perplexity = math.exp(loss)
    else:
        perplexity = float("inf")
    return perplexity
