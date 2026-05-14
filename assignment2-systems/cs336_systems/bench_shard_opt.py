import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from contextlib import nullcontext

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

from .toy_model import ToyModel
from .parallelism import ShardedOptimizer
from .train_parallel import MODEL_CONFIGS, MODEL_CLS, TLM_VOCAB_SIZE, \
    TLM_CONTEXT_LENGTH
from .train_parallel import setup, cleanup, train_one_step, compute_loss, \
    generate_data
from .bench_parallel import nvtx_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--shard-optimizer", action="store_true")
    parser.add_argument("--model-class", default="ToyModel",
        choices=["ToyModel", "TransformerLM"]
    )
    parser.add_argument("--model-size", default="xl",
        choices=["tiny", "small", "medium", "large", "xl", "10B"]
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)

    return parser.parse_args()


def run_benchmark(
    rank: int,
    world_size: int,
    shard_optimizer: bool,
    model_class: str,
    model_config: dict,
    backend: str,
    batch_size: int = 2,
):
    device = setup(rank, world_size, backend)
    torch.cuda.reset_peak_memory_stats(device)

    try:
        model = MODEL_CLS[model_class](**model_config)
        model.to(device)

        if shard_optimizer:
            optimizer = ShardedOptimizer(model.parameters(), AdamW)
        else:
            optimizer = AdamW(model.parameters())

        # initial memory
        torch.cuda.synchronize()
        init_mem = torch.cuda.max_memory_allocated(device)

        inputs, targets = generate_data(
            model_class, model_config, batch_size=batch_size
        )
        inputs = inputs.to(device)
        if targets is not None:
            targets = targets.to(device)

        optimizer.zero_grad()
        out = model(inputs)
        loss = compute_loss(out, targets)
        loss.backward()

        # peak memory before optimizer
        torch.cuda.synchronize()
        pre_opt_mem = torch.cuda.max_memory_allocated(device)
        optimizer.step()

        # peak memory after optimizer
        torch.cuda.synchronize()
        post_opt_mem = torch.cuda.max_memory_allocated(device)

        print(f"\t{device}:")
        print(f"\t\tPeak init mem: {init_mem}")
        print(f"\t\tPre-step mem: {pre_opt_mem}")
        print(f"\t\tPost-step mem: {post_opt_mem}")

    finally:
        cleanup()


def main():
    args = parse_args()

    if args.model_class == "ToyModel":
        model_config = {}
    elif args.model_class == "TransformerLM":
        model_config = {
            "vocab_size": TLM_VOCAB_SIZE,
            "context_length": TLM_CONTEXT_LENGTH,
            **MODEL_CONFIGS[args.model_size]
        }

    assert torch.cuda.is_available()
    assert args.world_size <= torch.cuda.device_count()

    mp.spawn(
        fn=run_benchmark,
        args=(
            args.world_size,
            args.shard_optimizer,
            args.model_class,
            model_config,
            "nccl",
            args.batch_size
        ),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()
