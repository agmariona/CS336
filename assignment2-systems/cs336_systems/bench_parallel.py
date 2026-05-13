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
from .parallelism import NaiveDDP, DDPStrategy, SingleStrategy, STRATEGIES
from .train_parallel import MODEL_CONFIGS, MODEL_CLS, TLM_VOCAB_SIZE, \
    TLM_CONTEXT_LENGTH
from .train_parallel import setup, cleanup, train_one_step, compute_loss, \
    generate_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--parallelism", required=True,
        choices=["naive-ddp", "flat-ddp", "overlapped-ddp"],
    )
    parser.add_argument("--model-class", default="ToyModel",
        choices=["ToyModel", "TransformerLM"]
    )
    parser.add_argument("--model-size", default="xl",
        choices=["tiny", "small", "medium", "large", "xl", "10B"]
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--local-batch-size", type=int, default=2)

    return parser.parse_args()

def run_benchmark(
    rank: int,
    world_size: int,
    strategy_name: str,
    model_class: str,
    model_config: dict,
    backend: str,
    local_batch_size: int = 2,
    n_warmup: int = 10,
    n_timed: int = 10
):
    device = setup(rank, world_size, backend)
    strategy = STRATEGIES[strategy_name]

    try:
        model = MODEL_CLS[model_class](**model_config)
        model.to(device)

        train_model = strategy.wrap_model(model)
        optimizer = AdamW(train_model.parameters())

        inputs, targets = generate_data(
            model_class, model_config, batch_size=local_batch_size
        )
        inputs = inputs.to(device)
        if targets is not None:
            targets = targets.to(device)

        for _ in range(n_warmup):
            timed_train_one_step(
                strategy, device,
                train_model, optimizer,
                inputs, targets
            )

        times = torch.empty((n_timed, 2), device=device)
        for i in range(n_timed):
            with nvtx_range(device, "timed_train_one_step"):
                time_entry = timed_train_one_step(
                    strategy, device,
                    train_model, optimizer,
                    inputs, targets
                )

            times[i,:] = time_entry

        gathered_times = [
            torch.empty((n_timed, 2), device=device)
            for _ in range(world_size)
        ]
        dist.all_gather(gathered_times, times)

        if rank == 0:
            gathered_times = torch.stack(gathered_times)
            per_step_max = torch.max(gathered_times, dim=0).values
            mean_times = torch.mean(per_step_max, dim=0)

            print(f"\tMean step time: {mean_times[0].item():.4f} s")
            print(f"\tMean comm time: {mean_times[1].item():.4f} s")

    finally:
        cleanup()


def timed_train_one_step(
    strategy,
    device: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor | None = None
) -> torch.Tensor:
    synchronize_if_cuda(device)
    total_start = time.perf_counter()

    # fwd + bwd
    optimizer.zero_grad()
    out = model(inputs)
    loss = compute_loss(out, targets)

    with nvtx_range(device, 'backward'):
        loss.backward()

    # communications
    comm_start = time.perf_counter()
    with nvtx_range(device, 'after_backward'):
        strategy.after_backward(model, optimizer)

    synchronize_if_cuda(device)
    comm_end = time.perf_counter()

    # optimizer
    with nvtx_range(device, 'optimizer'):
        optimizer.step()

    synchronize_if_cuda(device)
    total_end = time.perf_counter()

    times = [
        total_end - total_start,
        comm_end - comm_start
    ]

    return torch.tensor(times, device=device)


def nvtx_range(device: str, name: str):
    if device.startswith('cuda'):
        return torch.cuda.nvtx.range(name)
    return nullcontext()


def synchronize_if_cuda(device: str):
    if device.startswith('cuda'):
        torch.cuda.synchronize()


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

    if torch.cuda.is_available():
        backend = "nccl"
        assert args.world_size <= torch.cuda.device_count()
    else:
        backend = "gloo"

    mp.spawn(
        fn=run_benchmark,
        args=(
            args.world_size,
            args.parallelism,
            args.model_class,
            model_config,
            backend,
            args.local_batch_size
        ),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()
