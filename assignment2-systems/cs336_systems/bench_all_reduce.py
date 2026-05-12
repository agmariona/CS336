import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
from itertools import product
import time
import json
from statistics import mean

from .train_parallel import setup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-mb", type=int, nargs='+',
        default=[1, 10, 100, 1000])
    parser.add_argument("--world-sizes", type=int, nargs='+',
        default=[2, 4, 6])

    parser.add_argument("--backend", choices=["gloo", "nccl"],
        default="nccl")
    parser.add_argument("--out-path", default="results/all_reduce.json")

    return parser.parse_args()


def make_record(
    data_mb: int,
    world_size: int,
    backend: str,
    time_s: float
) -> dict:
    record = {
        "data_mb":      data_mb,
        "world_size":   world_size,
        "backend":      backend,
        "time_s":       time_s
    }

    return record


def benchmark_worker(
    rank: int,
    world_size: int,
    backend: str,
    data_mb: int,
    result_queue,
    n_warmup: int=5,
    n_trials: int=5
):
    device = setup(rank, world_size, backend)

    n_elements = data_mb * 1024**2 // 4     # fp32
    data = torch.randn(n_elements, device=device)

    # warmup
    for _ in range(n_warmup):
        dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    synchronize(device)
    dist.barrier()

    duration = 0
    for _ in range(n_trials):
        t_start = time.perf_counter()

        dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
        synchronize(device)

        t_end = time.perf_counter()
        dist.barrier()

        duration += t_end - t_start
    duration /= n_trials

    all_durations = [None for _ in range(world_size)]
    dist.all_gather_object(all_durations, duration)

    if rank == 0:
        record = make_record(
            data_mb,
            world_size,
            backend,
            mean(all_durations)
        )
        result_queue.put(record)

    dist.destroy_process_group()


def get_device(rank, backend):
    if backend == "nccl":
        return f"cuda:{rank}"
    elif backend == "gloo":
        return "cpu"
    else:
        raise ValueError(f"Unsupported backend {backend}")


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def validate_cuda(world_sizes):
    if not torch.cuda.is_available():
        raise ValueError("NCCL requires a GPU")

    n_gpu = torch.cuda.device_count()
    if max(world_sizes) > n_gpu:
        raise ValueError(f"Only {n_gpu} GPU(s) available")


def main():
    args = parse_args()

    if args.backend == "nccl":
        validate_cuda(args.world_sizes)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    param_grid = product(
        args.data_mb,
        args.world_sizes
    )
    records = []
    ctx = mp.get_context("spawn")

    for data_mb, world_size in param_grid:
        print(f"\t{data_mb=}\t{world_size=}")

        result_queue = ctx.Queue()

        mp.spawn(
            fn=benchmark_worker,
            args=(world_size, args.backend, data_mb, result_queue),
            nprocs=world_size,
            join=True
        )

        records.append(result_queue.get())

    with out_path.open('w') as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    main()
