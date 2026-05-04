import torch
import argparse
from pathlib import Path
import json
from timeit import default_timer as timer
from statistics import mean, stdev
from itertools import product

from cs336_basics.model import scaled_dot_product_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", nargs='+', type=int,
        default=[256, 1024, 4096, 8192, 16384]
    )
    parser.add_argument("--d-model", nargs='+', type=int,
        default=[16, 32, 64, 128]
    )
    parser.add_argument("--out-path", default="results/attn_bench.json")

    parser.add_argument("--compiled", action="store_true")

    return parser.parse_args()


def benchmark(
    batch_size: int,
    context_length: int,
    d_model: int,
    n_warmup: int = 20,
    n_timed: int = 100,
    device: str | torch.device = 'cuda',
    compiled: bool = False,
) -> tuple[float, float, float, float, float]:
    Q = torch.randn(
        batch_size, context_length, d_model,
        device=device, requires_grad=True
    )
    K = torch.randn(
        batch_size, context_length, d_model,
        device=device, requires_grad=True
    )
    V = torch.randn(
        batch_size, context_length, d_model,
        device=device, requires_grad=True
    )

    mask = torch.tril(torch.ones(
            context_length, context_length,
            device=device, dtype=torch.bool
    ))

    if compiled:
        attn = torch.compile(scaled_dot_product_attention)
    else:
        attn = scaled_dot_product_attention

    # warm up
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        Q.grad = None
        K.grad = None
        V.grad = None

        out = attn(Q, K, V, mask)
        loss = out.sum()
        loss.backward()
    torch.cuda.synchronize()

    fwd_times = []
    bwd_times = []
    mem_alloc = []
    for _ in range(n_timed):
        Q.grad = None
        K.grad = None
        V.grad = None

        # forward pass
        torch.cuda.synchronize()
        t0 = timer()
        out = attn(Q, K, V, mask)
        torch.cuda.synchronize()

        fwd_times.append(timer() - t0)

        # memory before backward pass
        mem_alloc.append(torch.cuda.memory_allocated())

        loss = out.sum()

        # backward pass
        torch.cuda.synchronize()
        t0 = timer()
        loss.backward()
        torch.cuda.synchronize()
        bwd_times.append(timer() - t0)

    return (
        mean(fwd_times), stdev(fwd_times),
        mean(bwd_times), stdev(bwd_times),
        max(mem_alloc)
    )


def main() -> None:
    args = parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for context_length, d_model in product(args.context_length, args.d_model):
        try:
            fwd_mean, fwd_stdev, bwd_mean, bwd_stdev, mem_alloc = benchmark(
                batch_size      = args.batch_size,
                context_length  = context_length,
                d_model         = d_model,
                compiled        = args.compiled
            )

            record = {
                "batch_size":       args.batch_size,
                "context_length":   context_length,
                "d_model":          d_model,
                "oom":              False,
                "fwd_mean_sec":     fwd_mean,
                "fwd_stdev_sec":    fwd_stdev,
                "bwd_mean_sec":     bwd_mean,
                "bwd_stdev_sec":    bwd_stdev,
                "mem_alloc_mib":    mem_alloc / 1024**2,
                "compiled":         args.compiled
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

            record = {
                "batch_size":       args.batch_size,
                "context_length":   context_length,
                "d_model":          d_model,
                "oom":              True,
                "compiled":         args.compiled
            }

        records.append(record)
        print(f'\t{context_length=} / {d_model=}')

    with out_path.open('w') as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    main()
