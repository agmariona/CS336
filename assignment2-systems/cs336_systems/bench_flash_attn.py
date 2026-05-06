import torch
import argparse
from pathlib import Path
import json
from itertools import product
from triton.testing import do_bench

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flash_attention import FlashAttention2_Triton

DTYPES = {
    "float32":  torch.float32,
    "bfloat16": torch.bfloat16
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", nargs='+', type=int,
        default=[128, 512, 2048, 8192, 32768, 65536]
    )
    parser.add_argument("--d-model", nargs='+', type=int,
        default=[16, 32, 64, 128]
    )
    parser.add_argument("--dtype", nargs='+', type=str,
        choices=["float32", "bfloat16"],
        default=["float32", "bfloat16"]
    )
    parser.add_argument("--out-path", default="results/flash_bench.json")

    return parser.parse_args()


def make_record(
    algorithm: str,
    sequence_length: int,
    d_model: int,
    dtype: str,
    batch_size: int,
    times_ms=None,
    oom: bool = False
) -> dict:
    record = {
        "algorithm":        algorithm,
        "sequence_length":  sequence_length,
        "d_model":          d_model,
        "dtype":            dtype,
        "batch_size":       batch_size,
        "oom":              oom
    }

    if not oom:
        record.update({
            "fwd_mean_ms":  times_ms[0],
            "bwd_mean_ms":  times_ms[1] - times_ms[0],
            "all_mean_ms":  times_ms[1],
        })

    return record


def vanilla_benchmark(
    batch_size: int,
    sequence_length: int,
    d_model: int,
    dtype: torch.dtype,
) -> tuple[float, float]:

    Q = torch.randn(
        batch_size, sequence_length, d_model,
        device='cuda', dtype=dtype, requires_grad=True
    )
    K = torch.randn(
        batch_size, sequence_length, d_model,
        device='cuda', dtype=dtype, requires_grad=True
    )
    V = torch.randn(
        batch_size, sequence_length, d_model,
        device='cuda', dtype=dtype, requires_grad=True
    )
    mask = torch.tril(torch.ones(
            sequence_length, sequence_length,
            device='cuda', dtype=torch.bool
    ))

    def fwd():
        scaled_dot_product_attention(Q, K, V, mask)

    def fwd_bwd():
        Q.grad = K.grad = V.grad = None
        out = scaled_dot_product_attention(Q, K, V, mask)
        out.sum().backward()

    fwd_ms = do_bench(fwd)
    fwd_bwd_ms = do_bench(fwd_bwd)

    return fwd_ms, fwd_bwd_ms


def flash_benchmark(
    batch_size: int,
    sequence_length: int,
    d_model: int,
    dtype: torch.dtype,
) -> tuple[float, float]:

    Q = torch.randn(
        batch_size, sequence_length, d_model,
        device='cuda', dtype=dtype, requires_grad=True
    )
    K = torch.randn(
        batch_size, sequence_length, d_model,
        device='cuda', dtype=dtype, requires_grad=True
    )
    V = torch.randn(
        batch_size, sequence_length, d_model,
        device='cuda', dtype=dtype, requires_grad=True
    )

    def fwd():
        FlashAttention2_Triton.apply(Q, K, V, True)

    def fwd_bwd():
        Q.grad = K.grad = V.grad = None
        out = FlashAttention2_Triton.apply(Q, K, V, True)
        out.sum().backward()

    fwd_ms = do_bench(fwd)
    fwd_bwd_ms = do_bench(fwd_bwd)

    return fwd_ms, fwd_bwd_ms


BENCHMARKS = {
    "vanilla":  vanilla_benchmark,
    "flash":    flash_benchmark
}


def main():
    args = parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    param_grid = product(
        args.sequence_length,
        args.d_model,
        args.dtype
    )
    records = []

    for sequence_length, d_model, dtype in param_grid:
        for algorithm, benchmark_fn in BENCHMARKS.items():
            try:
                times_ms = benchmark_fn(
                    batch_size      = args.batch_size,
                    sequence_length = sequence_length,
                    d_model         = d_model,
                    dtype           = DTYPES[dtype]
                )
                record = make_record(
                    algorithm, sequence_length, d_model,
                    dtype, args.batch_size,
                    times_ms, oom=False
                )
                print(
                    f"\t{algorithm} / "
                    f"{sequence_length=} / {d_model=} / {dtype=} / "
                    f"all_mean_time={times_ms[1]}ms"
                )

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

                record = make_record(
                    algorithm, sequence_length, d_model,
                    dtype, args.batch_size,
                    oom=True
                )
                print(
                    f"\t{algorithm} / "
                    f"{sequence_length=} / {d_model=} / {dtype=} / OOM"
                )

            records.append(record)

    with out_path.open('w') as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    main()
