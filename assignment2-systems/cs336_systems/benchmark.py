import torch
from pathlib import Path
import argparse
from timeit import default_timer as timer
from statistics import mean, stdev
import json
from contextlib import nullcontext

import cs336_basics.model
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from .profiling import *


MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12 },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32
    },
    "10B": {
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36
    }
}

DEFAULTS = {
    "vocab_size": 10000,
    "rope_theta": 10000,
    "lr": 1e-3,
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "weight_decay": 0.01
}

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-size",
        choices=["small", "medium", "large", "xl", "10B"],
        default="small"
    )

    parser.add_argument("--mode",
        choices=["fwd", "fwd-bwd", "full"],
        default="full"
    )

    parser.add_argument("--batch-size",     type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)

    parser.add_argument("--warmup-steps",   type=int, default=5)
    parser.add_argument("--timed-steps",    type=int, default=10)

    parser.add_argument("--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32"
    )
    parser.add_argument("--device", choices=["mps", "cuda"], default="cuda")

    parser.add_argument("--annotate-attention", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")

    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--memory-path",
        default="results/memory/mem_profile.pickle"
    )

    parser.add_argument("--checkpoint-block-size", type=int)

    parser.add_argument("--compiled", action="store_true")

    return parser.parse_args()


def benchmark(
    model: torch.nn.Module,
    base_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    warmup_steps: int,
    timed_steps: int,
    mode: str,
    profile_memory: bool,
    memory_path: str | None = None,
    checkpoint_block_size: int | None = None,
) -> tuple[float, float]:
    # generate inputs
    inputs = torch.randint(
        low     = 0,
        high    = base_model.vocab_size,
        size    = (batch_size, base_model.context_length),
        device  = base_model.device
    )

    device_type = torch.device(base_model.device).type
    if device_type == "cuda":
        sync = torch.cuda.synchronize
    elif device_type == "mps":
        sync = torch.mps.synchronize
    else:
        raise ValueError(f"Unsupported device: {device_type}")

    with nvtx_range("warmup"):
        if mode =='fwd':
            for _ in range(warmup_steps):
                with torch.no_grad():
                    logits = model(
                        inputs,
                        checkpoint_block_size=checkpoint_block_size
                    )
        else:
            for _ in range(warmup_steps):
                optimizer.zero_grad()
                logits = model(
                    inputs,
                    checkpoint_block_size=checkpoint_block_size
                )
                loss = cross_entropy(logits, inputs)
                loss.backward()
                optimizer.step()

    times = []
    sync()

    if profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    if mode == 'fwd':
        for _ in range(timed_steps):
            with nvtx_range("timed_step"):
                t = timer()
                with nvtx_range("forward"):
                    with torch.no_grad():
                        logits = model(
                            inputs,
                            checkpoint_block_size=checkpoint_block_size
                        )
                sync()
                times.append(timer() - t)

    elif mode == 'fwd-bwd':
        for _ in range(timed_steps):
            with nvtx_range("timed_step"):
                t = timer()
                with nvtx_range("zero_grad"):
                    optimizer.zero_grad()
                with nvtx_range("forward"):
                    logits = model(
                        inputs,
                        checkpoint_block_size=checkpoint_block_size
                    )
                with nvtx_range("loss"):
                    loss = cross_entropy(logits, inputs)
                with nvtx_range("backward"):
                    loss.backward()
                sync()
                times.append(timer() - t)

    elif mode == 'full':
        for _ in range(timed_steps):
            with nvtx_range("timed_step"):
                t = timer()
                with nvtx_range("zero_grad"):
                    optimizer.zero_grad()
                with nvtx_range("forward"):
                    logits = model(
                        inputs,
                        checkpoint_block_size=checkpoint_block_size
                    )
                with nvtx_range("loss"):
                    loss = cross_entropy(logits, inputs)
                with nvtx_range("backward"):
                    loss.backward()
                with nvtx_range("optimizer_step"):
                    optimizer.step()
                sync()
                times.append(timer() - t)
    else:
        raise ValueError(f"Unsupported benchmark mode: {mode}")

    if profile_memory:
        mpath = Path(memory_path)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(memory_path)
        torch.cuda.memory._record_memory_history(enabled=None)

    if timed_steps > 1:
        t_stdev = stdev(times)
    else:
        t_stdev = None

    return mean(times), t_stdev


def main() -> None:
    args = parse_args()

    if args.annotate_attention:
        cs336_basics.model.scaled_dot_product_attention = \
            annotated_scaled_dot_product_attention

    if args.mixed_precision:
        if args.device != 'cuda':
            raise ValueError("Mixed precision (bf16) only supported on CUDA")
        if args.dtype != 'float32':
            raise ValueError("Mixed precision (bf16) expects fp32 model params")
        precision_context = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        )
    else:
        precision_context = nullcontext()

    if args.profile_memory:
        if args.device != 'cuda':
            raise ValueError("Memory profiling only supported on CUDA")

    base_model = cs336_basics.model.TransformerLM(
        vocab_size      = DEFAULTS["vocab_size"],
        context_length  = args.context_length,
        num_layers      = MODEL_CONFIGS[args.model_size]["num_layers"],
        d_model         = MODEL_CONFIGS[args.model_size]["d_model"],
        num_heads       = MODEL_CONFIGS[args.model_size]["num_heads"],
        d_ff            = MODEL_CONFIGS[args.model_size]["d_ff"],
        rope_theta      = DEFAULTS["rope_theta"],
        device          = args.device,
        dtype           = DTYPES[args.dtype]
    )

    if args.compiled:
        model = torch.compile(base_model)
    else:
        model = base_model

    optimizer = AdamW(
        params          = base_model.parameters(),
        lr              = DEFAULTS["lr"],
        betas           = DEFAULTS["betas"],
        eps             = DEFAULTS["eps"],
        weight_decay    = DEFAULTS["weight_decay"],
    )

    with precision_context:
        time_mean, time_stdev = benchmark(
            model                   = model,
            base_model              = base_model,
            optimizer               = optimizer,
            batch_size              = args.batch_size,
            warmup_steps            = args.warmup_steps,
            timed_steps             = args.timed_steps,
            mode                    = args.mode,
            profile_memory          = args.profile_memory,
            memory_path             = args.memory_path,
            checkpoint_block_size   = args.checkpoint_block_size
        )

    record = {
        "time_mean":        time_mean,
        "time_stdev":       time_stdev,
        **vars(args)
    }

    print(json.dumps(record, indent=2))

if __name__ == "__main__":
    main()
