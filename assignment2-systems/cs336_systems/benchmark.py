import torch
import torch.cuda.nvtx as nvtx

import argparse
from timeit import default_timer as timer
from statistics import mean, stdev
import json

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--rope-theta", type=int, default=10000)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)

    parser.add_argument("--device", type=str, default="mps",
        choices=["mps", "cuda"]
    )
    parser.add_argument("--dtype", type=str, default="float32",
        choices=["float32", "float16", "bfloat16"]
    )

    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--mode", type=str, default='full',
        choices=["fwd", "fwd-bwd", "full"]
    )
    return parser.parse_args()

def benchmark(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    warmup_steps: int,
    timed_steps: int,
    mode: str
) -> tuple[float, float]:
    inputs = torch.randint(
        low = 0,
        high = model.vocab_size,
        size = (batch_size, model.context_length),
        device = model.device
    )

    device_type = torch.device(model.device).type
    if device_type == "cuda":
        sync = torch.cuda.synchronize
    elif device_type == "mps":
        sync = torch.mps.synchronize
    else:
        raise ValueError(f"Unsupported device: {device_type}")

    with nvtx.range("warmup"):
        for _ in range(warmup_steps):
            optimizer.zero_grad()
            logits = model(inputs)
            loss = cross_entropy(logits, inputs)
            loss.backward()
            optimizer.step()

    times = []
    sync()
    if mode == 'fwd':
        for _ in range(timed_steps):
            t = timer()
            logits = model(inputs)
            sync()
            times.append(timer() - t)
    elif mode == 'fwd-bwd':
        for _ in range(timed_steps):
            t = timer()
            optimizer.zero_grad()
            logits = model(inputs)
            loss = cross_entropy(logits, inputs)
            loss.backward()
            sync()
            times.append(timer() - t)
    elif mode == 'full':
        for _ in range(timed_steps):
            with nvtx.range("timed_step"):
                t = timer()
                with nvtx.range("zero_grad"):
                    optimizer.zero_grad()
                with nvtx.range("forward"):
                    logits = model(inputs)
                with nvtx.range("loss"):
                    loss = cross_entropy(logits, inputs)
                with nvtx.range("backward"):
                    loss.backward()
                with nvtx.range("optimizer_step"):
                    optimizer.step()
                sync()
                times.append(timer() - t)
    else:
        raise ValueError(f"Unsupported benchmark mode: {mode}")

    return mean(times), stdev(times)


def main() -> None:
    args = parse_args()

    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }

    model = TransformerLM(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        num_layers = args.num_layers,
        d_model = args.d_model,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = args.rope_theta,
        device = args.device,
        dtype = dtypes[args.dtype]
    )

    optimizer = AdamW(
        params = model.parameters(),
        lr = 1e-3,
        betas = [0.9, 0.95],
        eps = 1e-8,
        weight_decay = 0.01
    )

    time_mean, time_stdev = benchmark(
        model = model,
        optimizer = optimizer,
        batch_size = args.batch_size,
        warmup_steps = args.warmup_steps,
        timed_steps = args.timed_steps,
        mode = args.mode
    )

    print(json.dumps({
        "mean_s": time_mean,
        "std_s": time_stdev
    }))


if __name__ == "__main__":
    main()
