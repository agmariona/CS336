import argparse
import json
from pathlib import Path
import subprocess
from itertools import product

MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12
    },
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
    "batch_size": 4,
    "rope_theta": 10000
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-sizes", nargs="+",
        choices=["small", "medium", "large", "xl", "10B"],
        default=["small", "medium", "large", "xl", "10B"]
    )
    parser.add_argument("--modes", nargs="+",
        choices=["fwd", "fwd-bwd", "full"],
        default=["fwd", "fwd-bwd", "full"]
    )
    parser.add_argument("--context-lengths", nargs="+", type=int,
        default=[512]
    )
    parser.add_argument("--device",
        choices=["mps", "cuda"],
        default="mps",
    )
    parser.add_argument("--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32"
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=10)

    parser.add_argument("--out-path", default="benchmarks.jsonl")

    return parser.parse_args()

def config_to_cli_args(config):
    args = []
    for key, value in config.items():
        flag = "--" + key.replace("_", "-")
        args.extend([flag, str(value)])
    return args

def main() -> None:
    args = parse_args()

    out_path = Path('./results/' + args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_cmd = ["uv", "run", "python", "-m", "cs336_systems.benchmark"]
    with out_path.open("a") as f:
        for model_size, context_length, mode in product(
            args.model_sizes,
            args.context_lengths,
            args.modes
        ):
            config = {
                **DEFAULTS,
                **MODEL_CONFIGS[model_size],
                "context_length": context_length,
                "mode": mode,
                "device": args.device,
                "dtype": args.dtype,
                "warmup_steps": args.warmup_steps,
                "timed_steps": args.timed_steps
            }

            cmd = base_cmd + config_to_cli_args(config)

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(e.stderr)
                raise

            benchmark_result = json.loads(result.stdout)

            record = {
                "model_size": model_size,
                **config,
                **benchmark_result,
            }

            f.write(json.dumps(record) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
