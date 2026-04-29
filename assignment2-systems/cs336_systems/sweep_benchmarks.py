import argparse
import json
from pathlib import Path
import subprocess
from itertools import product

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-sizes", nargs="+",
        choices=["small", "medium", "large", "xl", "10B"],
        default=["small", "medium", "large"]
    )

    parser.add_argument("--modes", nargs="+",
        choices=["fwd", "fwd-bwd", "full"],
        default=["fwd", "fwd-bwd", "full"]
    )

    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4])
    parser.add_argument("--context-lengths", nargs="+", type=int,
        default=[512]
    )

    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=1)

    parser.add_argument("--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32"
    )
    parser.add_argument("--device", choices=["mps", "cuda"], default="cuda")

    parser.add_argument("--out-path", default="results/benchmarks.json")

    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", default="results/nsys")
    parser.add_argument("--profile-full-trace", action="store_true")

    parser.add_argument("--mixed-precision", action="store_true")

    return parser.parse_args()

BOOL_FLAGS = {
    "mixed_precision"
}

def config_to_args(config):
    args = []
    for key, value in config.items():
        flag = "--" + key.replace("_", "-")

        if key in BOOL_FLAGS:
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])
    return args

def config_to_name(config):
    parts = [
        config["model_size"],
        config["mode"].replace("-", "_"),
        f"bs{config['batch_size']}",
        f"ctx{config['context_length']}",
        # f"warm{config['warmup_steps']}",
        # f"time{config['timed_steps']}",
        config["dtype"],
        "mp" if config["mixed_precision"] else "no_mp",
        config["device"]
    ]
    return "_".join(parts)

def main() -> None:
    args = parse_args()

    if args.profile:
        profile_dir = Path(args.profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output = []

    for model_size, mode, batch_size, context_length in product(
        args.model_sizes,
        args.modes,
        args.batch_sizes,
        args.context_lengths,
    ):
        config = {
            "model_size":       model_size,
            "mode":             mode,
            "batch_size":       batch_size,
            "context_length":   context_length,
            "warmup_steps":     args.warmup_steps,
            "timed_steps":      args.timed_steps,
            "dtype":            args.dtype,
            "device":           args.device,
            "mixed_precision":  args.mixed_precision
        }

        if args.profile:
            profile_path = profile_dir / config_to_name(config)
            cmd = [
                "uv", "run", "nsys", "profile",
                "--trace=cuda,cudnn,cublas,osrt,nvtx",
                "--pytorch=functions-trace,autograd-shapes-nvtx"
            ]
            if args.profile_full_trace:
                cmd += [
                    "--cudabacktrace=all",
                    "--python-backtrace=cuda"
                ]
            cmd += [
                "--output", str(profile_path),
                "--",
                "python", "-m", "cs336_systems.benchmark",
                "--annotate-attention"
            ]
        else:
            cmd = [
                "uv", "run", "python", "-m", "cs336_systems.benchmark"
            ]
        cmd = cmd + config_to_args(config)

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

        if not args.profile:
            benchmark_result = json.loads(result.stdout)
            output.append(benchmark_result)

        print(f"\t{model_size} / {mode} / {context_length}")

    if not args.profile:
        with out_path.open("w") as f:
            json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
