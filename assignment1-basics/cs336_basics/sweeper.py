import argparse
from pathlib import Path
import yaml
from copy import deepcopy
import subprocess

from .training_utils import load_cfg

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def fmt_name(param: int) -> str:
    return f'ts_bsz_{param}'

def main() -> None:
    args = parse_args()
    base_config = load_cfg(args.base_config)
    bsz_sweep = [32, 64, 128, 256, 512]
    target_tokens = 81920000
    context_length = base_config["training"]["context_length"]

    for bsz in bsz_sweep:
        run_name = fmt_name(bsz)
        check_path = f"checkpoints/sweeps/batch_size/{run_name}.pt"
        total_iterations = target_tokens // (bsz * context_length)

        config = deepcopy(base_config)

        config["training"]["checkpoint_path"]   = check_path
        config["training"]["checkpoint_every"]  = total_iterations
        config["training"]["batch_size"]        = bsz
        config["training"]["iterations"]        = total_iterations
        config["optimizer"]["lr_schedule"]["warmup_iters"] = \
                                    max(1, int(0.05*total_iterations))
        config["optimizer"]["lr_schedule"]["cosine_cycle_iters"] = \
                                    total_iterations
        config["wandb"]["name"]                 = run_name

        out_dir = Path(args.config_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_name}.yaml"

        with open(out_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, sort_keys=False)

        cmd = [
            "uv", "run",
            "python", "-m", "cs336_basics.run_trainer",
            "--config", str(out_path)
        ]

        if args.dry_run:
            print(f' '.join(cmd))
        else:
            subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
