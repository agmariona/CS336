import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

from .toy_model import ToyModel
from .parallelism import NaiveDDP, ParallelStrategy, NaiveDDPStrategy, \
    SingleStrategy


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

MODEL_CLS = {
    'ToyModel': ToyModel,
    'TransformerLM': TransformerLM
}

TLM_VOCAB_SIZE = 10000
TLM_CONTEXT_LENGTH = 512

STRATEGIES: dict[str, ParallelStrategy] = {
    "naive-ddp": NaiveDDPStrategy()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--parallelism", required=True,
        choices=["naive-ddp"],
    )
    parser.add_argument("--model-class", default="ToyModel",
        choices=["ToyModel", "TransformerLM"]
    )
    parser.add_argument("--model-config", default="xl",
        choices=["small", "medium", "large", "xl", "10B"]
    )
    parser.add_argument("--world-size", type=int, default=2)

    return parser.parse_args()


def setup(rank: int, world_size: int, backend: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    if backend == "nccl":
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    elif backend == "gloo":
        device = "cpu"
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    else:
        raise ValueError(f"Unsupported backend {backend}")

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def cleanup():
    dist.destroy_process_group()


def run_parallel_worker(
    rank: int,
    world_size: int,
    strategy_name: str,
    model_class: type[torch.nn.Module],
    model_config: dict,
    initial_state,
    inputs: torch.Tensor,
    targets: torch.Tensor | None = None,
    backend: str="nccl"
):
    device = setup(rank, world_size, backend)
    strategy = STRATEGIES[strategy_name]

    try:
        local_inputs, local_targets = strategy.shard_data(
            device, rank, world_size,
            inputs, targets
        )

        model = model_class(**model_config)
        model.load_state_dict(initial_state)
        model.to(device)

        train_model = strategy.wrap_model(model)
        optimizer = AdamW(train_model.parameters())

        train_one_step(
            strategy, train_model, optimizer,
            local_inputs, local_targets,
        )

        # baseline testing
        if rank == 0:
            baseline_model = model_class(**model_config)
            baseline_model.load_state_dict(initial_state)
            baseline_model.to(device)
            baseline_optimizer = AdamW(baseline_model.parameters())
            baseline_inputs, baseline_targets = SingleStrategy().shard_data(
                device, rank, world_size,
                inputs, targets
            )

            train_one_step(
                SingleStrategy(),
                baseline_model,
                baseline_optimizer,
                baseline_inputs,
                baseline_targets,
            )

            baseline_state = SingleStrategy().full_state_dict(baseline_model)
            parallel_state = strategy.full_state_dict(train_model)

            assert states_allclose(baseline_state, parallel_state)
    finally:
        cleanup()


def train_one_step(
    strategy,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor | None = None
):
    optimizer.zero_grad()
    out = model(inputs)
    loss = compute_loss(out, targets)
    loss.backward()
    strategy.after_backward(model, optimizer)
    optimizer.step()


def states_allclose(a: dict, b: dict) -> bool:
    a_keys = set(a.keys())
    b_keys = set(b.keys())

    if a_keys != b_keys:
        return False

    allclose = True
    for key in sorted(a_keys):
        if not torch.allclose(a[key], b[key]):
            allclose = False
            break

    return allclose


def compute_loss(
    out: torch.Tensor,
    targets: torch.Tensor | None
) -> torch.Tensor:
    if targets is not None:
        return cross_entropy(out, targets)
    else:
        return out.square().mean()


def generate_data(
    model_class: str,
    model_config: dict | None = None,
    batch_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if model_class == "ToyModel":
        context_length = 10
        inputs = torch.randn(batch_size, context_length)
        targets = None

    elif model_class == "TransformerLM":
        assert model_config is not None
        inputs = torch.randint(
            low=0,
            high=model_config["vocab_size"],
            size=(batch_size, model_config["context_length"])
        )
        targets = torch.randint(
            low=0,
            high=model_config["vocab_size"],
            size=(batch_size, model_config["context_length"])
        )

    return inputs, targets


def driver(
    strategy_name: str,
    model_class: type[torch.nn.Module],
    model_config: dict,
    initial_state: dict,
    inputs: torch.Tensor,
    targets: torch.Tensor | None,
    world_size: int
):
    if torch.cuda.is_available():
        backend = "nccl"
        assert world_size <= torch.cuda.device_count()
    else:
        backend = "gloo"

    mp.spawn(
        fn=run_parallel_worker,
        args=(
            world_size,
            strategy_name,
            model_class,
            model_config,
            initial_state,
            inputs,
            targets,
            backend
        ),
        nprocs=world_size,
        join=True
    )


def main():
    args = parse_args()

    if args.model_class == "ToyModel":
        model_config = {}
    elif args.model_class == "TransformerLM":
        model_config = {
            "vocab_size": TLM_VOCAB_SIZE,
            "context_length": TLM_CONTEXT_LENGTH,
            **MODEL_CONFIGS[args.model_config]
        }
    model_class = MODEL_CLS[args.model_class]
    initial_model = model_class(**model_config)
    initial_state = {
        name: tensor.detach().clone()
        for name, tensor in initial_model.state_dict().items()
    }

    inputs, targets = generate_data(args.model_class, model_config)

    driver(
        args.parallelism,
        model_class,
        model_config,
        initial_state,
        inputs,
        targets,
        args.world_size
    )


if __name__ == "__main__":
    main()
