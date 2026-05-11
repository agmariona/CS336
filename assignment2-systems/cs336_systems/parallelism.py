import torch
import torch.distributed as dist
from typing import Protocol

class NaiveDDP(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        process_group: dist.ProcessGroup | None = None
    ):
        super().__init__()
        self.model = model
        self.world_size = dist.get_world_size(process_group)
        self.process_group = process_group

        with torch.no_grad():
            for param in self.model.parameters():
                dist.broadcast(
                    param,
                    src=0,
                    group=self.process_group
                )


    def forward(self, x: torch.Tensor):
        return self.model(x)


    def sync_gradients(self):
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(
                        tensor=param.grad,
                        op=dist.ReduceOp.SUM,
                        group=self.process_group
                    )
                    param.grad /= self.world_size


class ParallelStrategy(Protocol):
    is_distributed: bool

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        ...

    def shard_data(
        self,
        device: str,
        rank: int,
        world_size: int,
        inputs: torch.Tensor,
        targets: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ...

    def after_backward(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ):
        ...

    def full_state_dict(
        self,
        model: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        ...


class NaiveDDPStrategy:
    is_distributed = True

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return NaiveDDP(model)

    def shard_data(
        self,
        device: str,
        rank: int,
        world_size: int,
        inputs: torch.Tensor,
        targets: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return shard_batch(device, rank, world_size, inputs, targets)

    def after_backward(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ):
        if not isinstance(model, NaiveDDP):
            raise TypeError("Expected a NaiveDDP model")
        model.sync_gradients()

    def full_state_dict(
        self,
        model: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        if not isinstance(model, NaiveDDP):
            raise TypeError("Expected a NaiveDDP model")
        return detach(model.model.state_dict())

class SingleStrategy:
    is_distributed = False

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def shard_data(
        self,
        device: str,
        rank: int,
        world_size: int,
        inputs: torch.Tensor,
        targets: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        local_inputs = inputs.to(device)
        if targets is not None:
            local_targets = targets.to(device)
        else:
            local_targets = None
        return local_inputs, local_targets

    def after_backward(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ):
        return

    def full_state_dict(
        self,
        model: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        return detach(model.state_dict())


def detach(state: dict) -> dict:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
    }


def shard_batch(
    device,
    rank: int,
    world_size: int,
    inputs: torch.Tensor,
    targets: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch_size = inputs.size(0)
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size

    start_idx = rank * local_batch_size
    end_idx = start_idx + local_batch_size

    local_inputs = inputs[start_idx : end_idx].to(device)
    if targets is not None:
        local_targets = targets[start_idx : end_idx].to(device)
    else:
        local_targets = None

    return local_inputs, local_targets
