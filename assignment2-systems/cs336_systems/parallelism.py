import torch
import torch.distributed as dist
from typing import Any

# =============================================================================
# Data Parallelism
# =============================================================================

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

    # for test compatibility
    @property
    def module(self):
        return self.model

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


class FlatDDP(torch.nn.Module):
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

    # for test compatibility
    @property
    def module(self):
        return self.model

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def sync_gradients(self):
        with torch.no_grad():
            grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grads.append(param.grad)

            if not grads:
                return

            flat_grads = torch._utils._flatten_dense_tensors(grads)

            dist.all_reduce(
                tensor=flat_grads,
                op=dist.ReduceOp.SUM,
                group=self.process_group
            )
            flat_grads /= self.world_size

            synced_grads = torch._utils._unflatten_dense_tensors(
                flat_grads, grads
            )

            for (grad, synced_grad) in zip(grads, synced_grads):
                grad.copy_(synced_grad)


class OverlappedDDP(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        process_group: dist.ProcessGroup | None = None
    ):
        super().__init__()
        self.model = model
        self.world_size = dist.get_world_size(process_group)
        self.process_group = process_group

        self.hooks = []         # for gradient accumulation
        self.comm_handles = []  # for all-reduce

        for param in self.model.parameters():
            # add hook for overlapped communication
            if param.requires_grad:
                h = param.register_post_accumulate_grad_hook(
                    self._sync_grad
                )
                self.hooks.append(h)

            # synchronize initial values
            with torch.no_grad():
                dist.broadcast(
                    param,
                    src=0,
                    group=self.process_group
                )

    # for test compatibility
    @property
    def module(self):
        return self.model

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _sync_grad(self, param: torch.Tensor) -> None:
        with torch.no_grad():
            if param.grad is not None:
                param.grad /= self.world_size
                h = dist.all_reduce(
                    tensor=param.grad,
                    op=dist.ReduceOp.SUM,
                    group=self.process_group,
                    async_op=True
                )
                self.comm_handles.append(h)

    def finish_gradient_synchronization(self):
        for h in self.comm_handles:
            h.wait()
        self.comm_handles.clear()

    def sync_gradients(self):
        self.finish_gradient_synchronization()


class DDPStrategy:
    is_distributed = True

    def __init__(self, wrapper_cls: type[torch.nn.Module]):
        self.wrapper_cls = wrapper_cls

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return self.wrapper_cls(model)

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
        if not isinstance(model, self.wrapper_cls):
            raise TypeError(f"Expected a {self.wrapper_cls.__name__} model")
        model.sync_gradients()

    def full_state_dict(
        self,
        model: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        if not isinstance(model, self.wrapper_cls):
            raise TypeError(f"Expected a {self.wrapper_cls.__name__} model")
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


STRATEGIES = {
    "naive-ddp": DDPStrategy(NaiveDDP),
    "flat-ddp": DDPStrategy(FlatDDP),
    "overlapped-ddp": DDPStrategy(OverlappedDDP),
    "single": SingleStrategy(),
}

# =============================================================================
# Optimizer Sharding
# =============================================================================

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs: Any
    ):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.n_params = 0

        self.global_params = []
        self.local_param_groups = []
        self.optimizer = None

        super().__init__(params, kwargs)
        self.optimizer = optimizer_cls(self.local_param_groups, **kwargs)

    def step(self, closure=None, **kwargs):
        loss = self.optimizer.step(closure, **kwargs)

        with torch.no_grad():
            for i, param in enumerate(self.global_params):
                owner = i % self.world_size
                dist.broadcast(param, src=owner)

        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        raw = param_group["params"]
        if isinstance(raw, torch.Tensor):
            params = [raw]
        else:
            params = list(raw)

        self.global_params.extend(params)

        local_params = [
            p for i, p in enumerate(params)
            if (self.n_params + i) % self.world_size == self.rank
        ]
        self.n_params = self.n_params + len(params)

        if local_params:
            local_group = {
                k: param_group[k] for k in param_group if k != "params"
            }
            local_group["params"] = local_params

            self.local_param_groups.append(local_group)
            if self.optimizer is not None:
                self.optimizer.add_param_group(local_group)

    def zero_grad(self, set_to_none=True):
        for param in self.global_params:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.detach_()
                    param.grad.zero_()
