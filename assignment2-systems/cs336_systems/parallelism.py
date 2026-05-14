import torch
import torch.distributed as dist
from typing import Any
from dataclasses import dataclass
import math

from cs336_basics.model import Linear, Embedding

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
    else: local_targets = None


    return local_inputs, local_targets


STRATEGIES = {
    "naive-ddp": DDPStrategy(NaiveDDP),
    "flat-ddp": DDPStrategy(FlatDDP),
    "overlapped-ddp": DDPStrategy(OverlappedDDP),
    "single": SingleStrategy(),
}

# =============================================================================
# Fully-Sharded Data Parallel
# =============================================================================

@dataclass
class ShardedParamRecord:
    layer: torch.nn.Module
    param_name: str
    full_shape: torch.Size
    full_dtype: torch.dtype
    start_idx: int
    end_idx: int
    local_shard: torch.nn.Parameter
    shard_lens: list[int]


class FullyShardedDataParallel(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        compute_dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.module = module
        self.shard_records = []
        self.compute_dtype = compute_dtype
        self.hook_handles = []

        # shard Linear and Embedding layers
        for layer_name, layer in self.module.named_modules():
            if isinstance(layer, (Linear, Embedding)):
                weight = layer.weight
                full_shape = weight.shape
                full_dtype = weight.dtype
                weight = weight.flatten()

                numel = weight.numel()
                base = numel // self.world_size
                remainder = numel % self.world_size
                shard_lens = [
                    base + 1 if r < remainder else base
                    for r in range(self.world_size)
                ]

                start_idx = self.rank * base + min(self.rank, remainder)
                end_idx = start_idx + base
                if self.rank < remainder:
                    end_idx += 1
                weight_slice = weight[start_idx:end_idx].detach().clone()
                weight_shard = torch.nn.Parameter(weight_slice)

                layer.weight = weight_shard

                record = ShardedParamRecord(
                    layer=layer,
                    param_name="weight",
                    full_shape=full_shape,
                    full_dtype=full_dtype,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    local_shard=weight_shard,
                    shard_lens=shard_lens
                )
                self.shard_records.append(record)

        for record in self.shard_records:
            self.hook_handles.append(
                record.layer.register_forward_pre_hook(
                    self._make_forward_pre_hook(record)
            ))
            self.hook_handles.append(
                record.layer.register_forward_hook(
                    self._make_forward_hook(record)
            ))
            self.hook_handles.append(
                record.layer.register_full_backward_pre_hook(
                    self._make_full_backward_pre_hook(record)
            ))

    def _make_forward_pre_hook(self, record):
        def hook(module, inputs):
            full_weight = self._gather_full_weight(record)
            module.weight = torch.nn.Parameter(full_weight)
        return hook

    def _make_forward_hook(self, record):
        def hook(module, inputs, output):
            module.weight = record.local_shard
        return hook

    def _make_full_backward_pre_hook(self, record):
        def hook(module, grad_output):
            full_weight = self._gather_full_weight(record)
            module.weight = torch.nn.Parameter(full_weight)
            module.weight.register_post_accumulate_grad_hook(
                self._make_post_accumulate_grad_hook(record)
            )
        return hook

    def _make_post_accumulate_grad_hook(self, record):
        def hook(param):
            local_grad = self._reduce_scatter_grad(record, param)
            record.local_shard.grad = local_grad
            record.layer.weight = record.local_shard
        return hook

    def _gather_full_weight(self, record):
        if self.compute_dtype is not None:
            comm_dtype = self.compute_dtype
        else:
            comm_dtype = record.local_shard.dtype

        gathered_weights = [
            torch.empty(
                record.shard_lens[r],
                dtype=comm_dtype,
                device=record.local_shard.device
            )
            for r in range(self.world_size)
        ]

        dist.all_gather(
            gathered_weights,
            record.local_shard.to(dtype=comm_dtype)
        )

        full_flat = torch.cat(gathered_weights)
        full_weight = full_flat.view(record.full_shape)

        return full_weight

    def _reduce_scatter_grad(self, record, param):
        if self.compute_dtype is not None:
            comm_dtype = self.compute_dtype
        else:
            comm_dtype = record.local_shard.dtype

        assert param.grad is not None

        full_grad_flat = param.grad.flatten()
        assert full_grad_flat.dtype == comm_dtype

        offset = 0
        grad_chunks = []
        for length in record.shard_lens:
            grad_chunks.append(full_grad_flat[offset : offset+length])
            offset += length

        local_grad = torch.empty(
            record.local_shard.shape,
            device=record.local_shard.device,
            dtype=comm_dtype
        )

        dist.reduce_scatter(local_grad, grad_chunks)
        local_grad /= self.world_size
        param.grad = None

        return local_grad.to(dtype=record.local_shard.dtype)






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
