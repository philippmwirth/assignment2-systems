import torch
import torch.nn as nn
import torch.distributed as dist
import functools


class DDP(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        bucket_size_mb: float,
    ):
        super().__init__()
        self.module = module
        self.handles = []
        self.buckets = [[]]
        self.param_name_to_bucket = {}
        self.bucket_size_mb = bucket_size_mb

        current_bucket_size_mb = 0
        for param_name, param in self.module.named_parameters():
            if param.requires_grad:
                tensor_size_mb = (param.data.nelement() * param.data.element_size()) * 1e-6
                if current_bucket_size_mb + tensor_size_mb > self.bucket_size_mb:
                    self.buckets.append([])
                    current_bucket_size_mb = 0.0
                current_bucket_size_mb += tensor_size_mb
                self.buckets[-1].append(param_name)
                self.param_name_to_bucket[param_name] = len(self.buckets) - 1

        self.grads = [[] for _ in self.buckets]
        self.bucket_to_reduce = len(self.buckets) - 1

        def _all_reduce_grad_hook(name: str, t: torch.Tensor):
            bucket = self.param_name_to_bucket[name]
            self.grads[bucket].append(t.grad)
            while self.bucket_to_reduce >= 0 and len(self.grads[self.bucket_to_reduce]) == len(
                self.buckets[self.bucket_to_reduce]
            ):
                if grads := self.grads[self.bucket_to_reduce]:
                    flat = torch._utils._flatten_dense_tensors(grads)
                    handle = dist.all_reduce(
                        flat,
                        op=dist.ReduceOp.SUM,
                        async_op=True,
                    )
                    self.handles.append((handle, flat, grads))
                    self.grads[self.bucket_to_reduce] = []
                self.bucket_to_reduce = self.bucket_to_reduce - 1

        for param_name, param in self.module.named_parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(functools.partial(_all_reduce_grad_hook, param_name))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def synchronize_weights(self) -> None:
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def finish_gradient_synchronization(self) -> None:
        for handle, flat, grads in self.handles:
            handle.wait()
            unflat = torch._utils._unflatten_dense_tensors(flat, grads)
            for g, reduced in zip(grads, unflat):
                g.copy_(reduced)

        self.handles.clear()
        self.bucket_to_reduce = len(self.buckets) - 1
        # This fakes an AVG reduce op that is not available on Gloo backend.
        world_size = torch.distributed.get_world_size()
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad /= world_size
