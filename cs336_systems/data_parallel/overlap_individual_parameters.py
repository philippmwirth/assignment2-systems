import os
import copy
import torch
import torch.nn as nn
import torch.distributed as dist





class DDP(nn.Module):

    def __init__(
        self,
        module: nn.Module,
    ):
        super().__init__()
        self.module = module
        self.handles = []

        def _all_reduce_grad_hook(t: torch.Tensor):
            handle = dist.all_reduce(t.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(_all_reduce_grad_hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def synchronize_weights(self) -> None:
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def finish_gradient_synchronization(self) -> None:
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        # This fakes an AVG reduce op that is not available on Gloo backend.
        world_size = torch.distributed.get_world_size()
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad /= world_size
