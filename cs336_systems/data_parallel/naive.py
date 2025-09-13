import os
import copy
import torch
from jaxtyping import Float
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.adam import Adam
import dataclasses


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 1))

    def forward(self, x: Float[torch.Tensor, "... d"]) -> Float[torch.Tensor, "... one"]:
        return self.ff(x)


def assert_params_close(model_a: Model, model_b: Model):
    for (non_parallel_param_name, non_parallel_model_parameter), (
        ddp_model_param_name,
        ddp_model_parameter,
    ) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter), non_parallel_param_name


@dataclasses.dataclass
class TrainingConfig:
    n_steps: int
    global_batch_size: int


def setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def naive_ddp(rank: int, world_size: int, config: TrainingConfig) -> None:
    setup(rank=rank, world_size=world_size)
    dist.barrier()
    torch.manual_seed(rank)

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    non_parallel_model = Model().to(device)
    model = copy.deepcopy(non_parallel_model)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    if rank == 0:
        assert_params_close(model, non_parallel_model)

    # Train parallel model.
    optimizer = Adam(params=model.parameters(), lr=1e-4)
    for _ in range(config.n_steps):
        optimizer.zero_grad()
        data = torch.ones((config.global_batch_size // world_size, 10)) * rank
        logit = model(data)
        loss = logit.sum()
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

        optimizer.step()

    # Train local model
    if rank == 0:
        optimizer = Adam(params=non_parallel_model.parameters(), lr=1e-4)
        for _ in range(config.n_steps):
            optimizer.zero_grad()
            data = torch.cat(
                [torch.ones((config.global_batch_size // world_size, 10)) * i for i in range(world_size)],
                dim=0,
            )
            logit = non_parallel_model(data)
            loss = logit.sum()
            loss.backward()
            optimizer.step()

        assert_params_close(model, non_parallel_model)


if __name__ == "__main__":
    world_size = 4
    config = TrainingConfig(
        n_steps=10,
        global_batch_size=32,
    )
    mp.spawn(fn=naive_ddp, args=(world_size, config), nprocs=world_size, join=True)
