import dataclasses
import datetime
import enum
import timeit
import pandas as pd
from torch.utils import tensorboard
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import numpy.typing as npt
import argparse
import logging
import torch._dynamo

torch._dynamo.config.suppress_errors = True

from cs336_basics.training import adamw
from cs336_basics.training import checkpoints
from cs336_basics.training import cross_entropy_loss
from cs336_basics.training import data
from cs336_basics.training import schedule
from cs336_basics.transformer import transformer
from cs336_basics.transformer import utils

from cs336_systems import annotated_attention


class ModelConfigId(enum.StrEnum):
    SMALL = enum.auto()
    MEDIUM = enum.auto()
    LARGE = enum.auto()
    XL = enum.auto()
    XXL = enum.auto()  # 2.7B


@dataclasses.dataclass
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_CONFIGS = {
    ModelConfigId.SMALL: ModelConfig(
        d_model=768,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
    ),
    ModelConfigId.MEDIUM: ModelConfig(
        d_model=1024,
        d_ff=4096,
        num_layers=24,
        num_heads=16,
    ),
    ModelConfigId.LARGE: ModelConfig(
        d_model=1280,
        d_ff=5120,
        num_layers=36,
        num_heads=20,
    ),
    ModelConfigId.XL: ModelConfig(
        d_model=1600,
        d_ff=6400,
        num_layers=48,
        num_heads=25,
    ),
    ModelConfigId.XXL: ModelConfig(
        d_model=2560,
        d_ff=10240,
        num_layers=32,
        num_heads=32,
    ),
}


DEFAULT_MODEL_CONFIGS = [
    ModelConfigId.SMALL,
    ModelConfigId.MEDIUM,
    ModelConfigId.LARGE,
    ModelConfigId.XL,
    ModelConfigId.XXL,
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer model.")
    parser.add_argument("--configs", nargs="+", type=ModelConfigId, default=DEFAULT_MODEL_CONFIGS)
    parser.add_argument("--sequence_length", type=int, default=256, help="Length of input sequences (context length).")
    parser.add_argument("--vocab_size", type=int, default=10_000, help="Size of the vocabulary.")
    parser.add_argument(
        "--rope_theta", type=float, default=10000.0, help="Theta parameter for RoPE (Rotary Positional Embeddings)."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Data type for model parameters (e.g., float32, bfloat16).",
    )
    parser.add_argument("--max_lr", type=float, default=1e-4, help="Maximum learning rate for the AdamW optimizer.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for the AdamW optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps with cosine annealing.")
    parser.add_argument(
        "--betas", type=float, nargs=2, default=(0.9, 0.95), help="Beta coefficients for AdamW (beta1, beta2)."
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay (L2 penalty) for AdamW.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm.")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (e.g., 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--training_dataset", type=Path, required=True, help="Path to the memory-mapped training dataset (.bin file)."
    )
    parser.add_argument(
        "--eval_dataset", type=Path, required=True, help="Path to the memory-mapped evaluation dataset (.bin file)."
    )

    # Benchmarking params
    parser.add_argument("--w", type=int, default=5, help="Number of warmup steps before measuring time.")
    parser.add_argument("--n", type=int, default=10, help="Number of steps to measure time for.")
    parser.add_argument("--notime_backward", action="store_true")
    parser.add_argument("--use_annotated_attention", action="store_true")
    parser.add_argument("--full_optimizer_step", action="store_true")
    parser.add_argument("--profile_memory", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")

    args = parser.parse_args()

    match args.dtype:
        case "float32":
            model_dtype = torch.float32
        case "bfloat16":
            model_dtype = torch.bfloat16
        case "float16":
            model_dtype = torch.float16
        case _:
            raise ValueError(f"Unsupported dtype: {args.dtype}")

    time_df = pd.DataFrame()

    if args.profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    with torch.autocast(device_type=args.device, dtype=model_dtype):
        for config_id in args.configs:
            try:
                config = MODEL_CONFIGS[config_id]
                model = transformer.get_transformer(
                    d_model=config.d_model,
                    heads=config.num_heads,
                    d_ff=config.d_ff,
                    vocab_size=args.vocab_size,
                    context_length=args.sequence_length,
                    num_layers=config.num_layers,
                    rope_theta=args.rope_theta,
                    device=torch.device(args.device),
                    dtype=model_dtype,
                )
                if args.torch_compile:
                    model = torch.compile(model)
                assert next(model.parameters()).is_cuda, "Not on cuda"
                optimizer = adamw.AdamW(
                    params=model.parameters(),
                    lr=args.max_lr,
                    betas=args.betas,
                    weight_decay=args.weight_decay,
                )
                train_dataset = np.load(args.training_dataset, mmap_mode="r")
                inputs, targets = data.get_batch(
                    dataset=train_dataset,
                    batch_size=args.batch_size,
                    context_length=args.sequence_length,
                    device=args.device,
                )

                for _ in range(args.w):
                    optimizer.zero_grad()
                    logits = model(inputs)
                    del logits
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                if args.use_annotated_attention:
                    # Monkey patch.
                    attention_fn = utils.scaled_dot_product_attention
                    utils.scaled_dot_product_attention = annotated_attention.scaled_dot_product_attention

                ts_forward = []
                ts_backward = []
                for _ in range(args.n):
                    optimizer.zero_grad()
                    t0_forward = timeit.default_timer()
                    logits = model(inputs)
                    t1_forward = timeit.default_timer()
                    ts_forward.append(t1_forward - t0_forward)
                    if not args.notime_backward:
                        loss = cross_entropy_loss.cross_entropy_loss(logits=logits, targets=targets)
                        t0_backward = timeit.default_timer()
                        loss.backward()
                        if args.full_optimizer_step:
                            optimizer.step()
                        t1_backward = timeit.default_timer()
                        ts_backward.append(t1_backward - t0_backward)
                    else:
                        # Explicitly drop graph if not using backward.
                        del logits
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                if args.use_annotated_attention:
                    # Monkey patch undo.
                    utils.scaled_dot_product_attention = attention_fn

                time_df[f"{config_id}_forward"] = ts_forward
                if not args.notime_backward:
                    time_df[f"{config_id}_backward"] = ts_backward

            except RuntimeError as e:
                # Probably CUDA OOM
                print(config_id, e)

    if args.profile_memory:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    print(time_df)
    print(time_df.mean().to_markdown())
    print(time_df.var().to_markdown())
