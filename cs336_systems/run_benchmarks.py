import dataclasses
import datetime
import timeit
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer model.")

    # Model Arguments
    parser.add_argument(
        "--d_model", type=int, default=512, help="Dimension of the model's embeddings and hidden states."
    )
    parser.add_argument("--heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--d_ff", type=int, default=1344, help="Dimension of the feed-forward network.")
    parser.add_argument("--vocab_size", type=int, default=10_000, help="Size of the vocabulary.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument(
        "--rope_theta", type=float, default=10000.0, help="Theta parameter for RoPE (Rotary Positional Embeddings)."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "bfloat16", "float16"],
        help="Data type for model parameters (e.g., float32, bfloat16).",
    )

    # Optimizer Arguments
    parser.add_argument("--max_lr", type=float, default=1e-4, help="Maximum learning rate for the AdamW optimizer.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for the AdamW optimizer.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps with cosine annealing.")
    parser.add_argument(
        "--betas", type=float, nargs=2, default=(0.9, 0.95), help="Beta coefficients for AdamW (beta1, beta2)."
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay (L2 penalty) for AdamW.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm.")

    # Training Configuration
    parser.add_argument("--train_steps", type=int, default=5000, help="Total number of training steps.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
    parser.add_argument("--sequence_length", type=int, default=256, help="Length of input sequences (context length).")
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to run the model on (e.g., 'cuda', 'cpu').",
    )

    # Dataset Paths
    parser.add_argument(
        "--training_dataset", type=Path, required=True, help="Path to the memory-mapped training dataset (.bin file)."
    )
    parser.add_argument(
        "--eval_dataset", type=Path, required=True, help="Path to the memory-mapped evaluation dataset (.bin file)."
    )

    # Benchmarking params
    parser.add_argument(
        "--w", type=int, default=5, help="Number of warmup steps before measuring time."
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of steps to measure time for."
    )
    parser.add_argument(
        "--notime_backward", action="store_true"
    )

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

    model = transformer.get_transformer(
        d_model=args.d_model,
        heads=args.heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.sequence_length,
        num_layers=args.num_layers,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
        dtype=model_dtype,
    )
    model = torch.compile(model, backend="aot_eager")
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
        loss = cross_entropy_loss.cross_entropy_loss(logits=logits, targets=targets)
        loss = loss.float()
        loss.backward()
    
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
            loss = loss.float()
            loss.backward()
            t1_backward = timeit.default_timer()
            ts_backward.append(t1_backward - t0_backward)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    print(ts_forward)
    print(np.mean(ts_forward))
    print(np.std(ts_forward))
    if not args.notime_backward:
        print(ts_backward)
        print(np.mean(ts_backward))
        print(np.std(ts_backward))