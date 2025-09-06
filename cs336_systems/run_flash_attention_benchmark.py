import torch
import torch.nn as nn
from cs336_systems.flash_attention import triton_impl
import einops
from jaxtyping import Float
import math
import pandas as pd
import argparse
import triton
import pprint


class SimpleFlashAttention(nn.Module):
    """A simple attention implementation to faciliate benchmarking."""

    def forward(self, q, k, v, is_causal):
        return triton_impl.FlashAttention.apply(q, k, v, is_causal)


def softmax(x: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    y = (x - x.max(dim=dim, keepdim=True).values).exp()
    return y / y.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: Float[torch.Tensor, "b s d"],
    k: Float[torch.Tensor, "b s d"],
    v: Float[torch.Tensor, "b s d"],
    is_causal: bool,
) -> Float[torch.Tensor, "b s d"]:
    d_k = k.shape[-1]
    s = k.shape[-2]
    qk_scaled = einops.einsum(q, k, "... seq1 d, ... seq2 d -> ... seq1 seq2") / math.sqrt(d_k)
    if is_causal:
        mask = torch.triu(torch.ones((s, s), dtype=torch.long), diagonal=1).bool()
        qk_scaled[..., mask] -= 1e6
    weights = softmax(x=qk_scaled, dim=-1)
    return einops.einsum(weights, v, "... seq1 seq2, ... seq2 d -> ... seq1 d")


class SimpleAttention(nn.Module):
    """A simple attention implementation to faciliate benchmarking."""

    def forward(self, q, k, v, is_causal):
        return scaled_dot_product_attention(q, k, v, is_causal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_use_flash", action="store_true")
    parser.add_argument("--do_backward", action="store_true")
    args = parser.parse_args()

    def test_timing_flash_forward_backward(sequence_length: int, dim: int):
        q, k, v = torch.randn(3, 1, sequence_length, dim, device="cuda", dtype=torch.float16, requires_grad=True)
        if args.no_use_flash:
            flash = torch.compile(SimpleAttention())
        else:
            flash = torch.compile(SimpleFlashAttention())

        print("Compiled attention layer successfully.")

        def flash_forward():
            _ = flash(q, k, v, True)

        def flash_forward_backward():
            o = flash(q, k, v, True)
            loss = o.sum()
            loss.backward()

        if args.do_backward:
            bench_fn = flash_forward_backward
        else:
            bench_fn = flash_forward

        return triton.testing.do_bench(bench_fn, rep=100, warmup=1000)

    results = []
    for seq_len in [128, 1024, 8192, 65536]:
        for dim in [16, 32, 64, 128]:
            try:
                avg_time = test_timing_flash_forward_backward(seq_len, dim)
                results.append(
                    {
                        "seq_len": seq_len,
                        "dim": dim,
                        "time": avg_time,
                    }
                )
                pprint.pprint(results)
            except Exception as e:
                results.append(
                    {
                        "seq_len": seq_len,
                        "dim": dim,
                        "error": str(e),
                    }
                )

    pprint.pprint(results)
