import torch
import torch.nn as nn
import einops
import timeit
import pandas
from jaxtyping import Float, Bool
import math
import time
import itertools
import pandas as pd


def softmax(x: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    y = (x - x.max(dim=dim, keepdim=True).values).exp()
    return y / y.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: Float[torch.Tensor, "b s d"],
    k: Float[torch.Tensor, "b s d"],
    v: Float[torch.Tensor, "b s d"],
    mask: Bool[torch.Tensor, "s s"],
) -> Float[torch.Tensor, "b s d"]:
    d_k = k.shape[-1]
    qk_scaled = einops.einsum(q, k, "... seq1 d, ... seq2 d -> ... seq1 seq2") / math.sqrt(d_k)
    qk_scaled[..., ~mask] = -math.inf
    weights = softmax(x=qk_scaled, dim=-1)
    return einops.einsum(weights, v, "... seq1 seq2, ... seq2 d -> ... seq1 d")


class SimpleAttention(nn.Module):
    """A simple attention implementation to faciliate benchmarking."""

    def forward(self, q, k, v):
        return scaled_dot_product_attention(q, k, v)


BATCH_SIZE = 8
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
N_REPEATS = 100
N_WARMUP = 10


if __name__ == "__main__":
    assert torch.cuda.is_available(), "no cuda available"
    device = torch.device("cuda")

    result_df = pd.DataFrame()
    for d_model in D_MODELS:
        for seq_len in SEQ_LENS:
            print(f"\n--- Benchmarking: d_model={d_model}, seq_len={seq_len} ---")
            try:
                model = SimpleAttention().to(device)
                q = torch.randn(BATCH_SIZE, seq_len, d_model, device=device)
                k = torch.randn(BATCH_SIZE, seq_len, d_model, device=device)
                v = torch.randn(BATCH_SIZE, seq_len, d_model, device=device)

                for _ in range(N_WARMUP):
                    output = model(q, k, v)
                    grad_output = torch.randn_like(output)
                    output.backward(grad_output, retain_graph=False)

                torch.cuda.synchronize()
                t0_forward = timeit.default_timer()
                for _ in range(N_REPEATS):
                    _ = model(q, k, v)
                    torch.cuda.synchronize()
                t1_forward = timeit.default_timer()
                t_forward = (t0_forward - t1_forward) / N_REPEATS

                torch.cuda.reset_peak_memory_stats(device)
                output = model(q, k, v)
                torch.cuda.synchronize()
                mem_before_backward = torch.cuda.memory_allocated(device) / (1024**2)  # in MB
                del output

                t0_backward = timeit.default_timer()
                for _ in range(N_REPEATS):
                    output = model(q, k, v)
                    output.backward()
                    torch.cuda.synchronize()
                t1_backward = timeit.default_timer()
                ts_backward = (t1_backward - t0_backward) / N_REPEATS

                result_df.append(
                    {
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "Forward Time (ms)": f"{avg_time_fwd:.2f}",
                        "Backward Time (ms)": f"{avg_time_bwd:.2f}",
                        "Memory Before Backward (MB)": f"{mem_before_backward:.2f}",
                    },
                    ignore_index=True,
                )

            except torch.cuda.OutOfMemoryError:
                print("Failed: Out of Memory.")
                result_df.append(
                    {
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "Forward Time (ms)": "OOM",
                        "Backward Time (ms)": "OOM",
                        "Memory Before Backward (MB)": "OOM",
                    },
                    ignore_index=True,
                )
                torch.cuda.empty_cache()

    print(result_df.to_markdown())
