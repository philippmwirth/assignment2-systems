import torch
import torch.nn as nn
import einops
import timeit
from jaxtyping import Float
import math
import pandas as pd


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

    def forward(self, q, k, v):
        return scaled_dot_product_attention(q, k, v, True)


BATCH_SIZE = 1
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384, 65536]
N_REPEATS = 1000
N_WARMUP = 1000


if __name__ == "__main__":
    assert torch.cuda.is_available(), "no cuda available"
    device = torch.device("cuda")

    result_df = pd.DataFrame()
    for d_model in D_MODELS:
        for seq_len in SEQ_LENS:
            print(f"\n--- Benchmarking: d_model={d_model}, seq_len={seq_len} ---")
            try:
                model = SimpleAttention().to(device)
                # model = torch.compile(model)
                q = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
                k = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
                v = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)

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
                ts_forward = (t1_forward - t0_forward) / N_REPEATS

                torch.cuda.reset_peak_memory_stats(device)
                output = model(q, k, v)
                torch.cuda.synchronize()
                mem_before_backward = torch.cuda.memory_allocated(device) / (1024**2)  # in MB
                del output

                t0_backward = timeit.default_timer()
                for _ in range(N_REPEATS):
                    output = model(q, k, v)
                    output.mean().backward()
                    torch.cuda.synchronize()
                    del output
                t1_backward = timeit.default_timer()
                ts_backward = (t1_backward - t0_backward) / N_REPEATS

                result_df = pd.concat(
                    [
                        result_df,
                        pd.DataFrame(
                            [
                                {
                                    "d_model": d_model,
                                    "seq_len": seq_len,
                                    "Forward Time (ms)": f"{ts_forward:.5f}",
                                    "Backward Time (ms)": f"{ts_backward:.5f}",
                                    "Memory Before Backward (MB)": f"{mem_before_backward:.5f}",
                                }
                            ]
                        ),
                    ],
                )

            except torch.cuda.OutOfMemoryError:
                print("Failed: Out of Memory.")
                result_df = pd.concat(
                    [
                        result_df,
                        pd.DataFrame(
                            [
                                {
                                    "d_model": d_model,
                                    "seq_len": seq_len,
                                    "Forward Time (ms)": "OOM",
                                    "Backward Time (ms)": "OOM",
                                    "Memory Before Backward (MB)": "OOM",
                                }
                            ]
                        ),
                    ]
                )
                torch.cuda.empty_cache()
            print(result_df.to_markdown())

    print(result_df.to_markdown())
