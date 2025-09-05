import math
from jaxtyping import Float
import einops
import torch

# Required fix, no idea why.
# https://github.com/triton-lang/triton/issues/5142
import os
os.environ["TRITON_INTERPRET"] = "1"

import triton
import triton.language as tl


class FlashAttention(torch.autograd.Function):

    q_tile_size = 16
    k_tile_size = 16

    @staticmethod
    def forward(
        ctx,
        Q: Float[torch.Tensor, "... d"],
        K: Float[torch.Tensor, "... d"],
        V: Float[torch.Tensor, "... d"],
        is_causal: bool = False,
    ):
        batch_size = Q.shape[0]
        n_queries, dim = Q.shape[-2:]
        n_keys = K.shape[-2]
        scale = 1 / math.sqrt(dim)

        # Output buffers
        O = torch.empty_like(Q)
        L = torch.empty((batch_size, n_queries), device=Q.device)

        # Launch grid: (Tq , batch_size),
        flash_fwd_kernel[(tl.cdiv(n_queries, FlashAttention.q_tile_size), batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            scale,
            dim,
            FlashAttention.q_tile_size,
            FlashAttention.k_tile_size,
            is_causal,
        )
        ctx.save_for_backward(L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError()


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """A tiled implementation of the flash attention fwd kernel.

    Args:
        Q_ptr: Pointer to the query data.
        K_ptr: Pointer to the keys data.
        V_ptr: Pointer to the values data.
        stride_qb: Stride along query data batch dimension.
        stride_qq: Stride along first query data dimension.
        stride_qd: Stride along second query data dimension.
        ...
        N_QUERIES: Size of first query data dimension.
        N_KEYS: Size of first keys and values dimension.
        scale: 1/sqrt(dim)
        D: dim
        Q_TILE_SIZE: B_q
        K_TILE_SIZE: B_k
        is_causal: Whether or not to apply causal masks.
    """
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Prepare result accumulators.
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), value=-math.inf, dtype=tl.float32)

    # Load Q tile only once.
    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # Q_TILE_SIZE x D
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # K_TILE_SIZE x D
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # K_TILE_SIZE x D

        # Tile pre softmax attention scores
        S = tl.dot(Q, tl.trans(K)) * scale
        if is_causal:
            row_idx = tl.arange(0, Q_TILE_SIZE)[:, None]
            col_idx = tl.arange(0, K_TILE_SIZE)[None, :]
            off_diagonal = query_tile_index * Q_TILE_SIZE - i * K_TILE_SIZE
            mask = col_idx > (row_idx + off_diagonal)
            S -= mask * 1e6
        m_i = tl.maximum(m, tl.max(S, axis=1))
        P = tl.exp(S - m_i[:, None])
        l_i = tl.exp(m - m_i) * l + tl.sum(P, axis=1)
        P = P.to(V.dtype) # Tip from assignment.
        O_i = O * tl.exp(m - m_i)[:, None]
        O_i += tl.dot(P.to(V.dtype), V)

        # Update pointers.
        O = O_i
        l = l_i
        m = m_i
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # Compute results.
    O_final = O / l[:, None]
    L_final = m + tl.log(l)

    # Store results.
    tl.store(O_block_ptr, O_final.to(Q.dtype), boundary_check=(0, 1))
    tl.store(L_block_ptr, L_final, boundary_check=(0,))