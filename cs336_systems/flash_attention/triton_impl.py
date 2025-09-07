import math
from jaxtyping import Float
import torch

import triton
import triton.language as tl


class FlashAttention(torch.autograd.Function):
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

        # Launch grid: (Tq , batch_size).
        def grid(meta):
            return (triton.cdiv(n_queries, meta["TILE_SIZE"]), batch_size)

        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            n_queries,
            n_keys,
            scale,
            dim,
            is_causal,
        )
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, L, O = ctx.saved_tensors
        batch_size = Q.shape[0]
        n_queries, dim = Q.shape[-2:]
        n_keys = K.shape[-2]
        scale = 1 / math.sqrt(dim)

        # Compute D in global memory.
        D = (grad_out * O).sum(axis=-1)

        # Output buffers
        dQ = torch.empty_like(Q, dtype=torch.float32)
        dK = torch.empty_like(K, dtype=torch.float32)
        dV = torch.empty_like(V, dtype=torch.float32)

        # Launch grid: (Tq , batch_size)
        def grid_q(meta):
            return (triton.cdiv(n_queries, meta["TILE_SIZE"]), batch_size)

        flash_bwd_kernel_q[grid_q](
            Q,
            K,
            V,
            L,
            D,
            grad_out,
            dQ,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            L.stride(0),
            L.stride(1),
            D.stride(0),
            D.stride(1),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            dQ.stride(0),
            dQ.stride(1),
            dQ.stride(2),
            n_queries,
            n_keys,
            scale,
            dim,
            ctx.is_causal,
        )

        # Launch grid: (Tk , batch_size),
        def grid_kv(meta):
            return (triton.cdiv(n_keys, meta["TILE_SIZE"]), batch_size)

        flash_bwd_kernel_kv[grid_kv](
            Q,
            K,
            V,
            L,
            D,
            grad_out,
            dK,
            dV,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            L.stride(0),
            L.stride(1),
            D.stride(0),
            D.stride(1),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            dK.stride(0),
            dK.stride(1),
            dK.stride(2),
            dV.stride(0),
            dV.stride(1),
            dV.stride(2),
            n_queries,
            n_keys,
            scale,
            dim,
            ctx.is_causal,
        )
        return dQ, dK, dV, None  # None gradient for is_causal.


@triton.autotune(
    configs=[
        triton.Config({"TILE_SIZE": 16}),
        triton.Config({"TILE_SIZE": 32}),
        triton.Config({"TILE_SIZE": 64}),
        triton.Config({"TILE_SIZE": 128}),
    ],
    key=["N_QUERIES", "N_KEYS", "D"],
)
@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb: tl.constexpr,
    stride_qq: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oq: tl.constexpr,
    stride_od: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lq: tl.constexpr,
    N_QUERIES: tl.constexpr,
    N_KEYS: tl.constexpr,
    scale: tl.constexpr,
    D: tl.constexpr,
    is_causal: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    # Enable below for tests only.
    # **kwargs,
) -> None:
    """A tiled implementation of the flash attention fwd kernel."""
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * TILE_SIZE, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * TILE_SIZE, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * TILE_SIZE,),
        block_shape=(TILE_SIZE,),
        order=(0,),
    )

    # Prepare result accumulators.
    O = tl.zeros((TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((TILE_SIZE,), dtype=tl.float32)
    m = tl.full((TILE_SIZE,), value=-1e6, dtype=tl.float32)

    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    for i in range(tl.cdiv(N_KEYS, TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        S = tl.dot(Q, tl.trans(K)) * scale
        if is_causal:
            row_idx = tl.arange(0, TILE_SIZE)[:, None]
            col_idx = tl.arange(0, TILE_SIZE)[None, :]
            off_diagonal = query_tile_index * TILE_SIZE - i * TILE_SIZE
            mask = col_idx > (row_idx + off_diagonal)
            S -= mask * 1e6
        m_i = tl.maximum(m, tl.max(S, axis=1))
        P = tl.exp(S - m_i[:, None])
        l_i = tl.exp(m - m_i) * l + tl.sum(P, axis=1)
        P = P.to(V.dtype)  # Tip from assignment.
        O_i = O * tl.exp(m - m_i)[:, None]
        O_i += tl.dot(P.to(V.dtype), V)

        # Move pointers.
        O = O_i
        l = l_i
        m = m_i
        K_block_ptr = tl.advance(K_block_ptr, (TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (TILE_SIZE, 0))

    # Compute and store final results.
    O_final = O / l[:, None]
    L_final = m + tl.log(l)
    tl.store(O_block_ptr, O_final.to(Q.dtype), boundary_check=(0, 1))
    tl.store(L_block_ptr, L_final, boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({"TILE_SIZE": 16}),
        triton.Config({"TILE_SIZE": 32}),
        triton.Config({"TILE_SIZE": 64}),
        triton.Config({"TILE_SIZE": 128}),
    ],
    key=["N_QUERIES", "N_KEYS", "D"],
)
@triton.jit
def flash_bwd_kernel_q(
    Q_ptr,
    K_ptr,
    V_ptr,
    L_ptr,
    D_ptr,
    dO_ptr,
    dQ_ptr,
    stride_qb: tl.constexpr,
    stride_qq: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lq: tl.constexpr,
    stride_db: tl.constexpr,
    stride_dq: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doq: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_dqb: tl.constexpr,
    stride_dqq: tl.constexpr,
    stride_dqd: tl.constexpr,
    N_QUERIES: tl.constexpr,
    N_KEYS: tl.constexpr,
    scale: tl.constexpr,
    D: tl.constexpr,
    is_causal: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    # Enable below for tests only.
    # **kwargs,
):
    """A tiled implementation of the flash attention bwd kernel for calculating dQ."""
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * TILE_SIZE, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * TILE_SIZE,),
        block_shape=(TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * TILE_SIZE,),
        block_shape=(TILE_SIZE,),
        order=(0,),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * TILE_SIZE, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * TILE_SIZE, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )

    # Prepare dQ result accumulator.
    dQ = tl.zeros((TILE_SIZE, D), dtype=tl.float32)

    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    for i in range(tl.cdiv(N_QUERIES, TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        D_ = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        S = tl.dot(Q, tl.trans(K)) * scale
        if is_causal:
            row_idx = tl.arange(0, TILE_SIZE)[:, None]
            col_idx = tl.arange(0, TILE_SIZE)[None, :]
            off_diagonal = query_tile_index * TILE_SIZE - i * TILE_SIZE
            mask = col_idx > (row_idx + off_diagonal)
            S -= mask * 1e6
        P = tl.exp(S - L[:, None])
        dP = tl.dot(dO, tl.trans(V))
        dS = P * (dP - D_[:, None]) * scale
        dS = dS.to(K.dtype)
        dQ += tl.dot(dS, K)

        # Move pointers.
        K_block_ptr = tl.advance(K_block_ptr, (TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (TILE_SIZE,))
        D_block_ptr = tl.advance(D_block_ptr, (TILE_SIZE,))

    # Store result.
    tl.store(dQ_block_ptr, dQ, boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({"TILE_SIZE": 16}),
        triton.Config({"TILE_SIZE": 32}),
        triton.Config({"TILE_SIZE": 64}),
        triton.Config({"TILE_SIZE": 128}),
    ],
    key=["N_QUERIES", "N_KEYS", "D"],
)
@triton.jit
def flash_bwd_kernel_kv(
    Q_ptr,
    K_ptr,
    V_ptr,
    L_ptr,
    D_ptr,
    dO_ptr,
    dK_ptr,
    dV_ptr,
    stride_qb: tl.constexpr,
    stride_qq: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lq: tl.constexpr,
    stride_db: tl.constexpr,
    stride_dq: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doq: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkq: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvq: tl.constexpr,
    stride_dvd: tl.constexpr,
    N_QUERIES: tl.constexpr,
    N_KEYS: tl.constexpr,
    scale: tl.constexpr,
    D: tl.constexpr,
    is_causal: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    # Enable below for tests only.
    # **kwargs,
):
    """A tiled implementation of the flash attention fwd kernel."""
    # Program indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * TILE_SIZE, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * TILE_SIZE, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(TILE_SIZE,),
        order=(0,),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkq, stride_dkd),
        offsets=(key_tile_index * TILE_SIZE, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvq, stride_dvd),
        offsets=(key_tile_index * TILE_SIZE, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )

    # Prepare dK, dV result accumulators.
    dK = tl.zeros((TILE_SIZE, D), dtype=tl.float32)
    dV = tl.zeros((TILE_SIZE, D), dtype=tl.float32)

    K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    for i in range(tl.cdiv(N_QUERIES, TILE_SIZE)):
        Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        D_ = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        S = tl.dot(Q, tl.trans(K)) * scale
        if is_causal:
            row_idx = tl.arange(0, TILE_SIZE)[:, None]
            col_idx = tl.arange(0, TILE_SIZE)[None, :]
            off_diagonal = i * TILE_SIZE - key_tile_index * TILE_SIZE
            mask = col_idx > (row_idx + off_diagonal)
            S -= mask * 1e6
        P = tl.exp(S - L[:, None])
        P = P.to(dO.dtype)
        dV += tl.dot(tl.trans(P), dO)
        dP = tl.dot(dO, tl.trans(V))
        dS = P * (dP - D_[:, None]) * scale
        dS = dS.to(Q.dtype)
        dK += tl.dot(tl.trans(dS), Q)

        # Move pointers.
        Q_block_ptr = tl.advance(Q_block_ptr, (TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (TILE_SIZE,))
        D_block_ptr = tl.advance(D_block_ptr, (TILE_SIZE,))

    # Store results.
    tl.store(dK_block_ptr, dK, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV, boundary_check=(0, 1))


if __name__ == "__main__":

    def test_timing_flash_forward_backward():
        n_heads = 16
        d_head = 64
        sequence_length = 16384
        q, k, v = torch.randn(
            3, n_heads, sequence_length, d_head, device="cuda", dtype=torch.float16, requires_grad=True
        )
        flash = torch.compile(FlashAttention.apply)

        def flash_forward_backward():
            o = flash(q, k, v, True)
            loss = o.sum()
            loss.backward()

        results = triton.testing.do_bench(flash_forward_backward, rep=1000, warmup=1000)
        print(results)

    test_timing_flash_forward_backward()
