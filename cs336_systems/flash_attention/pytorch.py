import einops
import math
import torch
from jaxtyping import Float

B_q = 16
B_k = 16


class FlashAttentionReference(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[torch.Tensor, "... d"],
        K: Float[torch.Tensor, "... d"],
        V: Float[torch.Tensor, "... d"],
        is_causal: bool = False,
    ):
        b = Q.shape[0]
        d = Q.shape[-1]
        N_q = Q.shape[-2]
        N_k = K.shape[-2]
        Os = []
        Ls = []
        for i in range(0, N_q, B_q):
            # No need to check boundaries according to assignment.
            Q_i = Q[:, i : i + B_q]
            O_i = torch.zeros(b, B_q, d)
            l_i = torch.zeros(b, B_q)
            m_i = -torch.ones(b, B_q) * math.inf
            for j in range(0, N_k, B_k):
                K_j = K[:, j : j + B_k]
                V_j = V[:, j : j + B_k]
                # Tile pre softmax attention scores
                S_j = einops.einsum(Q_i, K_j, "... seq1 d, ... seq2 d -> ... seq1 seq2") / math.sqrt(d)
                m_j = torch.maximum(m_i, S_j.max(dim=-1).values)
                P_j = (S_j - m_j.unsqueeze(-1)).exp()
                l_j = (m_i - m_j.squeeze()).exp() * l_i + P_j.sum(dim=-1)
                O_j = torch.diag_embed((m_i - m_j).exp()) @ O_i + P_j @ V_j
                # Update pointers.
                O_i = O_j
                l_i = l_j
                m_i = m_j
            # Aggregate results.
            O_i = torch.diag_embed(1.0 / l_i) @ O_i
            L_i = m_i + l_i.log()
            Os.append(O_i)
            Ls.append(L_i)

        L = torch.concat(Ls, dim=1)
        O = torch.concat(Os, dim=1)
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.scale = 1 / math.sqrt(d)
        ctx.is_causal = is_causal
        ctx.seq_len = N_q
        return O

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, L, O = ctx.saved_tensors

        @torch.compile()
        def grads_from_recompute(Q_, K_, V_, L_, O_, dO):
            D = (O_ * dO).sum(axis=-1)
            S = einops.einsum(Q_, K_, "... seq1 d, ... seq2 d -> ... seq1 seq2") * ctx.scale
            if ctx.is_causal:
                mask = torch.triu(torch.ones((ctx.seq_len, ctx.seq_len), dtype=torch.long), diagonal=1).bool()
                S[..., mask] -= 1e6
            P = (S - L_.unsqueeze(-1)).exp()
            dV = einops.einsum(P, dO, "... seq1 seq2, ... seq1 d -> ... seq2 d")
            dP = einops.einsum(dO, V_, "... seq1 d, ... seq2 d -> ... seq1 seq2")
            dS = P * (dP - D.unsqueeze(-1))
            dQ = einops.einsum(dS, K_, "... seq1 seq2, ... seq2 d -> ... seq1 d") * ctx.scale
            dK = einops.einsum(dS, Q_, "... seq1 seq2, ... seq1 d -> ... seq2 d") * ctx.scale
            return dQ, dK, dV, None

        return grads_from_recompute(Q.detach(), K.detach(), V.detach(), L.detach(), O.detach(), grad_out)
