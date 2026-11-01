"""Factored tensors with one rank per transformed slice.

A tensor X is represented in the transformed domain by slice factors
Xbar_k = L_k R_k^H, with L_k of shape (n1, r_k) and R_k of shape (n2, r_k).
The tubal-rank-r factorization of the t-product literature is the special case
r_k = r for every k; the multi-rank vector (r_1, ..., r_m) is the general one.

Ranks are budgeted in full-domain components: a stored DFT slice that stands
for a conjugate pair costs weights[k] = 2 per component. A tubal rank r then
costs n3 * r, the same number of real parameters as s = n3 * r spread freely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transforms import Transform


def herm(A: np.ndarray) -> np.ndarray:
    return np.conj(np.swapaxes(A, -1, -2))


@dataclass
class Factors:
    L: list
    R: list

    @property
    def ranks(self) -> list[int]:
        return [Lk.shape[1] for Lk in self.L]

    def slices(self, n1: int, n2: int, dtype) -> np.ndarray:
        out = np.zeros((len(self.L), n1, n2), dtype=dtype)
        for k, (Lk, Rk) in enumerate(zip(self.L, self.R)):
            if Lk.shape[1]:
                out[k] = Lk @ herm(Rk)
        return out

    def copy(self) -> "Factors":
        return Factors([Lk.copy() for Lk in self.L], [Rk.copy() for Rk in self.R])


def budget_cost(ranks, T: Transform) -> float:
    return float(np.dot(np.asarray(ranks, float), T.weights))


def tubal_budget(r: int, T: Transform) -> int:
    """Budget of a tubal-rank-r factorization, in full-domain components."""
    return int(round(budget_cost([r] * T.m, T)))


def allocate(svals: np.ndarray, T: Transform, budget: float, rmax: int | None = None,
             floor: float = 0.0) -> list[int]:
    """Multi-rank that keeps the `budget` largest transformed singular values.

    This is the transformed-domain Eckart-Young truncation: among tensors whose
    multi-rank costs at most `budget`, it is the best Frobenius approximation
    (each kept singular value is weighted by the slice multiplicity, and the
    error is a weighted sum of discarded squared singular values). Ties and the
    integrality of the weights are handled greedily, which is exact when all
    weights are equal and within one component of exact otherwise. Singular
    values at or below `floor` are never kept, so the budget is an upper bound.
    """
    m, q = svals.shape
    cap = q if rmax is None else min(q, rmax)
    order = np.argsort(-svals[:, :cap], axis=None, kind="stable")
    ranks = np.zeros(m, dtype=int)
    used = 0.0
    for flat in order:
        k, i = divmod(int(flat), cap)
        if svals[k, i] <= floor or svals[k, i] <= 0:
            break  # sorted: nothing below this point can be kept
        if used + T.weights[k] > budget + 1e-9:
            continue
        if i != ranks[k]:  # keep components in order within a slice
            continue
        ranks[k] += 1
        used += T.weights[k]
        if used >= budget - 1e-9:
            break
    return ranks.tolist()


def truncated(Zb: np.ndarray, ranks) -> Factors:
    """Balanced factors of the per-slice rank-r_k truncation of Zb."""
    L, R = [], []
    for k, rk in enumerate(ranks):
        if rk == 0:
            L.append(np.zeros((Zb.shape[1], 0), Zb.dtype))
            R.append(np.zeros((Zb.shape[2], 0), Zb.dtype))
            continue
        U, s, Vh = np.linalg.svd(Zb[k], full_matrices=False)
        rs = np.sqrt(s[:rk])
        L.append(U[:, :rk] * rs)
        R.append(herm(Vh[:rk]) * rs)
    return Factors(L, R)


def slice_svals(Zb: np.ndarray) -> np.ndarray:
    return np.linalg.svd(Zb, compute_uv=False)


def budgeted_projection(Zb: np.ndarray, T: Transform, budget: float,
                        floor: float = 0.0) -> tuple[Factors, list[int]]:
    """Factors of the best approximation of Zb with multi-rank cost <= budget,
    keeping only singular values above `floor`."""
    U, s, Vh = np.linalg.svd(Zb, full_matrices=False)
    ranks = allocate(s, T, budget, floor=floor)
    L, R = [], []
    for k, rk in enumerate(ranks):
        rs = np.sqrt(s[k, :rk])
        L.append(U[k][:, :rk] * rs)
        R.append(herm(Vh[k][:rk]) * rs)
    return Factors(L, R), ranks
