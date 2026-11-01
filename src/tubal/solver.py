"""Factored first-order methods for transformed low-tubal-rank estimation.

The estimate is X = L *_Phi R^H, held as slice factors Xbar_k = L_k R_k^H in the
transformed domain, with ranks r_k (one per stored slice). With G the gradient
of the loss at X (problems.py), the methods are:

  gd        L_k <- L_k - (eta / sigma_1) Gbar_k R_k              (and R_k alike)
  scaledgd  L_k <- L_k - eta Gbar_k R_k (R_k^H R_k + lambda_k I)^{-1}
            the ScaledGD of Wu (2026) when lambda = 0 and r_k = r for all k
  apgd      alternating damped preconditioned steps with rebalancing after each
            half-step and lambda = f(X) / 10 (Liu et al., 2025)

Parameterizations: `rank=r` gives the tubal-rank factorization (r_k = r);
`budget=s` gives a multi-rank factorization whose ranks keep the s largest
transformed singular values of the pilot (budgeted transformed Eckart-Young).
With `realloc_every=K`, every K-th iteration up to `realloc_until` is replaced
by a budgeted SVP step X <- P_s(X - G), which moves rank between slices.

Damping (`damping`, `c_lam`):
  none    lambda = 0
  leak    lambda_k = c * nu(X), nu the predicted spectral norm of the sampling
          leakage in a slice (problems.py); it vanishes with the residual
  slice   lambda_k = c * ||Gbar_k||_F
  global  lambda_k = c * (weighted rms of ||Gbar_k||_F over slices)

With `c_floor`, the allocation keeps only singular values above c_floor * nu
(nu the leakage level), so the budget becomes an upper bound: components
below the sampling noise of their slice are not detectable and are not fitted.

MR-ScaledGD, the method of the paper, is scaledgd with a budget, reallocation,
the leakage floor and leak damping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .factors import Factors, budgeted_projection, herm, truncated, tubal_budget
from .metrics import psnr
from .transforms import Transform


@dataclass
class History:
    it: list = field(default_factory=list)
    time: list = field(default_factory=list)
    rel_err: list = field(default_factory=list)
    psnr: list = field(default_factory=list)
    loss: list = field(default_factory=list)
    ranks: list = field(default_factory=list)
    cond: list = field(default_factory=list)
    lam: list = field(default_factory=list)
    extra: list = field(default_factory=list)
    events: list = field(default_factory=list)

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _right_solve(A, P):
    """A @ inv(P) through a linear solve (LU with partial pivoting)."""
    return np.linalg.solve(P.T, A.T).T


def _rebalance(Lk, Rk):
    if Lk.shape[1] == 0:
        return Lk, Rk
    QL, WL = np.linalg.qr(Lk)
    QR, WR = np.linalg.qr(Rk)
    U, s, Vh = np.linalg.svd(WL @ herm(WR))
    rs = np.sqrt(s)
    return (QL @ U) * rs, (QR @ herm(Vh)) * rs


def _gram_cond(F):
    worst = 1.0
    for Rk in F.R:
        if Rk.shape[1]:
            ev = np.linalg.eigvalsh(herm(Rk) @ Rk)
            worst = max(worst, ev[-1] / max(ev[0], 1e-300))
    return float(worst)


def initial_factors(prob, T: Transform, rank=None, budget=None, c_floor=0.0):
    Zb = T.forward(prob.pilot())
    if budget is None:
        if rank is None:
            raise ValueError("give a tubal rank or a budget")
        return truncated(Zb, [rank] * T.m), tubal_budget(rank, T)
    floor = c_floor * prob.leak(np.zeros(prob.shape)) if c_floor else 0.0
    F, _ = budgeted_projection(Zb, T, budget, floor=floor)
    return F, budget


def solve(
    prob,
    T: Transform,
    *,
    method: str = "scaledgd",
    rank: int | None = None,
    budget: float | None = None,
    iters: int = 300,
    eta: float = 0.5,
    damping: str = "none",
    c_lam: float = 0.0,
    realloc_every: int = 0,
    realloc_until: int = 0,
    c_floor: float = 0.0,
    tol: float = 0.0,
    X_true: np.ndarray | None = None,
    record_every: int = 1,
    peak: float | None = None,
    diverge_at: float = 1e3,
    diagnostics: bool = False,
    init: Factors | None = None,
    callback=None,
):
    """Run one factored method. Returns (X, factors, history).

    `tol` stops once ||X_{t+1} - X_t||_F <= tol ||X_t||_F after the
    reallocation phase. Time in the history excludes the bookkeeping against
    X_true. `callback(t, X, F)`, called at every record, returns a dict stored
    in history.extra.
    """
    n1, n2, _ = prob.shape
    if init is None:
        F, budget = initial_factors(prob, T, rank, budget, c_floor)
    else:
        F = init.copy()
        budget = budget if budget is not None else sum(r * w for r, w in zip(F.ranks, T.weights))
    sig1 = max((np.linalg.norm(Lk, 2) ** 2 for Lk in F.L if Lk.shape[1]), default=1.0)
    hist = History()
    nrm_true = float(np.linalg.norm(X_true)) if X_true is not None else None
    t_alg = 0.0
    tic = time.perf_counter()
    X = T.inverse(F.slices(n1, n2, T.dtype)).real
    t_alg += time.perf_counter() - tic

    stop = False
    for t in range(iters + 1):
        tic = time.perf_counter()
        G = prob.grad(X)
        t_alg += time.perf_counter() - tic
        if t % record_every == 0 or t == iters or stop:
            hist.it.append(t)
            hist.time.append(t_alg)
            hist.loss.append(prob.loss(X))
            hist.ranks.append(F.ranks)
            if diagnostics:
                hist.cond.append(_gram_cond(F))
            if callback is not None:
                hist.extra.append(callback(t, X, F))
            if X_true is not None:
                e = float(np.linalg.norm(X - X_true) / nrm_true)
                hist.rel_err.append(e)
                hist.psnr.append(psnr(X, X_true, peak))
                if not np.isfinite(e) or e > diverge_at:
                    hist.events.append((t, "diverged"))
                    break
            elif not np.isfinite(hist.loss[-1]):
                hist.events.append((t, "diverged"))
                break
        if t == iters or stop:
            break

        tic = time.perf_counter()
        realloc = bool(realloc_every) and t < realloc_until and t % realloc_every == realloc_every - 1
        if realloc:
            floor = c_floor * prob.leak(X) if c_floor else 0.0
            try:
                F, ranks = budgeted_projection(T.forward(X - G), T, budget, floor=floor)
            except np.linalg.LinAlgError:  # the SVD fails on an iterate that has blown up
                hist.events.append((t, "diverged"))
                t_alg += time.perf_counter() - tic
                break
            hist.events.append((t, "realloc", list(map(int, ranks))))
        else:
            Gb = T.forward(G)
            lam = _lambdas(prob, X, Gb, T, damping, c_lam)
            if diagnostics and t % record_every == 0:
                hist.lam.append(float(np.max(lam)) if len(lam) else 0.0)
            try:
                F = _step(prob, T, F, Gb, method, eta, lam, sig1, n1, n2)
            except np.linalg.LinAlgError:
                hist.events.append((t, "singular"))
                t_alg += time.perf_counter() - tic
                break
        X_new = T.inverse(F.slices(n1, n2, T.dtype)).real
        t_alg += time.perf_counter() - tic
        if tol and not (bool(realloc_every) and t < realloc_until):
            chg = np.linalg.norm(X_new - X) / max(np.linalg.norm(X), 1e-300)
            if chg <= tol:
                hist.events.append((t + 1, "converged"))
                stop = True
        X = X_new
    return X, F, hist


def _lambdas(prob, X, Gb, T, damping, c_lam):
    if damping == "none" or c_lam == 0:
        return np.zeros(T.m)
    if damping == "leak":
        return np.full(T.m, c_lam * prob.leak(X))
    rho = np.linalg.norm(Gb, axis=(1, 2))
    if damping == "slice":
        return c_lam * rho
    if damping == "global":
        return np.full(T.m, c_lam * np.sqrt(np.dot(T.weights, rho**2) / T.weights.sum()))
    raise ValueError(f"unknown damping {damping!r}")


def _step(prob, T, F, Gb, method, eta, lam, sig1, n1, n2):
    L, R = F.L, F.R
    if method == "gd":
        step = eta / sig1
        return Factors([Lk - step * Gb[k] @ R[k] for k, Lk in enumerate(L)],
                       [Rk - step * herm(Gb[k]) @ L[k] for k, Rk in enumerate(R)])
    if method == "scaledgd":
        Ln, Rn = [], []
        for k in range(T.m):
            Lk, Rk = L[k], R[k]
            r = Lk.shape[1]
            if r == 0:
                Ln.append(Lk)
                Rn.append(Rk)
                continue
            D = lam[k] * np.eye(r)
            Ln.append(Lk - eta * _right_solve(Gb[k] @ Rk, herm(Rk) @ Rk + D))
            Rn.append(Rk - eta * _right_solve(herm(Gb[k]) @ Lk, herm(Lk) @ Lk + D))
        return Factors(Ln, Rn)
    if method == "apgd":
        X = T.inverse(F.slices(n1, n2, T.dtype)).real
        lam0 = prob.loss(X) / 10.0
        Lh = [Lk if Lk.shape[1] == 0 else
              Lk - eta * _right_solve(Gb[k] @ R[k], herm(R[k]) @ R[k] + lam0 * np.eye(Lk.shape[1]))
              for k, Lk in enumerate(L)]
        pairs = [_rebalance(Lh[k], R[k]) for k in range(T.m)]
        Lh, Rh = [a for a, _ in pairs], [b for _, b in pairs]
        Xh = T.inverse(Factors(Lh, Rh).slices(n1, n2, T.dtype)).real
        Gh = T.forward(prob.grad(Xh))
        Rn = [Rk if Rk.shape[1] == 0 else
              Rk - eta * _right_solve(herm(Gh[k]) @ Lh[k], herm(Lh[k]) @ Lh[k] + lam0 * np.eye(Rk.shape[1]))
              for k, Rk in enumerate(Rh)]
        pairs = [_rebalance(Lh[k], Rn[k]) for k in range(T.m)]
        return Factors([a for a, _ in pairs], [b for _, b in pairs])
    raise ValueError(f"unknown method {method!r}")
