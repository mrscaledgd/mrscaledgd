"""Baselines that are not ScaledGD-type factored methods.

tnn_complete   ADMM for min ||X||_TNN s.t. P_Omega(X) = P_Omega(Y), with the
               transformed tensor nuclear norm (Lu et al., IJCAI 2018 with the
               DFT; Lu, Peng and Wei, CVPR 2019 for other transforms), following
               the iteration of `lrtc_tnn` in Lu's LibADMM toolbox.
tctf_complete  TCTF (Zhou, Lu, Lin and Zhang, IEEE TIP 2018): alternating least
               squares on the transformed slices of X = L * R with an imputed
               data tensor, and a one-off decrease of the multi-rank at the
               largest gap of the pooled factor spectra (the LMaFit rule).
svp_solve      budgeted singular value projection (iterative hard thresholding):
               X <- P_s(X - mu G), P_s the budgeted transformed Eckart-Young
               projection. It is the reallocation step of MR-ScaledGD, used at
               every iteration.
"""

from __future__ import annotations

import time

import numpy as np

from .factors import Factors, budgeted_projection, herm
from .metrics import psnr
from .solver import History
from .transforms import Transform


def _record(hist, t, t_alg, X, X_true, peak, ranks, loss=None):
    hist.it.append(t)
    hist.time.append(t_alg)
    hist.ranks.append(list(map(int, ranks)))
    if loss is not None:
        hist.loss.append(loss)
    if X_true is not None:
        hist.rel_err.append(float(np.linalg.norm(X - X_true) / np.linalg.norm(X_true)))
        hist.psnr.append(psnr(X, X_true, peak))


def svt_slices(Zb, tau):
    out = np.empty_like(Zb)
    ranks = []
    for k in range(Zb.shape[0]):
        U, s, Vh = np.linalg.svd(Zb[k], full_matrices=False)
        s = np.maximum(s - tau, 0.0)
        r = int((s > 0).sum())
        ranks.append(r)
        out[k] = (U[:, :r] * s[:r]) @ Vh[:r]
    return out, ranks


def tnn_complete(Y, mask, T: Transform, *, iters=500, mu=1e-3, rho=1.1, mu_max=1e10,
                 tol=1e-8, X_true=None, record_every=1, peak=None):
    """Returns (X, history). Defaults are LibADMM's (mu, rho, mu_max, tol)."""
    obs = mask > 0
    Ym = mask * Y
    X = Ym.copy()
    Lam = np.zeros_like(Y)
    hist = History()
    t_alg = 0.0
    Z = X
    for t in range(iters):
        tic = time.perf_counter()
        Xk = X
        Zb, ranks = svt_slices(T.forward(X - Lam / mu), 1.0 / mu)
        Z = T.inverse(Zb).real
        X = Z + Lam / mu
        X[obs] = Ym[obs]
        dY = Z - X
        Lam = Lam + mu * dY
        mu = min(rho * mu, mu_max)
        chg = max(np.abs(Xk - X).max(), np.abs(dY).max())
        t_alg += time.perf_counter() - tic
        if t % record_every == 0 or chg < tol or t == iters - 1:
            _record(hist, t + 1, t_alg, Z, X_true, peak, ranks)
        if chg < tol:
            hist.events.append((t + 1, "converged"))
            break
    return Z, hist


def _pinv_solve_left(B, A):
    """B @ pinv(A) for a small square A, by least squares."""
    return np.linalg.lstsq(A.T, B.T, rcond=None)[0].T


def tctf_complete(Y, mask, T: Transform, *, r0=30, iters=500, tol=1e-5, gap=10.0,
                  rank_adjust=True, X_true=None, record_every=1, peak=None):
    """Returns (X, ranks, history)."""
    obs = mask > 0
    Ym = mask * Y
    n1, n2, _ = Y.shape
    Cb = T.forward(Ym)
    L, Rt = [], []  # Rt[k] is the r_k x n2 right factor
    for k in range(T.m):
        U, s, Vh = np.linalg.svd(Cb[k], full_matrices=False)
        r = min(r0, len(s))
        L.append(U[:, :r] * np.sqrt(s[:r]))
        Rt.append(np.sqrt(s[:r])[:, None] * Vh[:r])
    C = Ym.copy()
    hist = History()
    t_alg = 0.0
    for t in range(iters):
        tic = time.perf_counter()
        Cb = T.forward(C)
        for k in range(T.m):
            if L[k].shape[1] == 0:
                continue
            Rk = Rt[k]
            L[k] = _pinv_solve_left(Cb[k] @ herm(Rk), Rk @ herm(Rk))
            Lk = L[k]
            Rt[k] = np.linalg.lstsq(herm(Lk) @ Lk, herm(Lk) @ Cb[k], rcond=None)[0]
        Zb = np.stack([L[k] @ Rt[k] if L[k].shape[1] else np.zeros((n1, n2), T.dtype)
                       for k in range(T.m)])
        Z = T.inverse(Zb).real
        C_new = Z.copy()
        C_new[obs] = Ym[obs]
        chg = np.linalg.norm(C_new - C) / max(np.linalg.norm(C), 1e-300)
        C = C_new
        if rank_adjust:
            # pooled |diag(R)| of the QR factors of the L_k (LMaFit), with the DFT
            # multiplicity of each slice, sorted
            vals, owner = [], []
            for k in range(T.m):
                if L[k].shape[1]:
                    d = np.abs(np.diag(np.linalg.qr(L[k], mode="r")))
                    for v in np.sort(d)[::-1]:
                        vals.append(v)
                        owner.append(k)
            vals = np.asarray(vals)
            order = np.argsort(-vals, kind="stable")
            dv = vals[order]
            if len(dv) > 2:
                ratios = dv[:-1] / np.maximum(dv[1:], 1e-300)
                i = int(np.argmax(ratios))
                rel = ratios[i] * (len(dv) - 1) / max(ratios.sum() - ratios[i], 1e-300)
                if rel > gap:
                    keep = np.zeros(T.m, int)
                    for j in order[: i + 1]:
                        keep[owner[j]] += 1
                    for k in range(T.m):
                        rk = int(keep[k])
                        if rk < L[k].shape[1]:
                            U, s, Vh = np.linalg.svd(L[k] @ Rt[k], full_matrices=False)
                            L[k] = U[:, :rk] * np.sqrt(s[:rk])
                            Rt[k] = np.sqrt(s[:rk])[:, None] * Vh[:rk]
                    hist.events.append((t + 1, "rank decrease", keep.tolist()))
                    rank_adjust = False
        t_alg += time.perf_counter() - tic
        ranks = [Lk.shape[1] for Lk in L]
        if t % record_every == 0 or chg < tol or t == iters - 1:
            _record(hist, t + 1, t_alg, Z, X_true, peak, ranks)
        if chg < tol:
            hist.events.append((t + 1, "converged"))
            break
    return Z, ranks, hist


def svp_solve(prob, T: Transform, *, budget, iters=300, step=1.0, tol=0.0, X_true=None,
              record_every=1, peak=None, diverge_at=1e3):
    """Budgeted SVP / IHT. Returns (X, factors, history)."""
    n1, n2, _ = prob.shape
    hist = History()
    t_alg = 0.0
    tic = time.perf_counter()
    F, ranks = budgeted_projection(T.forward(prob.pilot()), T, budget)
    X = T.inverse(F.slices(n1, n2, T.dtype)).real
    t_alg += time.perf_counter() - tic
    for t in range(iters + 1):
        if t % record_every == 0 or t == iters:
            _record(hist, t, t_alg, X, X_true, peak, ranks, prob.loss(X))
            if X_true is not None and (not np.isfinite(hist.rel_err[-1]) or hist.rel_err[-1] > diverge_at):
                hist.events.append((t, "diverged"))
                break
        if t == iters:
            break
        tic = time.perf_counter()
        G = prob.grad(X)
        F, ranks = budgeted_projection(T.forward(X - step * G), T, budget)
        X_new = T.inverse(F.slices(n1, n2, T.dtype)).real
        t_alg += time.perf_counter() - tic
        chg = np.linalg.norm(X_new - X) / max(np.linalg.norm(X), 1e-300)
        X = X_new
        if tol and chg <= tol:
            _record(hist, t + 1, t_alg, X, X_true, peak, ranks, prob.loss(X))
            hist.events.append((t + 1, "converged"))
            break
    return X, F, hist
