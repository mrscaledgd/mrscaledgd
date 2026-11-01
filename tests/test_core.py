"""Checks of the properties the paper relies on. CPU-only, a few seconds."""

import itertools

import numpy as np
import pytest

from tubal import baselines, data, factors, metrics, problems, solver, transforms
from tubal.factors import herm

RNG = np.random.default_rng(1)


@pytest.mark.parametrize("name", ["DFT", "DCT"])
@pytest.mark.parametrize("n3", [5, 6])
def test_transform_roundtrip_and_norm(name, n3):
    T = transforms.make(name, n3)
    X = RNG.standard_normal((4, 3, n3))
    Xb = T.forward(X)
    assert Xb.shape == (T.m, 4, 3)
    assert np.allclose(T.inverse(Xb).real, X)
    assert np.isclose(T.norm(Xb), np.linalg.norm(X))


def test_data_dependent_transform_is_orthogonal():
    Y = RNG.standard_normal((5, 4, 6))
    T = transforms.make("UNI", 6, pilot=Y)
    X = RNG.standard_normal((5, 4, 6))
    assert np.allclose(T.inverse(T.forward(X)), X)
    assert np.isclose(T.norm(T.forward(X)), np.linalg.norm(X))


def tprod_dft(A, B):
    """t-product by its definition: block-circulant matrix times unfolded B."""
    n1, n2, n3 = A.shape
    n4 = B.shape[1]
    C = np.zeros((n1, n4, n3))
    for k in range(n3):
        for j in range(n3):
            C[:, :, k] += A[:, :, (k - j) % n3] @ B[:, :, j]
    return C


def test_dft_tproduct_is_slicewise_product():
    T = transforms.DFT(5)
    A = RNG.standard_normal((4, 3, 5))
    B = RNG.standard_normal((3, 2, 5))
    C = T.inverse(T.forward(A) @ T.forward(B)).real
    assert np.allclose(C, tprod_dft(A, B))


def test_allocation_is_optimal_by_brute_force():
    """Budgeted projection = best multi-rank approximation of cost <= s (DCT, equal weights)."""
    T = transforms.DCT(4)
    X = RNG.standard_normal((5, 5, 4)) * np.array([3.0, 1.0, 0.5, 0.2])
    Xb = T.forward(X)
    sv = np.linalg.svd(Xb, compute_uv=False)
    for s in range(0, 9):
        F, ranks = factors.budgeted_projection(Xb, T, s)
        err = np.linalg.norm(T.inverse(F.slices(5, 5, T.dtype)).real - X)
        best = min(
            np.sqrt(sum((sv[k, rk:] ** 2).sum() for k, rk in enumerate(rs)))
            for rs in itertools.product(range(6), repeat=4) if sum(rs) <= s
        )
        assert sum(ranks) <= s
        assert np.isclose(err, best)


def test_multirank_beats_tubal_at_equal_budget():
    T = transforms.DFT(6)
    X = data.synthetic(8, 8, T, [4, 2, 1, 1], kappa=3.0, rng=RNG)
    Xb = T.forward(X)
    for r in (1, 2):
        s = factors.tubal_budget(r, T)
        Ft = factors.truncated(Xb, [r] * T.m)
        Fm, _ = factors.budgeted_projection(Xb, T, s)
        et = np.linalg.norm(T.inverse(Ft.slices(8, 8, T.dtype)).real - X)
        em = np.linalg.norm(T.inverse(Fm.slices(8, 8, T.dtype)).real - X)
        assert em <= et + 1e-10


def test_synthetic_has_requested_multirank():
    T = transforms.DFT(6)
    ranks = [3, 2, 0, 1]
    X = data.synthetic(9, 7, T, ranks, kappa=10.0, rng=RNG)
    assert np.isrealobj(X)
    sv = np.linalg.svd(T.forward(X), compute_uv=False)
    assert [int((s > 1e-9 * sv.max()).sum()) for s in sv] == ranks


def test_scaledgd_step_matches_tensor_formula():
    """Transformed update = L - eta (G * R) * (R^H * R)^{-1} computed by t-products."""
    n1, n2, n3, r = 4, 3, 5, 2
    T = transforms.DFT(n3)
    L = RNG.standard_normal((n1, r, n3))
    R = RNG.standard_normal((n2, r, n3))
    G = RNG.standard_normal((n1, n2, n3))
    Lb, Rb, Gb = T.forward(L), T.forward(R), T.forward(G)
    Pinv = np.linalg.inv(herm(Rb) @ Rb)
    L_new_transformed = T.inverse(Lb - 0.5 * Gb @ Rb @ Pinv).real
    # tensor form: R^H under the DFT is the transpose with slices 2..n3 reversed
    RH = np.concatenate([R.transpose(1, 0, 2)[:, :, :1], R.transpose(1, 0, 2)[:, :, :0:-1]], axis=2)
    RHR = tprod_dft(RH, R)
    RHR_inv = T.inverse(np.linalg.inv(T.forward(RHR))).real
    L_new_tensor = L - 0.5 * tprod_dft(tprod_dft(G, R), RHR_inv)
    assert np.allclose(L_new_transformed, L_new_tensor)


def test_spurious_components_decay_geometrically():
    """Proposition: with rank(M) < r, U_perp^H L_t = (1 - eta)^t U_perp^H L_0 exactly."""
    n, rstar, r, eta = 12, 2, 4, 0.3
    U, _ = np.linalg.qr(RNG.standard_normal((n, n)))
    V, _ = np.linalg.qr(RNG.standard_normal((n, n)))
    M = (U[:, :rstar] * [2.0, 1.0]) @ V[:, :rstar].T
    L = RNG.standard_normal((n, r))
    R = RNG.standard_normal((n, r))
    B0 = U[:, rstar:].T @ L
    D0 = V[:, rstar:].T @ R
    for t in range(1, 6):
        E = L @ R.T - M
        L, R = (L - eta * E @ R @ np.linalg.inv(R.T @ R),
                R - eta * E.T @ L @ np.linalg.inv(L.T @ L))
        assert np.allclose(U[:, rstar:].T @ L, (1 - eta) ** t * B0)
        assert np.allclose(V[:, rstar:].T @ R, (1 - eta) ** t * D0)


@pytest.mark.parametrize("name", ["DFT", "DCT"])
def test_mr_scaledgd_recovers_nonuniform_multirank(name):
    T = transforms.make(name, 6)
    ranks = [3, 3, 2, 1] if name == "DFT" else [3, 3, 2, 1, 1, 0]
    X = data.synthetic(30, 30, T, ranks, kappa=5.0, rng=np.random.default_rng(3))
    mask = data.bernoulli_mask(X.shape, 0.5, np.random.default_rng(4))
    prob = problems.Completion(X, mask, T)
    s = factors.budget_cost(ranks, T)
    Xh, F, h = solver.solve(prob, T, method="scaledgd", budget=s, iters=300,
                            realloc_every=10, realloc_until=50, X_true=X, record_every=50)
    assert F.ranks == ranks
    assert h.rel_err[-1] < 1e-8


def test_tnn_admm_recovers_low_tubal_rank():
    T = transforms.DFT(5)
    X = data.synthetic(20, 20, T, [2, 2, 2], kappa=2.0, rng=np.random.default_rng(5))
    mask = data.bernoulli_mask(X.shape, 0.7, np.random.default_rng(6))
    Z, h = baselines.tnn_complete(X, mask, T, iters=300, X_true=X, record_every=50)
    assert h.rel_err[-1] < 1e-4


def test_metrics():
    X = RNG.random((16, 16, 3))
    assert metrics.ssim(X, X) == pytest.approx(1.0)
    assert metrics.psnr(X + 0.1, X, peak=1.0) == pytest.approx(20.0)
