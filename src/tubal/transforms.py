"""Invertible transforms along the third mode, and the transformed domain.

A transform L maps a real tensor X of shape (n1, n2, n3) to its transformed
frontal slices, stored batch-first as an array of shape (m, n1, n2). The
t-product under L is the slice-wise matrix product in that domain.

Every transform here satisfies Phi Phi^H = ell I, so that

    ||X||_F^2 = (1/ell) * sum_k weights[k] * ||Xbar_k||_F^2 .

`weights` counts how many full-domain slices each stored slice stands for: the
DFT of a real tensor is conjugate-symmetric, so only m = n3 // 2 + 1 slices are
stored and the interior ones count twice.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft


class Transform:
    name: str
    n3: int
    m: int
    ell: float
    weights: np.ndarray
    dtype: type

    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, Xb: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def norm(self, Xb: np.ndarray) -> float:
        """Frobenius norm of the original-domain tensor, from its slices."""
        sq = np.einsum("k,k->", self.weights, np.sum(np.abs(Xb) ** 2, axis=(1, 2)))
        return float(np.sqrt(sq / self.ell))


class DFT(Transform):
    """Unnormalized DFT along mode 3 (ell = n3); stores n3 // 2 + 1 slices."""

    def __init__(self, n3: int):
        self.name, self.n3, self.ell = "DFT", n3, float(n3)
        self.m = n3 // 2 + 1
        w = np.full(self.m, 2.0)
        w[0] = 1.0
        if n3 % 2 == 0:
            w[-1] = 1.0
        self.weights = w
        self.dtype = np.complex128

    def forward(self, X):
        return np.ascontiguousarray(np.moveaxis(sfft.rfft(X, axis=2), 2, 0))

    def inverse(self, Xb):
        return sfft.irfft(np.moveaxis(Xb, 0, 2), n=self.n3, axis=2)


class DCT(Transform):
    """Orthonormal DCT-II along mode 3 (ell = 1)."""

    def __init__(self, n3: int):
        self.name, self.n3, self.ell, self.m = "DCT", n3, 1.0, n3
        self.weights = np.ones(n3)
        self.dtype = np.float64

    def forward(self, X):
        return np.ascontiguousarray(np.moveaxis(sfft.dct(X, type=2, norm="ortho", axis=2), 2, 0))

    def inverse(self, Xb):
        return sfft.idct(np.moveaxis(Xb, 0, 2), type=2, norm="ortho", axis=2)


class Orthogonal(Transform):
    """A given real orthogonal n3 x n3 matrix Phi (ell = 1), e.g. data-dependent."""

    def __init__(self, Phi: np.ndarray, name: str = "UNI"):
        Phi = np.asarray(Phi, dtype=float)
        n3 = Phi.shape[0]
        if not np.allclose(Phi @ Phi.T, np.eye(n3), atol=1e-8):
            raise ValueError("Phi must be orthogonal")
        self.name, self.n3, self.ell, self.m = name, n3, 1.0, n3
        self.Phi = Phi
        self.weights = np.ones(n3)
        self.dtype = np.float64

    def forward(self, X):
        return np.ascontiguousarray(np.einsum("kj,abj->kab", self.Phi, X))

    def inverse(self, Xb):
        return np.einsum("kj,kab->abj", self.Phi, Xb)


def data_dependent(Y: np.ndarray, name: str = "UNI") -> Orthogonal:
    """Orthogonal transform from the left singular vectors of the mode-3
    unfolding of a pilot estimate Y (Song, Ng and Zhang, 2020)."""
    n3 = Y.shape[2]
    M = Y.reshape(-1, n3)  # (n1 n2) x n3
    # left singular vectors of the n3 x (n1 n2) unfolding = eigenvectors of its
    # n3 x n3 Gram matrix; never form the (n1 n2)-sized factor
    _, V = np.linalg.eigh(M.T @ M)
    return Orthogonal(V[:, ::-1].T, name=name)


def make(name: str, n3: int, pilot: np.ndarray | None = None) -> Transform:
    name = name.upper()
    if name in ("DFT", "FFT"):
        return DFT(n3)
    if name == "DCT":
        return DCT(n3)
    if name in ("UNI", "DATA"):
        if pilot is None:
            raise ValueError("a data-dependent transform needs a pilot tensor")
        return data_dependent(pilot)
    raise ValueError(f"unknown transform {name!r}")
