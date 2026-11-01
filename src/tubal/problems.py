"""Estimation problems seen through a gradient oracle.

Each problem exposes, for a current estimate X (a real n1 x n2 x n3 tensor):

  grad(X)    G, the gradient of the loss with respect to X; the factored
             methods use its transformed slices Gbar_k R_k and Gbar_k^H L_k
  pilot()    the tensor whose spectrum initializes the factors (spectral init)
  loss(X)    the loss value
  leak(X, G) predicted spectral norm, per transformed slice, of the part of G
             that is not X - X_star: the sampling (or measurement) noise that
             every slice receives whatever its own content. It sets the
             damping of the preconditioners.

The full gradient step X - G is the singular value projection (SVP) step used
to reallocate ranks.
"""

from __future__ import annotations

import numpy as np

from .transforms import Transform


class Completion:
    """Tensor completion: f(X) = (1/2p) ||P_Omega(X - Y)||_F^2, Bernoulli(p) mask."""

    def __init__(self, Y: np.ndarray, mask: np.ndarray, T: Transform):
        self.mask = mask.astype(float)
        self.p = float(self.mask.mean())
        self.Ym = self.mask * Y
        self.T = T
        self.shape = Y.shape
        self.n_obs = float(self.mask.sum())

    def pilot(self):
        return self.Ym / self.p

    def grad(self, X):
        return self.mask * X / self.p - self.Ym / self.p

    def loss(self, X):
        return 0.5 * float(np.linalg.norm(self.mask * X - self.Ym) ** 2) / self.p

    def residual_rms(self, X):
        return float(np.linalg.norm(self.mask * X - self.Ym) / np.sqrt(self.n_obs))

    def leak(self, X, G=None):
        """sqrt(ell (1-p)/p) * rms(E) * (sqrt(n1) + sqrt(n2)).

        ((1/p) P_Omega - I)(E) has independent entries of variance
        (1-p)/p * E_ijk^2. A transformed slice combines n3 of them with weights of
        total energy ell (a row of Phi), so for a residual of roughly uniform
        magnitude its spectral norm concentrates at the value above; rms(E) is
        estimated on the observed entries.
        """
        n1, n2, _ = self.shape
        p = self.p
        return np.sqrt(self.T.ell * (1 - p) / p) * self.residual_rms(X) * (np.sqrt(n1) + np.sqrt(n2))


class Sensing:
    """Tensor sensing: f(X) = (1/2) ||A vec(X) - y||^2 with a dense m x N matrix A."""

    def __init__(self, A: np.ndarray, y: np.ndarray, shape, T: Transform):
        self.A, self.y, self.shape, self.T = A, y, tuple(shape), T
        self.m = A.shape[0]

    def pilot(self):
        return (self.A.T @ self.y).reshape(self.shape)

    def grad(self, X):
        return (self.A.T @ (self.A @ X.ravel() - self.y)).reshape(self.shape)

    def loss(self, X):
        return 0.5 * float(np.linalg.norm(self.A @ X.ravel() - self.y) ** 2)

    def leak(self, X, G=None):
        """For Gaussian A with N(0, 1/m) entries, (A*A - I)(E) is close to a Gaussian
        tensor with entry variance ||E||_F^2 / m, and ||E||_F ~ ||A vec(E)||_2."""
        n1, n2, _ = self.shape
        r = np.linalg.norm(self.A @ X.ravel() - self.y)
        return np.sqrt(self.T.ell / self.m) * r * (np.sqrt(n1) + np.sqrt(n2))


class Factorization:
    """Fully observed problem f(X) = (1/2) ||X - Y||_F^2: the slices decouple.

    `init` is the tensor the spectral initialization starts from (Y itself by
    default; a perturbed copy puts the start inside the basin rather than at
    the exact solution, where the tubal-rank Gram matrices would be singular).
    """

    def __init__(self, Y: np.ndarray, T: Transform, init: np.ndarray | None = None):
        self.Y, self.T, self.shape = Y, T, Y.shape
        self.init = Y if init is None else init

    def pilot(self):
        return self.init

    def grad(self, X):
        return X - self.Y

    def loss(self, X):
        return 0.5 * float(np.linalg.norm(X - self.Y) ** 2)

    def leak(self, X, G=None):
        return 0.0
