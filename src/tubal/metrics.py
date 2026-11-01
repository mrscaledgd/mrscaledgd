"""Reconstruction metrics: relative error, PSNR, and mean SSIM over frontal slices."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def rel_err(X, Xs):
    return float(np.linalg.norm(X - Xs) / np.linalg.norm(Xs))


def psnr(X, Xs, peak=None):
    """10 log10(peak^2 / MSE) over all entries; peak defaults to max |Xs|."""
    peak = float(np.max(np.abs(Xs))) if peak is None else float(peak)
    mse = float(np.mean((X - Xs) ** 2))
    if not np.isfinite(mse):
        return float("nan")
    return float("inf") if mse == 0 else float(10 * np.log10(peak**2 / mse))


def ssim(X, Xs, data_range=1.0):
    """Mean SSIM (Wang et al., 2004) over the frontal slices X[:, :, k].

    11 x 11 Gaussian window with sigma 1.5, K1 = 0.01, K2 = 0.03; X is clipped
    to the data range first, as an image would be before display.
    """
    X = np.clip(X, 0.0, data_range)
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    vals = []
    for k in range(Xs.shape[2]):
        a, b = X[:, :, k], Xs[:, :, k]
        f = lambda z: gaussian_filter(z, 1.5, truncate=3.5)
        mu_a, mu_b = f(a), f(b)
        saa = f(a * a) - mu_a**2
        sbb = f(b * b) - mu_b**2
        sab = f(a * b) - mu_a * mu_b
        num = (2 * mu_a * mu_b + C1) * (2 * sab + C2)
        den = (mu_a**2 + mu_b**2 + C1) * (saa + sbb + C2)
        vals.append(float(np.mean(num / den)))
    return float(np.mean(vals))
