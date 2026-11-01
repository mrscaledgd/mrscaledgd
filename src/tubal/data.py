"""Test data: synthetic multi-rank tensors and loaders for the real datasets.

Real datasets are downloaded once into a data directory (default ./data, or
$TUBAL_DATA) by scripts/get_data.sh; nothing here downloads.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .transforms import Transform


def data_dir() -> Path:
    return Path(os.environ.get("TUBAL_DATA", "data"))


# --- synthetic -----------------------------------------------------------------

def synthetic(n1, n2, T: Transform, ranks, kappa, rng, spread="joint"):
    """Real tensor whose transformed slices have ranks `ranks` (length T.m).

    Singular values are linearly spaced in [1/kappa, 1] over all components of
    all slices jointly (`spread="joint"`, so kappa is the condition number of
    the block-diagonal matrix, as in Wu, 2026), and scaled so that ||X||_F^2 is
    n1 * n2 * n3 on average per entry 1. Conjugate-pair slices of the DFT get
    complex factors; real slices (DC, Nyquist, all DCT slices) real ones.
    """
    ranks = list(ranks)
    total = int(sum(ranks))
    sv = np.linspace(1.0, 1.0 / kappa, total) if total > 1 else np.ones(total)
    sv = sv[rng.permutation(total)]
    Xb = np.zeros((T.m, n1, n2), dtype=T.dtype)
    pos = 0
    for k, rk in enumerate(ranks):
        if rk == 0:
            continue
        cplx = np.iscomplexobj(Xb) and not (k == 0 or (T.n3 % 2 == 0 and k == T.m - 1))
        A = rng.standard_normal((n1, rk)) + (1j * rng.standard_normal((n1, rk)) if cplx else 0)
        B = rng.standard_normal((n2, rk)) + (1j * rng.standard_normal((n2, rk)) if cplx else 0)
        U, _ = np.linalg.qr(A)
        V, _ = np.linalg.qr(B)
        s = np.sort(sv[pos:pos + rk])[::-1]
        pos += rk
        Xb[k] = (U * s) @ np.conj(V).T
    X = T.inverse(Xb).real
    return X * np.sqrt(X.size) / np.linalg.norm(X)


def bernoulli_mask(shape, p, rng):
    return (rng.random(shape) < p).astype(float)


# --- real datasets ------------------------------------------------------------

def load(key: str) -> np.ndarray:
    """A real dataset in [0, 1] by key:

      video:<name>[:<frames>]   luma of data/video/<name>_qcif.y4m, first frames (50)
      cave:<scene>[:<factor>]   CAVE multispectral scene, block-averaged by factor (2)
      dcmall[:<size>[:<bands>]] HYDICE Washington DC Mall, top-left size x size crop
                                of the first `bands` bands (256 x 256 x 191)
    """
    kind, *rest = key.split(":")
    if kind == "video":
        frames = int(rest[1]) if len(rest) > 1 else 50
        return read_y4m(data_dir() / "video" / f"{rest[0]}_qcif.y4m", frames=frames)
    if kind == "cave":
        return load_cave(rest[0], int(rest[1]) if len(rest) > 1 else 2)
    if kind == "dcmall":
        size = int(rest[0]) if rest else 256
        bands = int(rest[1]) if len(rest) > 1 else 191
        return load_dcmall(size, bands)
    raise ValueError(f"unknown dataset {key!r}")


def _block_average(X, f):
    if f == 1:
        return X
    n1, n2 = (X.shape[0] // f) * f, (X.shape[1] // f) * f
    X = X[:n1, :n2]
    return X.reshape(n1 // f, f, n2 // f, f, -1).mean(axis=(1, 3))


def load_cave(scene: str, factor: int = 2) -> np.ndarray:
    from PIL import Image

    d = data_dir() / "cave" / f"{scene}_ms"
    files = sorted(p for p in d.rglob("*.png"))
    bands = [np.asarray(Image.open(p), dtype=float) for p in files]
    X = _block_average(np.stack(bands, axis=2), factor)
    return X / X.max()


def load_dcmall(size: int = 256, bands: int = 191) -> np.ndarray:
    X = np.load(data_dir() / "dcmall" / "dc.npy", mmap_mode="r")
    X = np.asarray(X[:size, :size, :bands], dtype=float)
    X = np.clip(X, 0.0, None)  # a few hundred corrupted entries are large negatives
    return X / X.max()


# --- video (YUV4MPEG2, luma only) ----------------------------------------------

def read_y4m(path, frames=None, start=0):
    """Luma frames of a 4:2:0 .y4m file as an (H, W, frames) array in [0, 1]."""
    with open(path, "rb") as f:
        header = f.readline().split()
        W = int(next(t for t in header if t.startswith(b"W"))[1:])
        H = int(next(t for t in header if t.startswith(b"H"))[1:])
        frame_bytes = W * H * 3 // 2
        out = []
        i = 0
        while True:
            line = f.readline()
            if not line:
                break
            buf = f.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            if i >= start:
                out.append(np.frombuffer(buf[: W * H], dtype=np.uint8).reshape(H, W))
            i += 1
            if frames is not None and len(out) == frames:
                break
    return np.stack(out, axis=2).astype(float) / 255.0
