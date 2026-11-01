#!/usr/bin/env python3
"""Visual comparison figure: re-runs the selected configurations (about ten minutes).

Rank-tuned baselines use their best rank from the campaign summaries (the same
choice as the tables); MR-ScaledGD uses its defaults. Needs the data of
scripts/get_data.sh; writes paper/figures/fig_visual.pdf.

    python scripts/make_visuals.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import figstyle as fs  # noqa: E402
from figstyle import plt  # noqa: E402
from make_figures import OUT, WIDTH, save  # noqa: E402
from make_tables import collect  # noqa: E402

from tubal import experiments, metrics  # noqa: E402

CASES = [("video", "video:news", 0.1, 25, "News, frame 26"),
         ("msi", "cave:balloons", 0.1, 20, "Balloons, band 21 (600 nm)")]
SHOW = [("tnn", "DCT", "TNN-DCT"), ("tctf", "DFT", "TCTF"), ("scaledgd", "*", "ScaledGD"),
        ("apgd", "*", "APGD"), ("mr", "DCT", "MR-ScaledGD")]


def reconstruct(cfg):
    spec = dict(cfg)
    Xs, T, prob = experiments.build(spec)
    from tubal import baselines, solver
    from tubal.experiments import budget_of
    meth = spec["method"]
    if meth == "tnn":
        X, _ = baselines.tnn_complete(Xs, prob.mask, T, iters=spec.get("iters", 500))
    elif meth == "tctf":
        X, _, _ = baselines.tctf_complete(Xs, prob.mask, T, r0=spec["r0"], iters=spec.get("iters", 500),
                                          tol=spec.get("tol", 1e-5))
    else:
        kw = dict(iters=spec.get("iters", 500), eta=spec.get("eta", 0.5),
                  damping=spec.get("damping", "none"), c_lam=spec.get("c_lam", 0.0),
                  realloc_every=spec.get("realloc_every", 0), realloc_until=spec.get("realloc_until", 0),
                  tol=spec.get("tol", 0.0), c_floor=spec.get("c_floor", 0.0))
        b = budget_of(spec, T, prob)
        if b is None:
            kw["rank"] = spec["rank"]
        else:
            kw["budget"] = b
        X, _, _ = solver.solve(prob, T, method=meth, **kw)
    return Xs, prob.mask, X


def main():
    fs.setup()
    fig, axes = plt.subplots(len(CASES), len(SHOW) + 2, figsize=(WIDTH, 1.2 * len(CASES) + 0.25))
    for i, (camp, ds, p, frame, title) in enumerate(CASES):
        best = collect(camp)[(ds, p)]
        imgs = []
        Xs = mask = None
        for meth, tr, lab in SHOW:
            cfg = best[(meth, tr)]["_cfg"]
            Xs, mask, X = reconstruct(cfg)
            imgs.append((lab, X, metrics.psnr(X, Xs, 1.0)))
            print(ds, lab, cfg.get("rank", cfg.get("r0", "")), f"{imgs[-1][2]:.2f}", flush=True)
        panels = [("original", Xs, None), (f"observed ({int(100 * p)}%)", mask * Xs, None)] + imgs
        for j, (lab, X, ps) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(np.clip(X[:, :, frame], 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(lab if ps is None else f"{lab}\n{ps:.2f} dB", fontsize=7)
        axes[i, 0].set_ylabel(title, fontsize=7)
    fig.tight_layout(w_pad=0.2, h_pad=0.4)
    save(fig, "fig_visual")


if __name__ == "__main__":
    main()
