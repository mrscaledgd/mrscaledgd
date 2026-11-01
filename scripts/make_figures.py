#!/usr/bin/env python3
"""Build the paper's figures from the run summaries (results/runs/*/*/metrics.json).

    python scripts/make_figures.py [names...]      default: all

Figures are written to paper/figures/ as PDF. The runs are read from
results/runs/, or from $TUBAL_RUNS (e.g. out/runs after a rerun).
fig_ceiling recomputes truncations from the data (no runs needed).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import figstyle as fs  # noqa: E402
from figstyle import plt  # noqa: E402

from tubal import data, factors, transforms  # noqa: E402

OUT = ROOT / "paper" / "figures"
RUNS = Path(os.environ.get("TUBAL_RUNS", ROOT / "results" / "runs"))
WIDTH = 6.3  # inches, the text width of the paper


def runs(campaign):
    for f in sorted(glob.glob(str(RUNS / campaign / "*" / "metrics.json"))):
        c = json.loads(Path(f).with_name("config.json").read_text())
        m = json.loads(Path(f).read_text())
        yield c, m


def tag(c):
    """Method tag from a run's configuration (not its name)."""
    meth = c["method"]
    if meth != "scaledgd":
        return meth
    damped = c.get("damping", "none") not in (None, "none") and c.get("c_lam", 0)
    budgeted = c.get("oracle_budget") or "rank_equiv" in c or "budget" in c
    if c.get("realloc_every"):  # MR-ScaledGD as published is the damped variant
        return "mr" if damped else "mr-nodamp"
    if budgeted:
        return "mr-oracle"
    return "scaledgd-damp" if damped else "scaledgd"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None})  # same input, same bytes
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# ------------------------------------------------------------------ ceiling ---

CEIL_SETS = [("video:akiyo", "Akiyo (video)"), ("video:coastguard", "Coastguard (video)"),
             ("cave:balloons", "Balloons (MSI)"), ("dcmall:256:191", "DC Mall (HSI)")]
CEIL_R = [1, 2, 3, 5, 8, 12, 20, 30]


def ceiling_curves(key, tname="DCT"):
    X = data.load(key)
    T = transforms.make(tname, X.shape[2])
    s = np.linalg.svd(T.forward(X), compute_uv=False)  # (m, q)
    N, peak = X.size, 1.0
    total = float(np.dot(T.weights, (s**2).sum(1)))
    def psnr_of(ranks):
        kept = sum(T.weights[k] * float((s[k, :rk] ** 2).sum()) for k, rk in enumerate(ranks))
        mse = (total - kept) / T.ell / N
        return 10 * np.log10(peak**2 / max(mse, 1e-300))
    tub, mul, alloc = [], [], {}
    for r in CEIL_R:
        tub.append(psnr_of([min(r, s.shape[1])] * T.m))
        ranks = factors.allocate(s, T, factors.tubal_budget(r, T))
        mul.append(psnr_of(ranks))
        alloc[r] = ranks
    return np.array(tub), np.array(mul), alloc, T


def fig_ceiling():
    fs.setup()
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 3.0), gridspec_kw=dict(height_ratios=[1.35, 1]))
    for j, (key, title) in enumerate(CEIL_SETS):
        tub, mul, alloc, T = ceiling_curves(key)
        ax = axes[0, j]
        fs.line(ax, CEIL_R, tub, "tubal")
        fs.line(ax, CEIL_R, mul, "multi")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 12, 30])
        ax.set_xticklabels(["1", "2", "5", "12", "30"])
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel(r"budget $s/n_3$ (tubal rank)")
        if j == 0:
            ax.set_ylabel("PSNR of best approx. (dB)")
        fs.finish(ax)
        ax2 = axes[1, j]
        ranks = alloc[5]
        ax2.bar(np.arange(T.m), ranks, width=0.85, color=fs.BLUE, linewidth=0)
        ax2.axhline(5, color=fs.ORANGE, lw=1.0, ls="-")
        ax2.set_xlabel("transformed slice $k$ (DCT)")
        if j == 0:
            ax2.set_ylabel(r"rank $r_k$ at $s = 5n_3$")
        ax2.set_ylim(0, max(ranks) * 1.08)
        ax2.text(0.97, 0.9, f"$r_1 = {ranks[0]}$", transform=ax2.transAxes, ha="right", va="top", fontsize=7)
        fs.finish(ax2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    save(fig, "fig_ceiling")


# ---------------------------------------------------------------- mechanism ---

def fig_mechanism():
    fs.setup()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.2))
    seen = set()
    for c, m in sorted(runs("spurious"), key=lambda cm: cm[0]["name"]):
        if c["transform"] != "DFT":
            continue
        t = tag(c)
        if t not in ("gd", "scaledgd", "apgd", "mr"):
            continue
        h = m["history"]
        it = np.array(h["it"])
        err = np.array([np.nan if v is None else v for v in h["rel_err"]], float)
        sp = np.array([np.nan if v is None else v for v in h["spurious"]], float)
        lab = None if t not in seen else "_"
        seen.add(t)
        for ax, y in ((axes[0], err), (axes[1], sp)):
            fs.line(ax, it, np.clip(y, 1e-17, 1e3), t, label=lab, markevery=40, lw=1.0)
            if m["diverged"]:
                ax.plot(it[-1], min(y[-1], 1e3) if np.isfinite(y[-1]) else 1e3, "x", color=fs.ORANGE, ms=6)
    for ax, ylab in ((axes[0], r"$\|\mathcal{X}_t-\mathcal{X}_\star\|_F/\|\mathcal{X}_\star\|_F$"),
                     (axes[1], "spurious-block energy (relative)")):
        ax.set_yscale("log")
        ax.set_ylim(1e-17, 1e3)
        ax.set_xlabel("iteration $t$")
        ax.set_ylabel(ylab)
        fs.finish(ax)
    axes[0].legend(loc="lower left", ncol=2)
    axes[1].text(0.98, 0.95, r"$\times$: diverged", transform=axes[1].transAxes, ha="right", va="top", fontsize=7)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_mechanism")


# -------------------------------------------------------------------- kappa ---

def fig_kappa(tname="DFT"):
    fs.setup()
    groups = defaultdict(list)
    for c, m in runs("tc_synth"):
        if c["transform"] != tname:
            continue
        groups[(c["profile"], c["kappa"], tag(c))].append(m)
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.6), sharex=True, sharey=True)
    for i, prof in enumerate(("linear", "uniform")):
        for j, kappa in enumerate((1.0, 10.0, 100.0)):
            ax = axes[i, j]
            for t in ("gd", "apgd", "scaledgd", "mr-oracle", "mr"):
                ms = groups.get((prof, kappa, t), [])
                if not ms:
                    continue
                L = max(len(m["history"]["rel_err"]) for m in ms)
                Y = np.full((len(ms), L), np.inf)
                for a, m in enumerate(ms):
                    e = np.array([np.inf if v is None else v for v in m["history"]["rel_err"]], float)
                    Y[a, :len(e)] = e
                med = np.median(Y, axis=0)
                med[~np.isfinite(med)] = np.nan
                it = np.arange(L)
                fs.line(ax, it, np.clip(med, 1e-17, 1e3), t, markevery=100, lw=1.1,
                        label=None if (i, j) == (0, 0) else "_")
                if any(m["diverged"] for m in ms) and np.isnan(med).any():
                    last = np.where(np.isfinite(med))[0]
                    if len(last):
                        ax.plot(last[-1], min(med[last[-1]], 1e3), "x", color=fs.STYLE[t]["color"], ms=6)
            ax.set_yscale("log")
            ax.set_ylim(1e-17, 1e2)
            if i == 0:
                ax.set_title(f"$\\kappa = {kappa:g}$")
            if i == 1:
                ax.set_xlabel("iteration $t$")
            if j == 0:
                ax.set_ylabel(("non-uniform multi-rank" if prof == "linear" else "uniform multi-rank")
                              + "\nrelative error")
            fs.finish(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(h_pad=0.8, w_pad=0.6, rect=(0, 0, 1, 0.93))
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
    save(fig, f"fig_kappa_{tname}")


# -------------------------------------------------------------------- phase ---

PROFILE_TITLE = {"uniform": "uniform", "linear": "linear decay", "steep": "steep decay",
                 "lowpass": "low-pass"}


def success_curve(campaign, xkey, filt):
    acc = defaultdict(list)
    for c, m in runs(campaign):
        if not filt(c):
            continue
        ok = (m["rel_err"] is not None) and (m["rel_err"] < 1e-4) and not m["diverged"]
        acc[(tag(c), c[xkey])].append(ok)
    return acc


def fig_phase():
    fs.setup()
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 3.5))
    for j, prof in enumerate(("uniform", "linear", "steep", "lowpass")):
        acc = success_curve("phase", "p", lambda c, prof=prof: c["profile"] == prof)
        ax = axes[0, j]
        for t in ("scaledgd", "apgd", "tnn", "mr"):
            xs = sorted({x for (tt, x) in acc if tt == t})
            ys = [np.mean(acc[(t, x)]) for x in xs]
            fs.line(ax, xs, ys, t, label=None if j == 0 else "_", lw=1.1)
        ax.set_title(f"completion, {PROFILE_TITLE[prof]}")
        ax.set_xlabel("sampling rate $p$")
        ax.set_ylim(-0.05, 1.05)
        if j == 0:
            ax.set_ylabel("success rate")
        fs.finish(ax)
    for j, prof in enumerate(("linear", "lowpass")):
        acc = success_curve("sensing", "m", lambda c, prof=prof: c["profile"] == prof)
        ax = axes[1, j]
        for t in ("scaledgd", "apgd", "svp", "mr"):
            xs = sorted({x for (tt, x) in acc if tt == t})
            ys = [np.mean(acc[(t, x)]) for x in xs]
            fs.line(ax, xs, ys, t, label=None if j == 0 else "_", lw=1.1)
        ax.set_title(f"sensing, {PROFILE_TITLE[prof]}")
        ax.set_xlabel("measurements $m$")
        ax.set_ylim(-0.05, 1.05)
        if j == 0:
            ax.set_ylabel("success rate")
        fs.finish(ax)
    # profiles of the ground truths, for reference
    ax = axes[1, 2]
    from tubal.experiments import profile
    for prof, ls in (("uniform", "-"), ("linear", "--"), ("steep", "-."), ("lowpass", ":")):
        ax.step(np.arange(7), profile(prof, 6, 7), where="mid", color=fs.INK2, ls=ls, lw=1.1,
                label=PROFILE_TITLE[prof])
    ax.set_title("multi-rank profiles (completion)")
    ax.set_xlabel("stored DFT slice $k$")
    ax.set_ylabel("$r_k$")
    ax.set_ylim(0, 10.5)
    ax.set_yticks([0, 2, 4, 6])
    ax.legend(loc="upper center", ncol=2, fontsize=6, handlelength=2.2, columnspacing=0.8)
    fs.finish(ax)
    axes[1, 3].axis("off")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    hs, ls_ = list(h1), list(l1)
    for h, l in zip(h2, l2):
        if l not in ls_:
            hs.append(h)
            ls_.append(l)
    axes[1, 3].legend(hs, ls_, loc="center")
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    save(fig, "fig_phase")


# -------------------------------------------------------------- sensitivity ---

SENS_SETS = [("video:carphone", "Carphone, $p=0.2$"), ("video:akiyo", "Akiyo, $p=0.2$"),
             ("cave:balloons", "Balloons, $p=0.1$"), ("cave:feathers", "Feathers, $p=0.1$")]


def tnn_reference(dataset, p, tname="DCT"):
    camp = "video" if dataset.startswith("video") else "msi"
    for c, m in runs(camp):
        if c["dataset"] == dataset and c["p"] == p and c["method"] == "tnn" and c["transform"] == tname:
            return m["psnr"]
    return None


def fig_sensitivity():
    fs.setup()
    res = defaultdict(dict)
    for c, m in runs("sensitivity"):
        v = c["name"].split("-v", 1)[1]
        if v.startswith("f-"):  # the undamped ablation
            continue
        res[c["dataset"]][v] = None if m["diverged"] else m["psnr"]
    caps = [1, 2, 3, 5, 8, 12, 20]
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 1.9), sharex=True)
    for j, (ds, title) in enumerate(SENS_SETS):
        ax = axes[j]
        y1 = [res[ds].get(f"cap{r}") for r in caps]
        y0 = [res[ds].get(f"cap{r}-nofloor") for r in caps]
        y1 = [np.nan if v is None else v for v in y1]
        y0 = [np.nan if v is None else v for v in y0]
        fs.line(ax, caps, y1, "mr", label="with leakage floor" if j == 0 else "_")
        fs.line(ax, caps, y0, "mr", color=fs.INK3, ls="--", marker="o",
                label="without floor" if j == 0 else "_")
        ref = tnn_reference(ds, 0.2 if ds.startswith("video") else 0.1)
        if ref is not None:
            ax.axhline(ref, color=fs.INK2, ls=":", lw=1.0, label="TNN-DCT" if j == 0 else "_")
        ax.set_xscale("log")
        ax.set_xticks(caps)
        ax.set_xticklabels([str(r) for r in caps])
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel(r"budget cap $s_{\max}/n_3$")
        if j == 0:
            ax.set_ylabel("PSNR (dB)")
        fs.finish(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig_sensitivity")


FIGS = {"ceiling": fig_ceiling, "mechanism": fig_mechanism, "kappa": fig_kappa, "phase": fig_phase,
        "sensitivity": fig_sensitivity}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
