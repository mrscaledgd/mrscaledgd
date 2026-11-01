#!/usr/bin/env python3
"""Build the paper's tables from the run summaries.

    python scripts/make_tables.py

Reads the runs where make_figures.py does (results/runs/, or $TUBAL_RUNS) and
writes LaTeX bodies to paper/tables/*.tex and the same numbers as CSV to
results/tables/*.csv. Rank-tuned baselines report their best rank on the grid
(an oracle choice that favours them); MR-ScaledGD runs with fixed defaults and
no tuned rank.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_figures import runs, tag  # noqa: E402

TEX = ROOT / "paper" / "tables"
CSV = ROOT / "results" / "tables"

COLS = [("tnn", "DFT", "TNN-DFT"), ("tnn", "DCT", "TNN-DCT"), ("tctf", "DFT", "TCTF$^\\dagger$"),
        ("scaledgd", "*", "ScaledGD$^\\dagger$"), ("apgd", "*", "APGD$^\\dagger$"),
        ("mr", "DFT", "MR-DFT"), ("mr", "DCT", "MR-DCT")]
NAMES = {"akiyo": "Akiyo", "carphone": "Carphone", "coastguard": "Coastguard", "foreman": "Foreman",
         "hall_monitor": "Hall", "news": "News", "balloons": "Balloons", "beads": "Beads",
         "feathers": "Feathers", "flowers": "Flowers", "superballs": "Superballs",
         "stuffed_toys": "Toys", "dcmall": "DC Mall"}


def collect(campaign):
    """best[(dataset, p)][(method, transform)] = metrics of the best run (by PSNR)."""
    best = defaultdict(dict)
    for c, m in runs(campaign):
        t = tag(c)
        if t not in ("tnn", "tctf", "scaledgd", "apgd", "mr"):
            continue
        key = (c["dataset"], c["p"])
        ps = m["psnr"] if (m["psnr"] is not None and not m["diverged"]) else -np.inf
        for tr in {c["transform"], "*"}:
            if tr == "*" and t not in ("scaledgd", "apgd"):
                continue
            if tr == "*" and c["transform"] not in ("DFT", "DCT"):
                continue
            k2 = (t, tr)
            cur = best[key].get(k2)
            if cur is None or ps > cur["_ps"]:
                best[key][k2] = dict(m, _ps=ps, _cfg=c)
    return best


def short(dataset):
    return NAMES.get(dataset.split(":")[1] if ":" in dataset else dataset, dataset) \
        if not dataset.startswith("dcmall") else "DC Mall"


def fmt(v, bold=False, nd=2):
    if v is None or not np.isfinite(v):
        return "--"
    s = f"{v:.{nd}f}"
    return f"\\textbf{{{s}}}" if bold else s


def psnr_table(best, datasets, rates, name, caption_rows):
    lines, rows_csv = [], []
    means = defaultdict(lambda: defaultdict(list))
    for ds in datasets:
        for k, p in enumerate(rates):
            cell = best.get((ds, p), {})
            vals = [cell.get((meth, tr), {}).get("_ps", np.nan) for meth, tr, _ in COLS]
            vals = [np.nan if v == -np.inf else v for v in vals]
            top = np.nanmax(vals) if np.any(np.isfinite(vals)) else np.nan
            label = short(ds) if k == 0 else ""
            lines.append(f"{label} & {p:g} & " + " & ".join(fmt(v, bold=np.isfinite(v) and v >= top - 1e-9)
                                                               for v in vals) + " \\\\")
            rows_csv.append([short(ds), p] + [None if not np.isfinite(v) else round(v, 3) for v in vals])
            for j, v in enumerate(vals):
                means[p][j].append(v)
        lines.append("\\addlinespace[2pt]")
    for p in rates:
        vals = [np.nanmean(means[p][j]) if means[p][j] else np.nan for j in range(len(COLS))]
        top = np.nanmax(vals)
        lines.append(f"Mean & {p:g} & " + " & ".join(fmt(v, bold=v >= top - 1e-9) for v in vals) + " \\\\")
        rows_csv.append(["Mean", p] + [round(v, 3) for v in vals])
    TEX.mkdir(parents=True, exist_ok=True)
    (TEX / f"{name}.tex").write_text("\n".join(lines) + "\n")
    CSV.mkdir(parents=True, exist_ok=True)
    with open(CSV / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "p"] + [c[2].replace("$^\\dagger$", "*") for c in COLS])
        w.writerows(rows_csv)
    print("wrote", TEX / f"{name}.tex")


def ssim_table(best, datasets, rates, name):
    """SSIM per case, as CSV only: the paper quotes the dataset means in the text."""
    rows_csv = []
    for ds in datasets:
        for p in rates:
            cell = best.get((ds, p), {})
            vals = [cell.get((meth, tr), {}).get("ssim") for meth, tr, _ in COLS]
            rows_csv.append([short(ds), p] + [None if v is None else round(v, 4) for v in vals])
    with open(CSV / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "p"] + [c[2].replace("$^\\dagger$", "*") for c in COLS])
        w.writerows(rows_csv)
    means = [np.nanmean([np.nan if r[2 + j] is None else r[2 + j] for r in rows_csv]) for j in range(len(COLS))]
    print(f"wrote {CSV / name}.csv; mean SSIM:", ", ".join(f"{c[2]} {m:.3f}" for c, m in zip(COLS, means)))


def timing_table():
    """Median time and iterations over the six videos at p = 0.2 (exclusive node)."""
    best = defaultdict(dict)
    for c, m in runs("timing"):
        t = tag(c)
        key = (t, c["transform"])
        ps = m["psnr"] if (m["psnr"] is not None and not m["diverged"]) else -np.inf
        cur = best[key].get(c["dataset"])
        if cur is None or ps > cur["_ps"]:
            best[key][c["dataset"]] = dict(m, _ps=ps)
    order = [("tnn", "DFT", "TNN-DFT"), ("tnn", "DCT", "TNN-DCT"), ("tctf", "DFT", "TCTF$^\\dagger$"),
             ("scaledgd", "DCT", "ScaledGD-DCT$^\\dagger$"), ("apgd", "DCT", "APGD-DCT$^\\dagger$"),
             ("mr", "DFT", "MR-DFT"), ("mr", "DCT", "MR-DCT")]
    lines, rows = [], []
    for meth, tr, lab in order:
        ms = list(best[(meth, tr)].values())
        if not ms:
            continue
        tt = [m["time"] for m in ms]
        it = [m["iters"] for m in ms]
        ps = [m["_ps"] for m in ms]
        per = [m["time"] / max(m["iters"], 1) for m in ms]
        lines.append(f"{lab} & {np.median(tt):.1f} & {np.median(it):.0f} & {1e3 * np.median(per):.0f} & "
                     f"{np.mean([p for p in ps if np.isfinite(p)]):.2f} \\\\")
        rows.append([lab, np.median(tt), np.median(it), np.median(per), np.mean(ps)])
    (TEX / "tab_time.tex").write_text("\n".join(lines) + "\n")
    with open(CSV / "tab_time.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "median_time_s", "median_iters", "median_s_per_iter", "mean_psnr"])
        w.writerows(rows)
    print("wrote", TEX / "tab_time.tex")


def ablation_table():
    res = defaultdict(dict)
    for c, m in runs("sensitivity"):
        v = c["name"].split("-v", 1)[1]
        res[v][c["dataset"]] = None if m["diverged"] else m["psnr"]
    sets = ["video:carphone", "video:akiyo", "cave:balloons", "cave:feathers"]
    rows = [("default", "defaults"), ("norealloc", "no reallocation"),
            ("realloc10", "reallocation every 10 iterations"), ("realloc-all", "reallocation until the end"),
            ("cap20-nofloor", "no floor, $s_{\\max}=20n_3$"), ("cap3-nofloor", "no floor, $s_{\\max}=3n_3$"),
            ("floor1.0", "$c_\\tau=1$"), ("floor1.5", "$c_\\tau=1.5$"),
            ("floor2.5", "$c_\\tau=2.5$"), ("floor3.0", "$c_\\tau=3$"),
            ("damp4.0", "$c_\\lambda=4$"), ("damp0.0", "no damping ($c_\\lambda=0$)"),
            ("f-floor1.5", "no damping, $c_\\tau=1.5$"), ("f-eta0.7", "no damping, $\\eta=0.7$"),
            ("f-cap20-nofloor", "no damping, no floor"),
            ("eta0.3", "$\\eta=0.3$"), ("eta0.7", "$\\eta=0.7$"), ("eta1.0", "$\\eta=1$")]
    lines, out = [], []
    for key, lab in rows:
        vals = [res[key].get(s) for s in sets]
        lines.append(f"{lab} & " + " & ".join("div." if v is None else f"{v:.2f}" for v in vals) + " \\\\")
        out.append([lab] + vals)
    (TEX / "tab_ablation.tex").write_text("\n".join(lines) + "\n")
    with open(CSV / "tab_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant"] + sets)
        w.writerows(out)
    print("wrote", TEX / "tab_ablation.tex")


if __name__ == "__main__":
    from tubal.experiments import CAVE, VIDEOS

    vb = collect("video")
    psnr_table(vb, [f"video:{v}" for v in VIDEOS], (0.1, 0.2, 0.3), "tab_video", None)
    ssim_table(vb, [f"video:{v}" for v in VIDEOS], (0.1, 0.2, 0.3), "tab_video_ssim")
    mb = collect("msi")
    hb = collect("hsi")
    mb.update(hb)
    psnr_table(mb, [f"cave:{c}" for c in CAVE] + ["dcmall:256:191"], (0.05, 0.1, 0.2), "tab_msi", None)
    ssim_table(mb, [f"cave:{c}" for c in CAVE] + ["dcmall:256:191"], (0.05, 0.1, 0.2), "tab_msi_ssim")
    timing_table()
    ablation_table()
