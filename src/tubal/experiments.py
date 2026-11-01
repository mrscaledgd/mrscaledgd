"""Experiment campaigns: each expands into run specs; run_one executes one spec.

A spec is a flat dict. Runs are named <campaign>/<name>, and write config.json,
env.json and metrics.json under out/runs/<campaign>/<name>/. A run whose
metrics.json exists is skipped, so a resubmitted campaign only does what is left.

Masks and synthetic ground truths are seeded from the spec, never from the
method, so every method of a comparison sees the same samples.
"""

from __future__ import annotations

import time
import traceback
import zlib
from pathlib import Path

import numpy as np

from . import baselines, data, factors, metrics, problems, runlog, solver, transforms
from .factors import herm

# ---------------------------------------------------------------- utilities ---


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def profile(name: str, r: int, m: int) -> list[int]:
    """Multi-rank profiles over the m stored slices (index 0 = lowest frequency)."""
    k = np.arange(m)
    if name == "uniform":
        out = np.full(m, r)
    elif name == "linear":  # r at the lowest frequency down to 1 at the highest
        out = np.maximum(1, np.round(r - (r - 1) * k / max(m - 1, 1)))
    elif name == "steep":   # halves every two slices, at least 1
        out = np.maximum(1, np.floor(r * 2.0 ** (-k / 2)))
    elif name == "lowpass":  # r on the lowest quarter, 0 elsewhere
        out = np.where(k < max(1, int(np.ceil(m / 4))), r, 0)
    else:
        raise ValueError(name)
    return [int(v) for v in out]


def _history(h, every=1):
    d = h.as_dict()
    if every > 1:
        for key in ("it", "time", "rel_err", "psnr", "loss", "ranks", "cond", "lam"):
            d[key] = d[key][::every]
    d["ranks"] = [sum(r) for r in d["ranks"]]  # total stored rank, not the vector
    if d.get("extra"):
        for key in d["extra"][0]:
            d[key] = [e[key] for e in d["extra"]][::max(every, 1)]
    d.pop("extra", None)
    return d


# --------------------------------------------------------------- one run ------


def build(spec):
    """Ground truth, transform and problem for a spec."""
    kind = spec["kind"]
    if spec.get("dataset"):
        Xs = data.load(spec["dataset"])
    else:
        Tg = transforms.make(spec["transform"], spec["n3"])
        ranks = profile(spec["profile"], spec["r"], Tg.m)
        rng = np.random.default_rng(seed_of("truth", spec["profile"], spec["r"], spec["kappa"],
                                            spec["n1"], spec["n3"], spec["transform"], spec["trial"]))
        Xs = data.synthetic(spec["n1"], spec["n2"], Tg, ranks, spec["kappa"], rng)
    n1, n2, n3 = Xs.shape
    T = transforms.make(spec["transform"], n3) if spec["transform"] not in ("UNI", "DATA") else None
    if kind == "tc":
        mrng = np.random.default_rng(seed_of("mask", spec.get("dataset") or spec["profile"],
                                             spec["p"], spec["trial"], n1, n3))
        mask = data.bernoulli_mask(Xs.shape, spec["p"], mrng)
        if T is None:  # data-dependent transform from the zero-filled, rescaled samples
            pilot = mask * Xs / spec["p"]
            T = transforms.make("UNI", n3, pilot=pilot)
        prob = problems.Completion(Xs, mask, T)
    elif kind == "sensing":
        arng = np.random.default_rng(seed_of("A", spec["m"], spec["trial"], n1, n3))
        A = arng.standard_normal((spec["m"], Xs.size)) / np.sqrt(spec["m"])
        prob = problems.Sensing(A, A @ Xs.ravel(), Xs.shape, T)
    elif kind == "fact":
        nrng = np.random.default_rng(seed_of("noise", spec["trial"]))
        Y0 = Xs + spec.get("noise", 0.0) * nrng.standard_normal(Xs.shape)
        prob = problems.Factorization(Xs, T, init=Y0)
    else:
        raise ValueError(kind)
    return Xs, T, prob


def spurious_monitor(Xs, T):
    """Callback measuring the part of the estimate outside the true slice subspaces.

    For each stored slice, with U_k and V_k spanning the column and row spaces of
    the true slice, the spurious block of the estimate is
    (I - U_k U_k^H) Xbar_k (I - V_k V_k^H); its energy (in original-domain units,
    relative to ||X_star||_F) is what the overparameterized directions carry.
    """
    Xb_true = T.forward(Xs)
    tol = 1e-9 * np.abs(Xb_true).max()
    bases = []
    for k in range(T.m):
        U, s, Vh = np.linalg.svd(Xb_true[k], full_matrices=False)
        r = int((s > tol * max(1, Xb_true.shape[1])).sum())
        bases.append((U[:, :r], herm(Vh[:r])))
    nrm = float(np.linalg.norm(Xs))

    def cb(t, X, F):
        Xb = T.forward(X)
        sp = 0.0
        for k, (U, V) in enumerate(bases):
            B = Xb[k] - U @ (herm(U) @ Xb[k])
            B = B - (B @ V) @ herm(V)
            sp += T.weights[k] * float(np.linalg.norm(B) ** 2)
        return {"spurious": float(np.sqrt(sp / T.ell) / nrm)}

    return cb


def budget_of(spec, T, prob):
    if "budget" in spec:
        return spec["budget"]
    if "rank_equiv" in spec:  # the budget of a tubal rank r: s = n3 * r
        return factors.tubal_budget(spec["rank_equiv"], T)
    if spec.get("oracle_budget"):
        ranks = profile(spec["profile"], spec["r"], T.m)
        return factors.budget_cost(ranks, T)
    return None


def run_one(spec, out_root="out/runs"):
    run_dir = Path(out_root) / spec["campaign"] / spec["name"]
    if runlog.done(run_dir):
        return "skip"
    t0 = time.perf_counter()
    try:
        Xs, T, prob = build(spec)
        peak = 1.0 if spec.get("dataset") else None
        common = dict(X_true=Xs, record_every=spec.get("record_every", 1), peak=peak)
        meth = spec["method"]
        ranks_final = None
        if meth == "tnn":
            X, h = baselines.tnn_complete(Xs, prob.mask, T, iters=spec.get("iters", 500), **common)
        elif meth == "tctf":
            X, ranks_final, h = baselines.tctf_complete(Xs, prob.mask, T, r0=spec["r0"],
                                                        iters=spec.get("iters", 500),
                                                        tol=spec.get("tol", 1e-5), **common)
        elif meth == "svp":
            X, F, h = baselines.svp_solve(prob, T, budget=budget_of(spec, T, prob),
                                          iters=spec.get("iters", 300), step=spec.get("step", 1.0),
                                          tol=spec.get("tol", 0.0), **common)
            ranks_final = F.ranks
        else:
            kw = dict(iters=spec.get("iters", 300), eta=spec.get("eta", 0.5),
                      damping=spec.get("damping", "none"), c_lam=spec.get("c_lam", 0.0),
                      realloc_every=spec.get("realloc_every", 0),
                      realloc_until=spec.get("realloc_until", 0), tol=spec.get("tol", 0.0),
                      c_floor=spec.get("c_floor", 0.0),
                      diagnostics=spec.get("diagnostics", False))
            budget = budget_of(spec, T, prob)
            if budget is None:
                kw["rank"] = spec["rank"]
            else:
                kw["budget"] = budget
            if spec.get("spurious"):
                kw["callback"] = spurious_monitor(Xs, T)
            X, F, h = solver.solve(prob, T, method=meth, **kw, **common)
            ranks_final = F.ranks
        out = {
            "rel_err": metrics.rel_err(X, Xs) if np.all(np.isfinite(X)) else None,
            "psnr": metrics.psnr(X, Xs, peak) if np.all(np.isfinite(X)) else None,
            "iters": h.it[-1] if h.it else 0,
            "time": h.time[-1] if h.time else None,
            "diverged": any(e[1] in ("diverged", "singular") for e in h.events),
            "events": [e for e in h.events if e[1] != "realloc"][:10],
            "n_realloc": sum(1 for e in h.events if e[1] == "realloc"),
            "ranks_final": ranks_final,
            "best_psnr": float(np.nanmax(h.psnr)) if h.psnr else None,
            "best_rel_err": float(np.nanmin(h.rel_err)) if h.rel_err else None,
            "history": _history(h, spec.get("store_every", 1)),
            "wall": time.perf_counter() - t0,
        }
        if spec.get("dataset") and out["psnr"] is not None:
            out["ssim"] = metrics.ssim(X, Xs)
        runlog.write_run(run_dir, spec, out)
        return "ok"
    except Exception:  # record the failure; a resubmit retries it
        (run_dir).mkdir(parents=True, exist_ok=True)
        (run_dir / "error.txt").write_text(traceback.format_exc())
        return "error"


# --------------------------------------------------------------- campaigns ----


def _name(**kv):
    return "-".join(f"{k}{v}" for k, v in kv.items())


# MR-ScaledGD: budgeted multi-rank factors and a budgeted SVP step every 20
# iterations during the first 200. MR_DAMPED adds leakage damping (c = 1) and is
# the method as published (run tag "mrscaledgd"): it matches MR at the default
# settings and does not diverge where MR does under aggressive ones
# (sensitivity campaign). MR (run tag "mr") is kept for that ablation.
MR = dict(method="scaledgd", realloc_every=20, realloc_until=200)
MR_DAMPED = dict(MR, damping="leak", c_lam=1.0)


def campaign_tc_synth():
    """Completion convergence for non-uniform and uniform multi-ranks, several kappa."""
    specs = []
    for tr in ("DFT", "DCT"):
        for prof in ("linear", "uniform"):
            for kappa in (1.0, 10.0, 100.0):
                for trial in range(5):
                    base = dict(campaign="tc_synth", kind="tc", n1=100, n2=100, n3=20, transform=tr,
                                profile=prof, r=10, kappa=kappa, p=0.3, trial=trial, iters=600,
                                record_every=1)
                    runs = [("gd", dict(method="gd", rank=10)),
                            ("scaledgd", dict(method="scaledgd", rank=10)),
                            ("scaledgd-damp", dict(method="scaledgd", rank=10, damping="leak", c_lam=1.0)),
                            ("apgd", dict(method="apgd", rank=10)),
                            ("mrscaledgd", dict(MR_DAMPED, oracle_budget=True)),
                            ("mr", dict(MR, oracle_budget=True)),
                            ("mr-oracle", dict(method="scaledgd", oracle_budget=True))]
                    for tag, kw in runs:
                        specs.append({**base, **kw,
                                      "name": _name(m=tag, T=tr, P=prof, k=kappa, t=trial)})
    return specs


def campaign_phase():
    """Success rate versus sampling rate for several multi-rank profiles."""
    specs = []
    for tr in ("DFT",):
        for prof in ("uniform", "linear", "steep", "lowpass"):
            for p in (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
                for trial in range(10):
                    base = dict(campaign="phase", kind="tc", n1=60, n2=60, n3=12, transform=tr,
                                profile=prof, r=6, kappa=5.0, p=p, trial=trial, iters=1000,
                                tol=1e-10, record_every=50)
                    for tag, kw in [("scaledgd", dict(method="scaledgd", rank=6)),
                                    ("apgd", dict(method="apgd", rank=6)),
                                    ("mrscaledgd", dict(MR_DAMPED, oracle_budget=True)),
                                    ("mr", dict(MR, oracle_budget=True)),
                                    ("tnn", dict(method="tnn", iters=500))]:
                        specs.append({**base, **kw,
                                      "name": _name(m=tag, T=tr, P=prof, p=p, t=trial)})
    return specs


def campaign_sensing():
    """Tensor sensing with Gaussian measurements: success versus m."""
    specs = []
    for prof in ("linear", "lowpass"):
        for m in (300, 400, 500, 600, 800, 1000, 1200, 1500, 2000):
            for trial in range(10):
                base = dict(campaign="sensing", kind="sensing", n1=16, n2=16, n3=8, transform="DFT",
                            profile=prof, r=4, kappa=5.0, m=m, trial=trial, iters=500,
                            tol=1e-10, record_every=25)
                for tag, kw in [("scaledgd", dict(method="scaledgd", rank=4)),
                                ("apgd", dict(method="apgd", rank=4)),
                                ("mrscaledgd", dict(MR_DAMPED, oracle_budget=True)),
                                ("mr", dict(MR, oracle_budget=True)),
                                ("svp", dict(method="svp", oracle_budget=True))]:
                    specs.append({**base, **kw, "name": _name(m=tag, P=prof, M=m, t=trial)})
    return specs


VIDEOS = ["akiyo", "carphone", "coastguard", "foreman", "hall_monitor", "news"]
CAVE = ["balloons", "beads", "feathers", "flowers", "superballs", "stuffed_toys"]
RGRID = (1, 2, 3, 5, 8)
# MR-ScaledGD on real data: no tuned rank. The budget is a generous cap (the
# budget of tubal rank 20) and the leakage floor decides what is fitted.
MR_REAL = dict(MR, c_floor=2.0, rank_equiv=20)
MR_REAL_DAMPED = dict(MR_DAMPED, c_floor=2.0, rank_equiv=20)


def _real_runs(campaign, dataset, p, trial, transforms_=("DFT", "DCT", "UNI"), iters=500):
    base = dict(campaign=campaign, kind="tc", dataset=dataset, p=p, trial=trial,
                iters=iters, tol=1e-5, record_every=10)
    dname = dataset.replace(":", "_")
    specs = []
    for tr in transforms_:
        b = {**base, "transform": tr}
        specs.append({**b, "method": "tnn", "tol": 1e-8,
                      "name": _name(d=dname, p=p, t=trial, m="tnn", T=tr)})
        specs.append({**b, **MR_REAL_DAMPED, "name": _name(d=dname, p=p, t=trial, m="mrscaledgd", T=tr)})
        specs.append({**b, **MR_REAL, "name": _name(d=dname, p=p, t=trial, m="mr", T=tr)})
        if tr == "UNI":
            continue
        for r in RGRID:
            specs.append({**b, "method": "scaledgd", "rank": r,
                          "name": _name(d=dname, p=p, t=trial, m="scaledgd", T=tr, r=r)})
            specs.append({**b, "method": "apgd", "rank": r,
                          "name": _name(d=dname, p=p, t=trial, m="apgd", T=tr, r=r)})
    for r0 in (3, 5, 10, 20):
        specs.append({**base, "transform": "DFT", "method": "tctf", "r0": r0,
                      "name": _name(d=dname, p=p, t=trial, m="tctf", T="DFT", r=r0)})
    return specs


def campaign_video():
    specs = []
    for v in VIDEOS:
        for p in (0.1, 0.2, 0.3):
            specs += _real_runs("video", f"video:{v}", p, 0)
    return specs


def campaign_msi():
    specs = []
    for c in CAVE:
        for p in (0.05, 0.1, 0.2):
            specs += _real_runs("msi", f"cave:{c}", p, 0)
    return specs


def campaign_hsi():
    specs = []
    for p in (0.05, 0.1, 0.2):
        specs += _real_runs("hsi", "dcmall:256:191", p, 0)
    return specs


def campaign_spurious():
    """Spurious-block energy along the iterations, completion with non-uniform multi-rank."""
    specs = []
    for tr in ("DFT", "DCT"):
        for trial in range(3):
            base = dict(campaign="spurious", kind="tc", n1=100, n2=100, n3=20, transform=tr,
                        profile="linear", r=10, kappa=10.0, p=0.3, trial=trial, iters=400,
                        record_every=2, spurious=True)
            for tag, kw in [("scaledgd", dict(method="scaledgd", rank=10)),
                            ("scaledgd-damp", dict(method="scaledgd", rank=10, damping="leak", c_lam=1.0)),
                            ("apgd", dict(method="apgd", rank=10)),
                            ("gd", dict(method="gd", rank=10)),
                            ("mrscaledgd", dict(MR_DAMPED, oracle_budget=True)),
                            ("mr", dict(MR, oracle_budget=True))]:
                specs.append({**base, **kw, "name": _name(m=tag, T=tr, t=trial)})
    return specs


def campaign_sensitivity():
    """MR-ScaledGD design choices on two videos and two MSI scenes at one rate."""
    specs = []
    for ds, p in (("video:carphone", 0.2), ("video:akiyo", 0.2), ("cave:balloons", 0.1),
                  ("cave:feathers", 0.1)):
        base = dict(campaign="sensitivity", kind="tc", dataset=ds, p=p, trial=0, iters=500,
                    tol=1e-5, record_every=10, transform="DCT")
        dname = ds.replace(":", "_")
        variants = {"default": {}}
        for r in (1, 2, 3, 5, 8, 12, 20):  # budget, with and without the floor
            variants[f"cap{r}"] = dict(rank_equiv=r)
            variants[f"cap{r}-nofloor"] = dict(rank_equiv=r, c_floor=0.0)
        for cf in (1.0, 1.5, 2.5, 3.0):
            variants[f"floor{cf}"] = dict(c_floor=cf)
        for cl in (0.0, 0.5, 2.0, 4.0):
            variants[f"damp{cl}"] = dict(c_lam=cl, damping="leak" if cl else "none")
        variants["norealloc"] = dict(realloc_every=0, realloc_until=0)
        variants["realloc10"] = dict(realloc_every=10, realloc_until=200)
        variants["realloc-all"] = dict(realloc_every=20, realloc_until=500)
        for eta in (0.3, 0.7, 1.0):
            variants[f"eta{eta}"] = dict(eta=eta)
        variants["damp1.0"] = dict(c_lam=1.0, damping="leak")  # undamped base only: else the default
        for tag, kw in variants.items():
            if tag != "damp1.0":
                specs.append({**base, **MR_REAL_DAMPED, **kw, "name": _name(d=dname, p=p, v=tag)})
            specs.append({**base, **MR_REAL, **kw, "name": _name(d=dname, p=p, v="f-" + tag)})
    return specs


def campaign_timing():
    """Run times on an otherwise idle node: every video at p = 0.2, few workers."""
    specs = []
    for v in VIDEOS:
        for tr in ("DFT", "DCT"):
            b = dict(campaign="timing", kind="tc", dataset=f"video:{v}", p=0.2, trial=0,
                     iters=500, tol=1e-5, record_every=10, transform=tr)
            dn = f"video_{v}"
            specs.append({**b, "method": "tnn", "tol": 1e-8, "name": _name(d=dn, m="tnn", T=tr)})
            specs.append({**b, **MR_REAL, "name": _name(d=dn, m="mr", T=tr)})
            specs.append({**b, **MR_REAL_DAMPED, "name": _name(d=dn, m="mrscaledgd", T=tr)})
            for r in RGRID:
                specs.append({**b, "method": "scaledgd", "rank": r, "name": _name(d=dn, m="scaledgd", T=tr, r=r)})
                specs.append({**b, "method": "apgd", "rank": r, "name": _name(d=dn, m="apgd", T=tr, r=r)})
        for r0 in (3, 5, 10, 20):
            specs.append({**b, "transform": "DFT", "method": "tctf", "r0": r0,
                          "name": _name(d=dn, m="tctf", T="DFT", r=r0)})
    return specs


CAMPAIGNS = {
    "timing": campaign_timing,
    "spurious": campaign_spurious,
    "tc_synth": campaign_tc_synth,
    "phase": campaign_phase,
    "sensing": campaign_sensing,
    "video": campaign_video,
    "msi": campaign_msi,
    "hsi": campaign_hsi,
    "sensitivity": campaign_sensitivity,
}
