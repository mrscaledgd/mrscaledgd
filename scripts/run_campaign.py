#!/usr/bin/env python3
"""Run a campaign (src/tubal/experiments.py) in a pool of single-threaded workers.

    python scripts/run_campaign.py <campaign> [--workers N] [--shard i/n] [--limit K]
                                   [--filter SUBSTR] [--out out/runs] [--list]

Each worker pins its BLAS and FFT libraries to one thread, so N workers use N
cores. Completed runs (metrics.json present) are skipped, so a resubmitted job
only does what is left. --shard i/n takes every n-th spec starting at i, so n
jobs can split one campaign. On a Slurm node, --workers defaults to
$SLURM_CPUS_PER_TASK.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tubal import experiments, runlog  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", choices=sorted(experiments.CAMPAIGNS))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--filter", default="")
    ap.add_argument("--out", default="out/runs")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    specs = experiments.CAMPAIGNS[a.campaign]()
    names = [s["name"] for s in specs]
    assert len(set(names)) == len(names), "run names must be unique within a campaign"
    i, n = map(int, a.shard.split("/"))
    specs = [s for j, s in enumerate(specs) if j % n == i and a.filter in s["name"]]
    todo = [s for s in specs if not runlog.done(Path(a.out) / s["campaign"] / s["name"])]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{a.campaign}: {len(specs)} specs in shard {a.shard}, {len(todo)} to run, "
          f"{a.workers} workers", flush=True)
    if a.list:
        for s in todo:
            print(s["name"])
        return

    t0 = time.time()
    status = Counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(experiments.run_one, s, a.out): s["name"] for s in todo}
        for k, f in enumerate(as_completed(futs), 1):
            st = f.result()
            status[st] += 1
            if st == "error" or k % max(1, len(todo) // 20) == 0:
                print(f"[{time.time() - t0:7.0f}s] {k}/{len(todo)} {dict(status)} last={futs[f]} ({st})",
                      flush=True)
    print(f"done in {time.time() - t0:.0f}s: {dict(status)}", flush=True)


if __name__ == "__main__":
    main()
