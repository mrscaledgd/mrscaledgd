"""One directory per run with three small JSON sidecars.

  config.json   what was run: the experiment, its parameters, the seed
  env.json      with what: library versions and BLAS thread counts
  metrics.json  what came out: final numbers and the recorded history
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def environment() -> dict:
    return {
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                   "MKL_NUM_THREADS")},
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def write_run(run_dir, config: dict, metrics: dict):
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(_jsonable(config), indent=1))
    (d / "env.json").write_text(json.dumps(_jsonable(environment()), indent=1))
    (d / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=1))


def done(run_dir) -> bool:
    """A run is complete once its metrics.json exists (resubmits skip it)."""
    return (Path(run_dir) / "metrics.json").exists()
