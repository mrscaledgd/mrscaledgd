# MR-ScaledGD

Multi-rank scaled gradient descent for low-rank tensor completion and sensing
under the transformed tensor singular value decomposition (t-SVD). This
repository holds the method and its baselines, the 5059 runs behind every
table and figure of the paper, and the scripts that rebuild them.

## The method

Under the transformed t-SVD, an n1 × n2 × n3 tensor is low-rank when its frontal
slices, after a transform along the third mode (DFT, DCT or a data-dependent
orthogonal transform), are low-rank matrices. Factored methods such as ScaledGD
parameterize the estimate by a tubal rank r, the same rank in every transformed
slice. Real tensors concentrate their transformed spectra in a few slices: at
an equal parameter count, keeping the largest transformed singular values
across slices approximates the test videos and spectral images 2.6 to 21.2 dB
better than tubal-rank truncation. The tubal-rank factorization is therefore
overparameterized in most slices. In completion, every slice also receives
sampling noise set by the global residual, and a ScaledGD step applies it to
the full factor ranges, however small the redundant columns are: ScaledGD
diverges, and a damped alternating variant (APGD) stagnates.

MR-ScaledGD factorizes each transformed slice at its own rank and takes
ScaledGD steps on each slice's factors. A global budget s sets the ranks:

- **Budgeted allocation.** The initial factors keep the s largest transformed
  singular values of the rescaled, zero-filled observations, pooled across
  slices (a transformed Eckart–Young projection).
- **Reallocation.** Every 20 iterations during the first 200, a budgeted
  singular value projection step replaces the scaled step and can move rank
  between slices.
- **Leakage floor and damping.** Sampling leaks the residual into every slice
  at a level ν that can be predicted. The allocation keeps only singular values
  above 2ν, which turns s into a cap, and the scaled step is damped by λ = ν,
  which vanishes with the residual.

For tensor sensing, MR-ScaledGD without the floor and the damping converges
linearly at a rate free of the condition number under a restricted isometry
property over multi-ranks. In simulated completion, the tubal-rank methods
never recover tensors with a non-uniform multi-rank, whereas MR-ScaledGD needs
fewer samples than tensor nuclear norm (TNN) minimization. On six videos and
seven spectral images, it is 8 to 13 dB more accurate than the other factored
t-SVD methods and, on the videos, six times faster than TNN minimization, which
remains 1.3 to 2.1 dB more accurate.

## Installation

```bash
git clone https://github.com/mrscaledgd/mrscaledgd.git
cd mrscaledgd
uv sync --extra dev               # Python 3.12 and the versions locked in uv.lock
.venv/bin/python -m pytest        # 15 checks, about a second
```

Without uv: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`
(Python 3.10 to 3.13; NumPy, SciPy, Matplotlib and Pillow). The shipped runs
used Python 3.12.13, NumPy 2.5.4 and SciPy 1.18.1, as each run's `env.json`
records. Run the commands below from the repository root.

## Usage

Completion of a 100 × 100 × 20 tensor whose 11 stored DFT slices have ranks
decreasing from 10 to 1, from 30% of its entries:

```python
import numpy as np
from tubal import data, experiments, factors, problems, solver, transforms

rng = np.random.default_rng(0)
T = transforms.make("DFT", 20)                  # stores 11 of the 20 Fourier slices
ranks = experiments.profile("linear", 10, T.m)  # non-uniform multi-rank, 10 down to 1
X = data.synthetic(100, 100, T, ranks, kappa=10.0, rng=rng)
mask = data.bernoulli_mask(X.shape, 0.3, rng)   # observe 30% of the entries
prob = problems.Completion(X * mask, mask, T)

# MR-ScaledGD, given the budget of the true multi-rank
_, F, h = solver.solve(prob, T, method="scaledgd", budget=factors.budget_cost(ranks, T),
                       realloc_every=20, realloc_until=200, damping="leak", c_lam=1.0,
                       iters=300, X_true=X)
print("true multi-rank  ", ranks)
print("MR-ScaledGD ranks", F.ranks, f"relative error {h.rel_err[-1]:.1e}")

# ScaledGD with the tubal-rank factorization: rank 10 in every slice
_, _, h = solver.solve(prob, T, method="scaledgd", rank=10, iters=300, X_true=X)
print("ScaledGD", h.events)
```

prints, in a few seconds,

```
true multi-rank   [10, 9, 8, 7, 6, 6, 5, 4, 3, 2, 1]
MR-ScaledGD ranks [10, 9, 8, 7, 6, 6, 5, 4, 3, 2, 1] relative error 7.6e-16
ScaledGD [(90, 'diverged')]
```

MR-ScaledGD finds the multi-rank and recovers the tensor to machine precision;
ScaledGD with the tubal rank diverges at iteration 90. Another BLAS library can
change the last digits and the iteration of the divergence.

The synthetic experiments give MR-ScaledGD the budget of the true multi-rank,
as above. On real data the multi-rank is unknown: the experiments cap the
budget at that of tubal rank 20 and let the leakage floor choose, with
`budget=factors.tubal_budget(20, T), c_floor=2.0, tol=1e-5, iters=500` and the
other arguments as above. `data.load("video:akiyo")` returns one of the
datasets fetched by `scripts/get_data.sh` (the keys are listed in
`src/tubal/data.py`). The baselines are `method="gd"` and `method="apgd"` of
`solver.solve`, and `baselines.tnn_complete`, `baselines.tctf_complete` and
`baselines.svp_solve`.

## Contents

| path | contents |
|---|---|
| `src/tubal/` | the package: `transforms`; `factors` (multi-rank factors, budgeted Eckart–Young projection); `problems` (completion, sensing, factorization, leakage level); `solver` (GD, ScaledGD, APGD, MR-ScaledGD); `baselines` (TNN-ADMM, TCTF, budgeted SVP); `metrics`; `data`; `experiments` (the campaigns); `runlog` |
| `tests/` | checks of the properties the method relies on |
| `scripts/` | `run_campaign.py` runs a campaign; `make_tables.py`, `make_figures.py` and `make_visuals.py` build the tables and figures; `get_data.sh` fetches the datasets |
| `results/runs/` | the runs, one directory each: `config.json` (parameters and seed), `env.json` (library versions, thread counts), `metrics.json` (final metrics and history) |
| `results/tables/` | the tables as CSV, plus the SSIM of the real-data comparisons |
| `paper/tables/`, `paper/figures/` | the tables as LaTeX bodies and the figures as PDF |

## Tables and figures

The scripts read the runs in `results/runs/`, or in `$TUBAL_RUNS` if it is set.
Each table goes to `paper/tables/<name>.tex` and `results/tables/<name>.csv`,
each figure to `paper/figures/<name>.pdf`.

| name | content | built by | from |
|---|---|---|---|
| `tab_video` | PSNR of each method on the six videos at sampling rates 0.1, 0.2 and 0.3 | `make_tables.py` | `video` runs |
| `tab_video_ssim` (CSV only) | SSIM of the same runs | `make_tables.py` | `video` runs |
| `tab_msi` | PSNR on the six multispectral images and the hyperspectral image at rates 0.05, 0.1 and 0.2 | `make_tables.py` | `msi`, `hsi` runs |
| `tab_msi_ssim` (CSV only) | SSIM of the same runs | `make_tables.py` | `msi`, `hsi` runs |
| `tab_time` | median run time, iterations and time per iteration on the six videos at rate 0.2 | `make_tables.py` | `timing` runs |
| `tab_ablation` | PSNR of MR-ScaledGD variants (reallocation, budget cap, floor, damping, step size) on two videos and two multispectral images | `make_tables.py` | `sensitivity` runs |
| `fig_ceiling` | PSNR of the best approximation at equal budget, tubal-rank against multi-rank truncation, and the ranks the budget allocates | `make_figures.py ceiling` | the data |
| `fig_mechanism` | relative error and energy of the spurious components along the iterations | `make_figures.py mechanism` | `spurious` runs |
| `fig_kappa_DFT` | convergence on synthetic tensors for condition numbers 1, 10 and 100, non-uniform and uniform multi-rank | `make_figures.py kappa` | `tc_synth` runs |
| `fig_phase` | success rate against the sampling rate (completion) and the number of measurements (sensing) | `make_figures.py phase` | `phase`, `sensing` runs |
| `fig_sensitivity` | PSNR of MR-ScaledGD against the budget cap, with and without the leakage floor | `make_figures.py sensitivity` | `sensitivity`, `video`, `msi` runs |
| `fig_visual` | reconstructions of a video frame and a multispectral band at rate 0.1 | `make_visuals.py` | the data; re-runs ten completions, with the ranks of the `video` and `msi` runs |

To rebuild them all (`fig_ceiling` and `fig_visual` need `scripts/get_data.sh`):

```bash
.venv/bin/python scripts/make_tables.py     # seconds
.venv/bin/python scripts/make_figures.py    # seconds
.venv/bin/python scripts/make_visuals.py    # about ten minutes
git status --short                          # nothing if every file came out as shipped
```

With the locked library versions, every file comes out byte for byte. On
another processor or BLAS library, the two figures that compute from the data,
`fig_ceiling` and `fig_visual`, can differ in their last digits.

## Rerunning the experiments

The campaigns are defined in `src/tubal/experiments.py`, and
`scripts/run_campaign.py <campaign> --list` prints the runs of one that are not
done yet. Each run is single-threaded and writes `out/runs/<campaign>/<run>/`;
a run whose `metrics.json` exists is skipped, so an interrupted campaign
resumes where it stopped. Fetch the data first (260 MB to download):

```bash
bash scripts/get_data.sh
```

Locally, for example the convergence experiment on eight cores:

```bash
.venv/bin/python scripts/run_campaign.py tc_synth --workers 8
TUBAL_RUNS=out/runs .venv/bin/python scripts/make_figures.py kappa
git diff --stat                             # what the rerun changed
```

On a Slurm cluster, from a login node that shares the file system with the
compute nodes (which may have no internet access, hence the data first),
`--workers` defaults to `$SLURM_CPUS_PER_TASK`:

```bash
sbatch --cpus-per-task=32 --mem-per-cpu=2G --time=01:00:00 \
       --wrap ".venv/bin/python scripts/run_campaign.py video"
```

`--shard i/n` takes every n-th run starting at the i-th, so a job array splits
a campaign:

```bash
sbatch --array=0-3 --cpus-per-task=32 --mem-per-cpu=2G --time=01:00:00 \
       --wrap '.venv/bin/python scripts/run_campaign.py msi --shard $SLURM_ARRAY_TASK_ID/4'
```

The shipped runs took 31 core-hours on AMD EPYC 9655 cores, from the `wall`
times recorded in their `metrics.json`:

| campaign | runs | core-hours | longest run | used by |
|---|---|---|---|---|
| `spurious` | 36 | 0.02 | 7 s | `fig_mechanism` |
| `tc_synth` | 420 | 0.6 | 17 s | `fig_kappa_DFT` |
| `phase` | 2000 | 0.4 | 5 s | `fig_phase` |
| `sensing` | 900 | 0.3 | 12 s | `fig_phase` |
| `video` | 594 | 9.1 | 5.6 min | `tab_video`, `fig_sensitivity`, `fig_visual` |
| `msi` | 594 | 9.9 | 2.8 min | `tab_msi`, `fig_sensitivity`, `fig_visual` |
| `hsi` | 99 | 8.6 | 16 min | `tab_msi` |
| `sensitivity` | 236 | 1.7 | 69 s | `tab_ablation`, `fig_sensitivity` |
| `timing` | 180 | 0.4 | 24 s | `tab_time` |
| total | 5059 | 31 | | |

A worker needs at most about 1.3 GB, for TNN minimization on the hyperspectral
image, and less elsewhere. The run times of `tab_time` depend on the machine:
that campaign ran alone on an otherwise idle node with eight workers
(`--workers 8`). A rerun on another processor or BLAS library can change the
last digits, and a run that diverges can diverge at another iteration.

## Data

`scripts/get_data.sh` downloads the three datasets into `data/` (or
`$TUBAL_DATA`) and checks every file against the SHA-256 of the copy the
experiments used. Their terms do not allow redistribution, so none is included
here.

| dataset | used | source | terms |
|---|---|---|---|
| Xiph.org video test media ("derf's collection") | luma of the first 50 frames of Akiyo, Carphone, Coastguard, Foreman, Hall monitor and News in QCIF, 144 × 176 × 50 | <https://media.xiph.org/video/derf/> | no license; the collection's page says its sequences "are believed to be freely redistributable" and that some may impose further restrictions |
| CAVE multispectral image database, Columbia University | Balloons, Beads, Feathers, Flowers, Stuffed toys and Superballs: 31 bands from 400 to 700 nm, block-averaged from 512 × 512 to 256 × 256 | <https://cave.cs.columbia.edu/repository/Multispectral> | made available to the research community; the site's copyright notice retains all rights and says that its works may in most cases not be reposted without the copyright holder's permission |
| HYDICE Washington DC Mall, Purdue University (MultiSpec) | top-left 256 × 256 crop, all 191 bands | <https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html> | "free to use for testing and/or research"; provided with the permission of the Spectral Information Technology Application Center of Virginia, which collected it |

The CAVE database is described in F. Yasuma, T. Mitsunaga, D. Iso and
S. K. Nayar, "Generalized assorted pixel camera: postcapture control of
resolution, dynamic range, and spectrum" (2010),
[doi:10.1109/TIP.2010.2046811](https://doi.org/10.1109/TIP.2010.2046811). All
three are scaled to [0, 1]; the DC Mall image is first clipped at zero, since a
few hundred of its entries are corrupted to large negative values. The
synthetic tensors are generated by the code.

## License

MIT (see `LICENSE`) for the code, the runs, and the tables and figures. The
datasets keep their own terms.
