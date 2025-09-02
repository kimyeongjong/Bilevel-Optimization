# Bilevel Optimization Applied to L1-Regularized Linear Classification

This project implements bilevel optimization algorithms for nonsmooth convex problems, applied to linear classification with L1 regularization (sparsity) on the weights and hinge loss as the training objective.

The code includes:

- Bi-CS (Bilevel Co-Subgradient) — an any-time method for nonsmooth bilevel problems.
- FC-BiO — a binary-search-based variant using a composite objective max(f(x)−t, g(x)−g⋆).
- Exact baselines via Gurobi to compute the lower-level optimum (hinge loss) and then the upper-level optimum (minimal L1 norm among lower-level minimizers).

The implementation is sparse-first: data is kept in CSR sparse format and all core computations support sparse matrices. Core scripts live under `scripts/` (previously `algorithms/`).

## Installation

1) Create and activate a virtual environment (optional but recommended).

2) Install dependencies:

```
pip install -r requirements.txt
```

Note: `gurobipy` requires a valid Gurobi installation and license. If unavailable, you can skip the exact baseline stage by using `--skip-optimum` when running `main.py`.

## Data

The script uses the RCV1 dataset from scikit-learn. On first run, if a matching pre-saved `.npz` file is not found under the specified data directory, it will download the dataset and save a subset to a sparse-friendly `.npz` file that contains CSR components (`X_data`, `X_indices`, `X_indptr`, `X_shape`) and labels `y`.

You can also pre-place a file named `rcv1_{n_samples}samples_idx{label_idx}.npz` in the data directory with the same fields to avoid downloading.

## Usage

Two common workflows are supported: run a single algorithm, or orchestrate both and compare.

1) Run a single algorithm and save results

```
# BiCS
python main.py --algo bics --n-samples 10000 --label-idx 0 --bound 50 --bics-iters 1000 \
  --results-dir results/results_10000samples

# FCBiO (reuses saved baselines if present)
python main.py --algo fcbio --n-samples 10000 --label-idx 0 --bound 50 --fcbio-T 1000 --eps 0.1 \
  --results-dir results/results_10000samples --skip-optimum
```

2) Plot/compare saved results

```
python plot_compare.py --results-dir results/results_10000samples --algos BiCS FCBiO
```

Key outputs (under `--results-dir`):

- `baselines.json`: cached g* and f* for the dataset/config
- `BiCS.pkl`, `FCBiO.pkl`: pickled solver objects with histories
- `plot_upper.png`, `plot_lower.png`: produced by `plot_compare.py`

## Arguments

- `--n-samples`: Number of samples to use (default: 10000)
- `--label-idx`: Label column index from RCV1 (default: 0)
- `--data-dir`: Relative data subdirectory to store/find the `.npz` (default: `data`)
- `--bound`: Box bound for parameters `(w, b)` (default: 50)
- `--algo {bics,fcbio}`: Select algorithm to run (default: bics)
- `--domain {box,ball}`: Feasible domain. `box` uses a hypercube constraint `(w,b) ∈ [-bound, bound]^{d+1}`; `ball` uses an L2-ball constraint `||(w,b)||_2 ≤ bound` (default: box)
- `--bics-iters`: BiCS total iterations (default: 1000)
- `--fcbio-T`: FC-BiO total iterations across all episodes (default: 1000)
- `--eps`: Target accuracy for FC-BiO (default: 1e-1)
- `--results-dir`: Output directory (default: `results/results_{n_samples}samples/`)
- `--seed`: Random seed (default: 42)
- `--skip-optimum`: Reuse or write surrogate baselines instead of solving with Gurobi
- `--only-baselines`: Compute/save baselines and exit

Example:

```
# Baselines only (ball domain), then parallel runs via run.sh
python main.py --only-baselines --n-samples 5000 --label-idx 2 --bound 25 --domain ball \
  --results-dir results/exp_rcv1_5k
PARALLEL=1 NSAMPLES=5000 LABEL_IDX=2 BOUND=25 ./run.sh
```

## Convenience Runner

`run.sh` orchestrates the full pipeline: baselines → run BiCS/FCBiO (sequential or parallel) → plot.

Environment variables to override defaults:

- `NSAMPLES`, `LABEL_IDX`, `BOUND`, `BICS_ITERS`, `FCBIO_T`, `EPS`, `SEED`, `DATA_DIR`, `RESULTS_DIR`
- `PARALLEL=1` to run BiCS and FCBiO concurrently after baselines

## Domains and Baselines

- Domain selection affects both the algorithm projections and the exact baselines (g⋆, f⋆):
  - `--domain box`: `(w,b)` constrained to the hypercube `[-bound, bound]^{d+1}`.
  - `--domain ball`: `(w,b)` constrained to the L2-ball of radius `bound` via a quadratic constraint.
- `baselines.json` stores the domain used to compute g⋆ and f⋆. When loading existing baselines with a different `--domain`, a warning is shown; re-run `--only-baselines` to regenerate for the current domain.
- Gurobi note: `--domain ball` uses a quadratic constraint (QCQP). Ensure your Gurobi installation/license supports QCP.

## Notes

- Headless plotting is enabled by default (matplotlib Agg backend; no `plt.show()`).
- If you skip the optimum stage, the upper/lower gaps are computed against surrogate baselines (e.g., zero vector for g⋆); the absolute values are not directly comparable to the exact optimal gaps but still show convergence trends.
- Lipschitz constants use heuristic estimates. Consider feature normalization or conservative bounds for stability on different datasets.
