# Bilevel Optimization Applied to L1-Regularized Linear Classification

This project implements bilevel optimization algorithms for nonsmooth convex problems, applied to linear classification with L1 regularization (sparsity) on the weights and hinge loss as the training objective.

The code includes:

- Bi-CS (Bilevel Co-Subgradient, Algorithm 2.1) with modes RL (know R and L), R (know R only), N (know none), and ER (episodic reset for unbounded domain).
- FC-BiO — a binary-search-based variant using a composite objective max(f(x)−t, g(x)−g⋆).
- a-IRG — deterministic Iteratively Regularized Gradient specialized to our bilevel setup.
- IIBA — a simple iterative inner-then-outer baseline for comparison.
- Exact baselines via Gurobi to compute the lower-level optimum (hinge loss) and then the upper-level optimum (minimal L1 norm among lower-level minimizers).

The implementation is sparse-first: data is kept in CSR sparse format and all core computations support sparse matrices. Core scripts live under `scripts/`.

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
# Bi-CS (mode selectable via --bics-mode)
python main.py --algo bics --bics-mode RL --n-samples 10000 --label-idx 0 --bound 50 --iters 1000

# FCBiO (reuses saved baselines if present)
python main.py --algo fcbio --n-samples 10000 --label-idx 0 --bound 50 --iters 1000 --eps 0.1 --skip-optimum

# a-IRG (uses same iteration flag)
python main.py --algo airg --n-samples 10000 --label-idx 0 --bound 50 --iters 1000 \
  --airg-gamma0 1.0 --airg-eta0 1.0 --airg-b 0.25 --airg-r 0.5
```

2) Plot/compare saved results

```
python plot_compare.py --results-dir results/bd50_box_iters1000_eps0.1 --algos "Bi-CS-RL" "Bi-CS-R" "Bi-CS-N" "Bi-CS-ER" "FC-BiO" "a-IRG" "IIBA"
```

Key outputs (under `--results-dir`):

- `baselines.json`: cached g* and f* for the dataset/config
- Pickled solver objects with histories: `Bi-CS-RL.pkl`, `Bi-CS-R.pkl`, `Bi-CS-N.pkl`, `Bi-CS-ER.pkl`, `FCBiO.pkl`, `aIRG.pkl`, `IIBA.pkl`
- `plot_upper.png`, `plot_lower.png`: produced by `plot_compare.py`

## Arguments

- `--n-samples`: Number of samples to use (default: 10000)
- `--label-idx`: Label column index from RCV1 (default: 0)
- `--data-dir`: Relative data subdirectory to store/find the `.npz` (default: `data`)
- `--bound`: Box bound for parameters `(w, b)` (default: 50)
- `--algo {bics,fcbio,airg,iiba}`: Select algorithm to run (default: bics)
  - `--airg-gamma0`, `--airg-eta0`, `--airg-b`, `--airg-r`: a-IRG hyperparameters per Cor. 3.5
- `--domain {box,ball}`: Feasible domain. `box` uses a hypercube constraint `(w,b) ∈ [-bound, bound]^{d+1}`; `ball` uses an L2-ball constraint `||(w,b)||_2 ≤ bound` (default: box)
- `--iters`: Total iterations for algorithms (default: 1000)
- `--eps`: Target accuracy for FC-BiO (default: 1e-1)
- `--results-dir`: Output directory (default: `results/bd{bound}_{domain}_iters{iters}_eps{eps}`)
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

`run.sh` orchestrates the full pipeline: baselines → run 7 algorithms (Bi-CS RL/R/N/ER, FC-BiO, a-IRG, IIBA) → plot.

Environment variables to override defaults:

- `NSAMPLES`, `LABEL_IDX`, `BOUND`, `ITERS`, `EPS`, `SEED`, `DATA_DIR`, `RESULTS_DIR`, `DOMAIN`
- `PARALLEL=1` to run all algorithms concurrently after baselines

Examples:

```
# Default (box domain), parallel
PARALLEL=1 ./run.sh

# Ball domain, custom size/iters, parallel
DOMAIN=ball NSAMPLES=5000 BOUND=25 ITERS=1500 PARALLEL=1 ./run.sh

# Sequential run (PARALLEL=0) with box domain
PARALLEL=0 NSAMPLES=10000 ./run.sh
```

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
