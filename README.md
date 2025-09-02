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

Run experiments via:

```
python main.py [options]
```

Key outputs:

- Pickled algorithm objects in `results/` (by default), one per algorithm.
- Upper/Lower gap plots saved as PNGs.

## Arguments

- `--n-samples`: Number of samples to use (default: 10000)
- `--label-idx`: Label column index from RCV1 (default: 0)
- `--data-dir`: Relative data subdirectory to store/find the `.npz` (default: `data`)
- `--bound`: Box bound for parameters `(w, b)` (default: 50)
- `--bics-iters`: BiCS total iterations (default: 1000)
- `--fcbio-T`: FC-BiO total iterations across all episodes (default: 1000)
- `--eps`: Target accuracy for FC-BiO (default: 1e-1)
- `--results-dir`: Output directory for pickles and plots (default: `results/results_{n_samples}samples/`)
- `--seed`: Random seed (default: 42)
- `--skip-optimum`: Skip exact baseline via Gurobi; uses loose surrogates for plotting baselines (default: off)
- `--no-plots`: Do not save plots (default: off)

Example:

```
python main.py \
  --n-samples 5000 \
  --label-idx 2 \
  --bound 25 \
  --bics-iters 1500 \
  --fcbio-T 1500 \
  --eps 5e-2 \
  --results-dir results/exp_rcv1_5k \
  --skip-optimum
```

## Notes

- Headless plotting is enabled by default (matplotlib Agg backend; no `plt.show()`).
- If you skip the optimum stage, the upper/lower gaps are computed against surrogate baselines (e.g., zero vector for g⋆); the absolute values are not directly comparable to the exact optimal gaps but still show convergence trends.
- Lipschitz constants use heuristic estimates. Consider feature normalization or conservative bounds for stability on different datasets.
