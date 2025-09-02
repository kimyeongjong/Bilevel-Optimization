import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataloader import load_rcv1_data
from scripts import BiCS


def run_and_plot(n_samples=10000, label_idx=0, data_dir='data', bound=50.0, iters=500, seed=42):
    # Load sparse data
    X, y = load_rcv1_data(n_samples=n_samples, label_idx=label_idx, path=data_dir)

    # Compute constants (heuristics consistent with main.py)
    n, d = X.shape
    R = bound * np.sqrt(d + 1)
    Lf = np.sqrt(d) / d if d > 0 else 0.0
    X_mean = np.asarray(X.mean(axis=0)).ravel()
    Lg = np.sqrt(np.sum(X_mean**2) + 1)
    L = max(Lf, Lg)

    rng = np.random.default_rng(seed)
    initial = rng.uniform(-bound, bound, size=d + 1)

    # Run BiCS
    bics = BiCS(X, y, Lg, L, R, bound, initial, num_iter=iters)
    bics.solve(bics.initial, start_iter=1, end_iter=iters)

    # Plot gy_ trajectory
    its = np.arange(1, len(bics.gy_min_hist) + 1)
    plt.figure()
    plt.plot(its, bics.gy_min_hist, label='BiCS gy_min (running min of g(y_t))')
    plt.xlabel('Iteration')
    plt.ylabel('g(y) running minimum')
    plt.grid(True)
    plt.legend()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'gy_min.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved gy_min plot to {out_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Track and plot BiCS gy_ (running min of g(y_t))')
    ap.add_argument('--n-samples', type=int, default=10000)
    ap.add_argument('--label-idx', type=int, default=0)
    ap.add_argument('--data-dir', type=str, default='data')
    ap.add_argument('--bound', type=float, default=50.0)
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    run_and_plot(
        n_samples=args.n_samples,
        label_idx=args.label_idx,
        data_dir=args.data_dir,
        bound=args.bound,
        iters=args.iters,
        seed=args.seed,
    )

