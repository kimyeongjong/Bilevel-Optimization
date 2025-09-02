import os
import argparse
import numpy as np
from dataloader import load_rcv1_data
from scripts import BiCS, FCBiO, hinge_loss_minimize, L1_norm_second_minimize
from utils import save, plot
from scripts.settings import hinge_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description='Bilevel Optimization Applied to L1-Regularized Linear Classification'
    )
    # Data
    parser.add_argument('--n-samples', type=int, default=10000, help='Number of samples from RCV1')
    parser.add_argument('--label-idx', type=int, default=0, help='Label index from RCV1 targets')
    parser.add_argument('--data-dir', type=str, default='data', help='Relative data subdirectory under dataloader/')
    # Optimization / geometry
    parser.add_argument('--bound', type=float, default=50.0, help='Box bound for parameters (w,b)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # Algorithm iterations and accuracy
    parser.add_argument('--bics-iters', type=int, default=1000, help='BiCS iterations')
    parser.add_argument('--fcbio-T', type=int, default=1000, help='FC-BiO total iterations')
    parser.add_argument('--eps', type=float, default=1e-1, help='Target accuracy for FC-BiO')
    # Output
    parser.add_argument('--results-dir', type=str, default=None, help='Results directory (default: results/results_{n}samples)')
    parser.add_argument('--no-plots', action='store_true', help='Disable saving plots')
    # Baselines
    parser.add_argument('--skip-optimum', action='store_true', help='Skip exact baselines via Gurobi')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    # Load data
    X, y = load_rcv1_data(n_samples=args.n_samples, label_idx=args.label_idx, path=args.data_dir)
    print(f"Data loaded: X shape {X.shape}, y shape {y.shape}")

    # Obtain g_opt and f_opt
    bound = args.bound
    if not args.skip_optimum:
        _, w_for_fhat, g_opt = hinge_loss_minimize(X, y, bound=bound)
        print(f"Optimal hinge loss: {g_opt:.4f}")
        w_opt, b_opt, xi_opt, u_opt, f_opt = L1_norm_second_minimize(X, y, g_opt, bound=bound)
        print(f"Optimal upper L1 norm: {f_opt:.4f}")
    else:
        # Surrogate baselines when Gurobi is not available
        x0 = np.zeros(X.shape[1] + 1, dtype=np.float32)
        g_opt = hinge_loss(X, y, x0)
        f_opt = 0.0
        w_for_fhat = np.zeros(X.shape[1], dtype=np.float32)
        print(f"[Skip optimum] Surrogate g*: {g_opt:.4f}, f*: {f_opt:.4f}")

    # Prepare common parameters for algorithms (sparse-friendly)
    n, d = X.shape
    f_hat = np.linalg.norm(w_for_fhat.copy(), ord=1) / d if d > 0 else 0.0
    if f_hat <= 0.0:
        f_hat = bound  # loose upper bound if unavailable
    R = bound * np.sqrt(d + 1)
    Lf = np.sqrt(d) / d if d > 0 else 0.0
    # compute Lg from sparse mean vector
    X_mean = np.asarray(X.mean(axis=0)).ravel()
    Lg = np.sqrt(np.sum(X_mean**2) + 1)
    L  = max(Lf, Lg)
    rng = np.random.default_rng(args.seed)
    initial = rng.uniform(-bound, bound, size=d + 1)  # Initial point for optimization

    # Initialize and train BiCS algorithm
    # Pass Lg and L separately to match BiCS signature
    bics = BiCS(X, y, Lg, L, R, bound, initial, num_iter=args.bics_iters)
    bics.solve(bics.initial, start_iter=1, end_iter=args.bics_iters)
    
    # Initialize and train FCBiO algorithm
    fcbio = FCBiO(X, y, L, bound, initial, T=args.fcbio_T, l=0, u=f_hat, g_star_hat=g_opt, eps=args.eps)
    _, fcbio_f, fcbio_g = fcbio.solve()

    # Save results
    default_results = f"results/results_{args.n_samples}samples/" if args.results_dir is None else args.results_dir
    save_path = os.path.join(here, default_results)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = [bics, fcbio]
    algorithms = {'0': 'BiCS', '1': 'FCBiO'}
    save(save_path, *results, **algorithms)
    print(f"Results saved to {save_path}")

    # Plot results
    if not args.no_plots:
        plot_path_upper = os.path.join(save_path, "plot_upper.png")
        plot_upper = [np.array(bics.f_plot) - f_opt, fcbio_f - f_opt]
        plot(plot_path_upper, 'Upper Gap', *plot_upper, **algorithms)
        print(f"Plots saved to {plot_path_upper}")

        plot_path_lower = os.path.join(save_path, "plot_lower.png")
        plot_lower = [np.array(bics.g_plot) - g_opt, fcbio_g - g_opt]
        plot(plot_path_lower, 'Lower Gap', *plot_lower, **algorithms)
        print(f"Plots saved to {plot_path_lower}")
    
