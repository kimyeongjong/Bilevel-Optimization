import os
import argparse
import numpy as np
from dataloader import load_rcv1_data
from scripts import BiCS, FCBiO, hinge_loss_minimize, L1_norm_second_minimize
from utils import save
from scripts.settings import hinge_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description='Bilevel Optimization Applied to L1-Regularized Linear Classification'
    )
    # Data
    parser.add_argument('--n-samples', type=int, default=10000, help='Number of samples from RCV1')
    parser.add_argument('--label-idx', type=int, default=0, help='Label index from RCV1 targets')
    parser.add_argument('--data-dir', type=str, default='data', help='Relative data subdirectory under dataloader/')
    # Which algorithm to run
    parser.add_argument('--algo', type=str, default='bics', choices=['bics','fcbio'], help='Algorithm to run')
    # Optimization / geometry
    parser.add_argument('--bound', type=float, default=50.0, help='Box bound for parameters (w,b)')
    parser.add_argument('--domain', type=str, default='box', choices=['box','ball'], help='Feasible domain: hypercube (box) or L2 ball')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # Algorithm iterations and accuracy
    parser.add_argument('--bics-iters', type=int, default=1000, help='BiCS iterations')
    parser.add_argument('--fcbio-T', type=int, default=1000, help='FC-BiO total iterations')
    parser.add_argument('--eps', type=float, default=1e-2, help='Target accuracy for FC-BiO')
    # Output
    parser.add_argument('--results-dir', type=str, default=None, help='Results directory (default: results/results_{n}samples)')
    # Baselines
    parser.add_argument('--skip-optimum', action='store_true', help='Skip exact baselines via Gurobi')
    parser.add_argument('--only-baselines', action='store_true', help='Compute/save baselines and exit')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    # Load data
    X, y = load_rcv1_data(n_samples=args.n_samples, label_idx=args.label_idx, path=args.data_dir)
    print(f"Data loaded: X shape {X.shape}, y shape {y.shape}")

    # Results directory and baselines file
    bound = args.bound
    default_results = f"results/results_{args.n_samples}samples/" if args.results_dir is None else args.results_dir
    save_path = os.path.join(here, default_results)
    os.makedirs(save_path, exist_ok=True)

    import json
    baselines_path = os.path.join(save_path, 'baselines.json')

    # Obtain g_opt and f_opt
    if args.skip_optimum and os.path.exists(baselines_path):
        with open(baselines_path, 'r') as f:
            base = json.load(f)
        g_opt = float(base['g_opt'])
        f_opt = float(base['f_opt'])
        w_for_fhat = np.array(base.get('w_gopt', np.zeros(X.shape[1], dtype=np.float32)))
        print(f"Loaded baselines from {baselines_path}")
    else:
        if not args.skip_optimum:
            w_gopt, b_gopt, g_opt = hinge_loss_minimize(X, y, bound=bound)
            print(f"Optimal hinge loss: {g_opt:.4f}")
            w_opt, b_opt, xi_opt, u_opt, f_opt = L1_norm_second_minimize(X, y, g_opt, bound=bound)
            print(f"Optimal upper L1 norm: {f_opt:.4f}")
            w_for_fhat = w_gopt
            with open(baselines_path, 'w') as f:
                json.dump({'g_opt': g_opt, 'f_opt': f_opt, 'w_gopt': w_for_fhat.tolist()}, f)
            print(f"Saved baselines to {baselines_path}")
        else:
            # Surrogate baselines when Gurobi is not available and no saved baselines
            x0 = np.zeros(X.shape[1] + 1, dtype=np.float32)
            g_opt = hinge_loss(X, y, x0)
            f_opt = 0.0
            w_for_fhat = np.zeros(X.shape[1], dtype=np.float32)
            with open(baselines_path, 'w') as f:
                json.dump({'g_opt': g_opt, 'f_opt': f_opt, 'w_gopt': w_for_fhat.tolist()}, f)
            print(f"[Skip optimum] Saved surrogate baselines to {baselines_path}")

    # Early exit if only preparing baselines
    if args.only_baselines:
        print(f"Baselines ready at {baselines_path}")
        raise SystemExit(0)

    # Run a single algorithm and save its result object
    if args.algo == 'bics':
        n, d = X.shape
        f_hat = np.linalg.norm(w_for_fhat.copy(), ord=1) / d if d > 0 else 0.0
        if f_hat <= 0.0:
            f_hat = bound
        R = bound * np.sqrt(d + 1)
        Lf = np.sqrt(d) / d if d > 0 else 0.0
        X_mean = np.asarray(X.mean(axis=0)).ravel()
        Lg = np.sqrt(np.sum(X_mean**2) + 1)
        L  = max(Lf, Lg)
        rng = np.random.default_rng(args.seed)
        initial = rng.uniform(-bound, bound, size=d + 1)

        bics = BiCS(X, y, Lg, L, R, bound, initial, num_iter=args.bics_iters, domain=args.domain)
        bics.solve(bics.initial, start_iter=1, end_iter=args.bics_iters)
        save(save_path, bics, **{'0': 'BiCS'})
        print(f"Saved BiCS result to {save_path}/BiCS.pkl")
    elif args.algo == 'fcbio':
        n, d = X.shape
        R = bound * np.sqrt(d + 1)
        Lf = np.sqrt(d) / d if d > 0 else 0.0
        X_mean = np.asarray(X.mean(axis=0)).ravel()
        Lg = np.sqrt(np.sum(X_mean**2) + 1)
        L  = max(Lf, Lg)
        rng = np.random.default_rng(args.seed)
        initial = rng.uniform(-bound, bound, size=d + 1)

        fcbio = FCBiO(X, y, L, bound, initial, T=args.fcbio_T, l=0, u=max(1e-6, np.linalg.norm(w_for_fhat, 1) / max(1, d)), g_star_hat=g_opt, eps=args.eps, domain=args.domain)
        _ = fcbio.solve()
        save(save_path, fcbio, **{'0': 'FCBiO'})
        print(f"Saved FCBiO result to {save_path}/FCBiO.pkl")
    
