import os
import argparse
import numpy as np
from dataloader import load_rcv1_data
from scripts import BiCS, FCBiO, aIRG, IIBA, hinge_loss_minimize, L1_norm_second_minimize
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
    parser.add_argument('--algo', type=str, default='bics', choices=['bics','fcbio','airg','iiba'], help='Algorithm to run')
    parser.add_argument('--bics-mode', type=str, default='RL', choices=['RL','R','N','ER'], help='Bi-CS variant: RL (know R & L), R (know R), N (know none), ER (unbounded)')
    # Optimization / geometry
    parser.add_argument('--bound', type=float, default=50.0, help='Box bound for parameters (w,b)')
    parser.add_argument('--domain', type=str, default='box', choices=['box','ball'], help='Feasible domain: hypercube (box) or L2 ball')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # Algorithm iterations and accuracy
    parser.add_argument('--iters', type=int, default=1000, help='Total iterations for algorithms')
    parser.add_argument('--eps', type=float, default=1e-2, help='Target accuracy for FC-BiO')
    # a-IRG hyperparameters (optional)
    parser.add_argument('--airg-eta0', type=float, default=1.0, help='a-IRG η0 for η_k=η0/(k+1)^b')
    parser.add_argument('--airg-b', type=float, default=0.25, help='a-IRG exponent b in (0, 0.5)')
    parser.add_argument('--airg-r', type=float, default=0.5, help='a-IRG averaging exponent r in [0,1)')
    # Output
    parser.add_argument('--results-dir', type=str, default=None, help='Results directory (default encodes bound/domain/iters/eps)')
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
    if args.results_dir is None:
        # Include only bound/domain/iteration/epsilon per user spec
        tag = f"bd{args.bound:g}_{args.domain}_iters{args.iters}_eps{args.eps}"
        default_results = os.path.join("results", tag)
    else:
        default_results = args.results_dir
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
        if base.get('domain') and base['domain'] != args.domain:
            print("[Warning] Loaded baselines computed for domain='%s', current run uses domain='%s'." % (base['domain'], args.domain))
        print(f"Loaded baselines from {baselines_path}")
    else:
        if not args.skip_optimum:
            w_gopt, b_gopt, g_opt = hinge_loss_minimize(X, y, bound=bound, domain=args.domain)
            print(f"Optimal hinge loss: {g_opt:.4f}")
            w_opt, b_opt, xi_opt, u_opt, f_opt = L1_norm_second_minimize(X, y, g_opt, bound=bound, domain=args.domain)
            print(f"Optimal upper L1 norm: {f_opt:.4f}")
            w_for_fhat = w_gopt
            with open(baselines_path, 'w') as f:
                json.dump({'g_opt': g_opt, 'f_opt': f_opt, 'w_gopt': w_for_fhat.tolist(), 'domain': args.domain}, f)
            print(f"Saved baselines to {baselines_path}")
        else:
            # Surrogate baselines when Gurobi is not available and no saved baselines
            x0 = np.zeros(X.shape[1] + 1, dtype=np.float32)
            g_opt = hinge_loss(X, y, x0)
            f_opt = 0.0
            w_for_fhat = np.zeros(X.shape[1], dtype=np.float32)
            with open(baselines_path, 'w') as f:
                json.dump({'g_opt': g_opt, 'f_opt': f_opt, 'w_gopt': w_for_fhat.tolist(), 'domain': args.domain}, f)
            print(f"[Skip optimum] Saved surrogate baselines to {baselines_path}")

    # Early exit if only preparing baselines
    if args.only_baselines:
        print(f"Baselines ready at {baselines_path}")
        raise SystemExit(0)
    
    n, d = X.shape
    if args.domain == 'box':
        R = 2.0 * bound * np.sqrt(d + 1)
    else:
        R = 2.0 * bound
    Lf = np.sqrt(d) / d if d > 0 else 0.0
    X_mean = np.asarray(X.mean(axis=0)).ravel()
    L = max(Lf, np.sqrt(np.sum(X_mean**2) + 1))
    rng = np.random.default_rng(args.seed)
    if args.domain == 'box':
        initial = rng.uniform(-bound, bound, size=d + 1)
        #bound = np.linalg.norm(initial, ord=np.inf)
        #R = 2.0 * bound * np.sqrt(d + 1)
    else:
        direction = rng.normal(size=d + 1)
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            direction = np.zeros(d + 1)
            direction[0] = 1.0
            norm = 1.0
        direction /= norm
        radius = bound * (rng.random() ** (1.0 / (d + 1)))
        initial = direction * radius
        #bound = np.linalg.norm(initial, ord=2)
        #R = 2.0 * bound

    # Run a single algorithm and save its result object
    if args.algo == 'bics':
        f_hat = np.linalg.norm(w_for_fhat.copy(), ord=1) / d if d > 0 else 0.0
        if f_hat <= 0.0:
            f_hat = bound

        bics = BiCS(X, y, L, R, bound, initial, num_iter=args.iters, eps=args.eps, domain=args.domain, mode=args.bics_mode)
        bics.solve(start_iter=1, end_iter=args.iters)
        name = f"Bi-CS-{args.bics_mode}"
        save(save_path, bics, **{'0': name})
        print(f"Saved BiCS result to {save_path}/{name}.json")
    elif args.algo == 'fcbio':
        fcbio = FCBiO(X, y, L, bound, initial, T=args.iters, l=0, u=max(1e-6, f_opt * 64), g_star_hat=g_opt, eps=args.eps, domain=args.domain)
        _ = fcbio.solve()
        save(save_path, fcbio, **{'0': 'FCBiO'})
        print(f"Saved FCBiO result to {save_path}/FCBiO.json")
    elif args.algo == 'airg':
        airg = aIRG(
            X,
            y,
            L,
            R,
            bound=bound,
            initial=initial,
            num_iter=args.iters,
            domain=args.domain,
            eta0=args.airg_eta0,
            b=args.airg_b,
            r=args.airg_r,
        )
        _ = airg.solve(start_iter=0, end_iter=args.iters)
        save(save_path, airg, **{'0': 'aIRG'})
        print(f"Saved aIRG result to {save_path}/aIRG.json")
    elif args.algo == 'iiba':
        iiba = IIBA(X, y, L, R, bound, initial, num_iter=args.iters, domain=args.domain)
        _ = iiba.solve(start_iter=1, end_iter=args.iters)
        save(save_path, iiba, **{'0': 'IIBA'})
        print(f"Saved IIBA result to {save_path}/IIBA.json")
    
