import os
import argparse
from dataloader import load_rcv1_data
from algorithms import *
from utils import save, load, plot
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    # Load data
    X, y = load_rcv1_data(n_samples=10000, label_idx=0, path='data')
    print(f"Data loaded: X shape {X.shape}, y shape {y.shape}")

    # Obtain g_opt and f_opt
    bound = 50
    _, __, g_opt = hinge_loss_minimize(X, y, bound=bound)
    print(f'Type of _: {type(_)}')
    print(f"Optimal hinge loss: {g_opt:.4f}")
    w_opt, b_opt, xi_opt, u_opt, f_opt = L1_norm_second_minimize(X, y, g_opt, bound=bound)
    print(f"Optimal upper L1 norm: {f_opt:.4f}")

    # Prepare common parameters for algorithms
    X = X.toarray()  # Convert to dense array if needed
    n, d = X.shape
    f_hat = np.linalg.norm(_.copy(), ord=1) / d # upper bound for FC-BiO
    R = bound * np.sqrt(d + 1)
    Lf = np.sqrt(d) / d
    Lg = np.sqrt(np.sum(np.square(np.mean(X, axis=0))) + 1)
    L  = max(Lf, Lg)
    initial = np.random.uniform(-bound, bound, size=d + 1)  # Initial point for optimization

    # Initialize and train BiCS algorithm
    bics = BiCS(X, y, L, R, bound, initial, num_iter=1000)
    bics.solve(bics.initial, start_iter=1, end_iter=1000)
    
    # Initialize and train FCBiO algorithm
    fcbio = FCBiO(X, y, L, bound, initial, T=1000, l=0, u=f_hat, hat_opt_g_x=g_opt, eps=1e-1)
    _, fcbio_f, fcbio_g = fcbio.solve()

    # Save results
    save_path = os.path.join(here, f"results/results_1000samples/")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = [bics, fcbio]
    algorithms = {'0': 'BiCS', '1': 'FCBiO'}
    save(save_path, *results, **algorithms)
    print(f"Results saved to {save_path}")

    # Plot results
    plot_path_upper = os.path.join(save_path, f"plot_upper.png")
    plot_upper = [np.array(bics.f_plot) - f_opt, fcbio_f - f_opt]
    plot(plot_path_upper, 'Upper Gap', *plot_upper, **algorithms)
    print(f"Plots saved to {plot_path_upper}")

    plot_path_lower = os.path.join(save_path, f"plot_lower.png")
    plot_lower = [np.array(bics.g_plot) - g_opt, fcbio_g - g_opt]
    plot(plot_path_lower, 'Lower Gap', *plot_lower, **algorithms)
    print(f"Plots saved to {plot_path_lower}")
    