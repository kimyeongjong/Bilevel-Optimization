import math
import time
import copy
from tqdm import trange
import numpy as np
from .settings import (
    l1_norm,
    subgradient_l1_norm,
    hinge_loss,
    subgradient_hinge_loss,
    project_onto_box,
    project_onto_ball,
)


class FCBiO:
    def __init__(self, X, y, L, bound, initial, T, l, u, g_star_hat, eps, domain='box'):
        """
        Initialize FC-BiO solver
        
        Parameters:
         - X, y: data and labels (sparse X supported)
         - initial: Initial parameter vector (d+1,)
         - T: Number of iterations
         - g_star_hat: approximate lower-level optimum value
         - l: lower bound for upper-level objective
         - u: upper bound (e.g., value at lower-level solution)
         - L: Lipschitz constant for psi subgrad norm bound
         - eps: target accuracy
        """
        self.X = X
        self.y = y
        self.initial = initial.copy()
        self.T = T
        
        self.l = l
        self.u = u
        self.g_star_hat = g_star_hat
        self.eps = eps
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.p = self.d + 1
        self.L = L
        
        self.N = math.ceil(np.log2((self.u - self.l) / (self.eps / 2)))
        self.K = math.ceil(self.T / self.N)
        
        # Trajectories for iterates (x) and \eta_t
        self.x_trajectory_mean = np.zeros((self.N * self.K, self.p), dtype=np.float32)
        # parameter trajectory during the outer loop
        self.x_trajectory_mean[0] = self.initial.copy()
        self.w_trajectory = None
        # parameter trajectory during the inner loop
        self.w_u_trajectory = None
        # u parameter trajectory in the inner loop
        
        # Projection bound (hypercube radius)
        self.bound = bound
        self.domain = domain  # 'box' or 'ball'
        
        # Stepsize for \eta_t
        D = 2 * self.bound * np.sqrt(self.d + 1)
        self.eta = D / (L * self.K**(1/2))
        
        # To store history of objective evaluations
        self.l1_norm_history = None
        self.hinge_loss_history = None
        
    def subgradient_psi(self, X, y, x, t): # x = (w, b)
        # tilde_g_x = g_x - g_star_hat
        f_x = l1_norm(x, self.d)
        tilde_g_x = (hinge_loss(X, y, x) - self.g_star_hat) / 2
        if f_x - t >= tilde_g_x:
            subgrad_psi = subgradient_l1_norm(x, self.d)
        else:
            subgrad_psi = subgradient_hinge_loss(X, y, x) / 2
        return subgrad_psi
        
    def solve(self):
        print("FC-BiO starts.")
        start_total = time.time()  # Runtime measurement start
        
        bar_x = copy.deepcopy(self.x_trajectory_mean[0])
        
        for n in range(self.N):
            iter_start = time.time()  # binary search iteration start time
            t = (self.l + self.u) / 2
            print("──────────── FC-BiO ────────────")
            print(f"[{n+1}/{self.N}] t = {(self.l + self.u) / 2:.6f}")
            
            # Initialize trajectories
            self.w = np.zeros(shape=(self.K, self.p), dtype=np.float32)
            self.w[0] = copy.deepcopy(bar_x)
            self.w_u = np.zeros(shape=(self.K, self.p), dtype=np.float32)
            self.w_u[0] = copy.deepcopy(bar_x)
            
            for k in trange(self.K - 1, desc=f"FC-BiO Sub-iterations (n={n+1})", leave=False):
                s = self.subgradient_psi(self.X, self.y, self.w[k], t)
                if self.domain == 'box':
                    self.w[k + 1] = project_onto_box(self.w[k] - self.eta * s, self.bound)
                else:
                    self.w[k + 1] = project_onto_ball(self.w[k] - self.eta * s, self.bound)
                
                s_u = self.subgradient_psi(self.X, self.y, self.w_u[k], self.u)
                if self.domain == 'box':
                    self.w_u[k + 1] = project_onto_box(self.w_u[k] - self.eta * s_u, self.bound)
                else:
                    self.w_u[k + 1] = project_onto_ball(self.w_u[k] - self.eta * s_u, self.bound)
            
            self.w_u_mean = np.cumsum(self.w_u, axis=0) / (np.arange(1, len(self.w_u) + 1))[:, np.newaxis]
            self.x_trajectory_mean[n * self.K:(n + 1) * self.K] = copy.deepcopy(self.w_u_mean)
            
            hat_x_t = np.mean(self.w, axis=0)
            f_hat_x_t = l1_norm(hat_x_t, self.d)
            tilde_g_hat_x_t = (hinge_loss(self.X, self.y, hat_x_t) - self.g_star_hat) / 2
            hat_psi_ast_t = max(f_hat_x_t - t, tilde_g_hat_x_t)
            
            bar_x = copy.deepcopy(hat_x_t)
            
            # binary search update
            if hat_psi_ast_t > self.eps / 2:
                self.l = t
                decision = "↑ l ← t"
            else:
                self.u = t
                decision = "↓ u ← t"
            
            iter_time = time.time() - iter_start
            print(f"ψ̂(t) = {hat_psi_ast_t:.6e} → {decision} (Runtime: {iter_time:.2f}s)")
        
        total_time = time.time() - start_total
        print(f"FC-BiO terminated (Total runtime: {total_time:.2f}s)")
        
        # Evaluate trajectory
        self.l1_norm_history = np.array([
            l1_norm(self.x_trajectory_mean[i], self.d)
            for i in range(self.T)
        ])
        self.hinge_loss_history = np.array([
            hinge_loss(self.X, self.y, self.x_trajectory_mean[i])
            for i in range(self.T)
        ])
        
        return self.x_trajectory_mean, self.l1_norm_history, self.hinge_loss_history
