import numpy as np
from tqdm import tqdm
from .settings import (
    l1_norm,
    subgradient_l1_norm,
    hinge_loss,
    subgradient_hinge_loss,
    project_onto_box,
    project_onto_ball,
)

class IIBA:
    def __init__(self, X, y, L, R, bound, initial, num_iter=1000, domain='box'):
        # Data and dimensions
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        # Lipschitz constant
        self.L = L
        # Radius and bounds
        self.R = R
        self.bound = bound
        self.num_iter = num_iter
        self.domain = domain  # 'box' or 'ball'
        self.initial = initial.copy()
        self.checkpoint = np.zeros(self.n_features + 1)
        self.f_plot = [None]
        self.g_plot = [None]
        self.x_hist = [None]

    # Project onto domain
    def _proj(self, x):
        if self.domain == "box":
            return project_onto_box(x, self.bound)
        return project_onto_ball(x, self.bound)

    def lambda_t(self, t):
        return t ** (-2 / 3)
    
    def mu_t(self, t):
        return t ** (-1)
    
    def subgrad_f(self, x):
        return subgradient_l1_norm(x, self.n_features)

    def subgrad_g(self, x):
        return subgradient_hinge_loss(self.X, self.y, x)

    def g_val(self, x):
        return hinge_loss(self.X, self.y, x)

    def f_val(self, x):
        return l1_norm(x, self.n_features)
    
    def solve(self, start_iter, end_iter):
        x = self.initial.copy()

        for t in tqdm(range(start_iter, end_iter + 1), desc="IIBA Progress"):
            self.x_hist.append(x.copy())
            self.f_plot.append(self.f_val(x))
            self.g_plot.append(self.g_val(x))
            
            lambda_t = self.lambda_t(t)
            mu_t = self.mu_t(t)
            x = x - lambda_t * self.subgrad_g(x)
            x = x - mu_t * self.subgrad_f(x)
            x = self._proj(x)

        self.checkpoint = x.copy()
        return x
            