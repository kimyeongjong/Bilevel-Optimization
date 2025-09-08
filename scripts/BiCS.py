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

class BiCS:
    def __init__(self, X, y, Lg, L, R, bound, initial, num_iter=1000, domain='box'):
        # Data and dimensions
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        # Lipschitz constants
        self.L = L
        self.Lg = Lg
        # Radius and bounds
        self.R = R
        self.bound = bound
        self.num_iter = num_iter
        self.domain = domain  # 'box' or 'ball'
        self.initial = initial.copy()
        self.checkpoint = np.zeros(self.n_features + 1)
        self.f_hist = [None]
        self.g_hist = [None]
        self.x_hist = [None]
        self.criterion = [None]
        self.f_plot = []
        self.g_plot = []
        # Track running minimum of g(y_t), i.e., gy_
        self.gy_min_hist = []

    def u_t(self, t): 
        return 3 / np.sqrt(t)

    def subgrad_f(self, x):
        return subgradient_l1_norm(x, self.n_features)

    def subgrad_g(self, x):
        return subgradient_hinge_loss(self.X, self.y, x)

    def g_val(self, x):
        return hinge_loss(self.X, self.y, x)

    def f_val(self, x):
        return l1_norm(x, self.n_features)

    def solve(self, initial, start_iter, end_iter):
        x = self.initial.copy() # x_t
        x_ = self.initial.copy() # y_t

        Gsq = 0.0 # denominator for Adagrad
        delta_prev = np.inf
        gy_ = np.inf

        for t in tqdm(range(start_iter, end_iter + 1)):
            self.x_hist.append(x.copy())
            self.f_hist.append(self.f_val(x))
            self.g_hist.append(self.g_val(x))
            
            gt = self.g_val(x)
            x_ -= self.subgrad_g(x_) * (2 * self.R / (self.Lg * np.sqrt(t))) # y_t
            x_ = project_onto_box(x_, self.bound) if self.domain == 'box' else project_onto_ball(x_, self.bound)
            gy = self.g_val(x_)
            gy_ = min(gy, gy_)
            # record the running minimum gy_
            self.gy_min_hist.append(gy_)
            dt = min(delta_prev, self.u_t(t))
            # choose subgradient
            if gt <= gy_ + dt:
                v = self.subgrad_f(x)
                self.criterion.append(True)
            else:
                v = self.subgrad_g(x)
                self.criterion.append(False)

            l = (t + 1) // 2
            tau = l + np.argmin([self.f_hist[i] if self.criterion[i] else np.inf for i in range(l, t + 1)])

            z = self.x_hist[tau]
            self.f_plot.append(self.f_val(z))
            self.g_plot.append(self.g_val(z))
            
            Gsq += np.dot(v, v)
            eta = self.R / np.sqrt(Gsq + 1e-16)
            x = x - eta * v
            # project back into box [-bound, bound]
            x = project_onto_box(x, self.bound) if self.domain == 'box' else project_onto_ball(x, self.bound)
            delta_prev = dt
        self.checkpoint = x.copy()
