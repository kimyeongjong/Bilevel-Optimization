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
    def __init__(self, X, y, Lg, L, R, bound, initial, num_iter=1000, domain='box', mode='RL'):
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
        # Mode: 'RL' (know R & L), 'R' (know R only), 'N' (know none), 'ER' (unbounded domain, Lipschitz f,g)
        self.mode = mode.upper()
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

    def solve(self, start_iter, end_iter):
        x = self.initial.copy() # x_t
        x_ = self.initial.copy() # y_t

        Gsq = 0.0 # denominator for Adagrad on x
        Gsq_y = 0.0 # accumulator for y when Lg unknown
        delta_prev = np.inf
        gy_ = np.inf

        for t in tqdm(range(start_iter, end_iter + 1)):
            self.x_hist.append(x.copy())
            self.f_hist.append(self.f_val(x))
            self.g_hist.append(self.g_val(x))
            
            gt = self.g_val(x)
            # Update y_t depending on mode
            gy_grad = self.subgrad_g(x_)
            if self.mode == 'RL':
                step_y = 2 * self.R / (self.Lg * np.sqrt(t))
                x_ = x_ - step_y * gy_grad
                # project into domain
                x_ = project_onto_box(x_, self.bound) if self.domain == 'box' else project_onto_ball(x_, self.bound)
            elif self.mode == 'R':
                # Unknown Lg: AdaGrad on y using R
                Gsq_y += float(np.dot(gy_grad, gy_grad))
                step_y = 2 * self.R / np.sqrt(Gsq_y + 1e-16)
                x_ = x_ - step_y * gy_grad
                x_ = project_onto_box(x_, self.bound) if self.domain == 'box' else project_onto_ball(x_, self.bound)
            elif self.mode == 'N':
                # Unknown R and Lg (parameter-free style): AdaGrad on y without R
                Gsq_y += float(np.dot(gy_grad, gy_grad))
                step_y = 1.0 / np.sqrt(Gsq_y + 1e-16)
                x_ = x_ - step_y * gy_grad
                x_ = project_onto_box(x_, self.bound) if self.domain == 'box' else project_onto_ball(x_, self.bound)
            else:  # 'ER' — unbounded domain; skip projection, parameter-free AdaGrad
                Gsq_y += float(np.dot(gy_grad, gy_grad))
                step_y = 1.0 / np.sqrt(Gsq_y + 1e-16)
                x_ = x_ - step_y * gy_grad
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
            
            Gsq += float(np.dot(v, v))
            if self.mode in ('RL', 'R'):
                eta = self.R / np.sqrt(Gsq + 1e-16)
            else:
                eta = 1.0 / np.sqrt(Gsq + 1e-16)
            x = x - eta * v
            # project back into domain unless ER (unbounded)
            if self.mode != 'ER':
                x = project_onto_box(x, self.bound) if self.domain == 'box' else project_onto_ball(x, self.bound)
            delta_prev = dt
        self.checkpoint = x.copy()
