import numpy as np
from tqdm import tqdm

class BiCS:
    def __init__(self, X, y, L, R, bound, initial, num_iter=1000):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.L = L
        self.R = R
        self.bound = bound
        self.num_iter = num_iter
        self.initial = initial.copy()
        self.checkpoint = np.zeros(self.n_features + 1)
        self.f_hist = [None]
        self.g_hist = [None]
        self.x_hist = [None]
        self.criterion = [None]
        self.f_plot = []
        self.g_plot = []

    def u_t(self, t): 
        return 3 * self.L * self.R / np.sqrt(t)

    def subgrad_f(self, x):
        # subgradient of L1: sign(x), break ties with zero→0
        res = np.sign(x)
        res[-1] = 0
        return res / self.n_features

    def subgrad_g(self, x):
        # subgradient of hinge loss: -sum_{i in M} y_i x_i
        margins = self.y * (self.X.dot(x[:-1]) + x[-1])
        active = np.where(margins < 1)[0]
        if len(active)==0:
            return np.zeros_like(x)
        # average subgradient
        z1 = -np.sum([self.y[i] * self.X[i] for i in active], axis=0) / self.n_samples
        z2 = -np.sum([self.y[i] for i in active], axis=0) / self.n_samples
        
        return np.concatenate([z1, [z2]], axis=0)

    def g_val(self, x):
        margins = self.y * (self.X.dot(x[:-1]) + x[-1])
        return np.mean(np.maximum(0, 1 - margins))

    def f_val(self, x):
        return np.linalg.norm(x[:-1], ord=1) / self.n_features

    def solve(self, initial, start_iter, end_iter):
        x = self.initial.copy() # x_t
        x_ = self.initial.copy() # y_t

        Gsq = 0.0 # denominator for Adagrad
        delta_prev = np.inf

        for t in tqdm(range(start_iter, end_iter + 1)):
            self.x_hist.append(x.copy())
            self.f_hist.append(self.f_val(x))
            self.g_hist.append(self.g_val(x))
            
            gt = self.g_val(x)
            x_ -= self.subgrad_g(x_) * (2 / self.L) # y_t
            x = np.clip(x, -self.bound, self.bound)
            gy = self.g_val(x_)
            dt = min(delta_prev, self.u_t(t))
            # choose subgradient
            if gt <= gy + dt:
                v = self.subgrad_f(x)
                self.criterion.append(True)
            else:
                v = self.subgrad_g(x)
                self.criterion.append(False)

            l = (t + 1) // 2
            tau = l + np.argmin([self.f_hist[i] if self.g_hist[i] == True else np.inf for i in range(l, t + 1)])

            z = self.x_hist[tau]
            self.f_plot.append(self.f_val(z))
            self.g_plot.append(self.g_val(z))
            
            Gsq += np.dot(v, v)
            eta = self.R / np.sqrt(Gsq + 1e-16)
            x = x - eta * v
            # project back into box [-bound, bound]
            x[:-1] = np.clip(x[:-1], -self.bound, self.bound)
            delta_prev = dt
        self.checkpoint = x.copy()