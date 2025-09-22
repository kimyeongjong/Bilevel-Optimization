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


class aIRG:
    """
    Algorithm 3.1 (a-IRG) — deterministic Iteratively Regularized Gradient method
    for optimization with VI constraints, specialized to our bilevel setting with
    F = ∂g (hinge-loss subgradient) and f = ||w||_1/d.

    Update (full-vector block):
        x_{k+1} = Proj_X( x_k - γ_k F(x_k) + η_k ∂f(x_k) ),
    with averaging (weights γ_{k+1}^r):
        S_{k+1} = S_k + γ_{k+1}^r,
        x̄_{k+1} = (S_k x̄_k + γ_{k+1}^r x_{k+1}) / S_{k+1}.

    Stepsizes per Corollary 3.5 (bounded X):
        γ_k = γ0 / (k+1)^{1/2},  η_k = η0 / (k+1)^b, with 0 < b < 1/2, 0 ≤ r < 1.
    """

    def __init__(
        self,
        X,
        y,
        L,
        R,
        bound,
        initial,
        num_iter=1000,
        domain="box",
        eta0=1.0,
        b=0.25,
        r=0.5,
    ):
        # Data and dimensions
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape

        # Geometry / domain
        self.L = float(L)
        self.R = float(R)
        self.bound = float(bound)
        self.domain = domain  # 'box' or 'ball'
        self.num_iter = int(num_iter)

        # Stepsize hyperparameters (per Corollary 3.5)
        self.gamma0 = self.R / self.L
        self.eta0 = float(eta0)
        self.b = float(b)
        self.r = float(r)

        # State
        self.initial = self._proj(initial.copy())
        self.checkpoint = np.zeros(self.n_features + 1)

        # Histories
        self.f_plot = []  # evaluate f at averaged iterate x̄_k
        self.g_plot = []  # evaluate g at averaged iterate x̄_k
        self.x_hist = [None]  # raw iterates for potential inspection
        self.f_hist = [None]
        self.g_hist = [None]

    # Project onto domain
    def _proj(self, x):
        if self.domain == "box":
            return project_onto_box(x, self.bound)
        return project_onto_ball(x, self.bound)

    # Objective/subgrad wrappers
    def f_val(self, x):
        return l1_norm(x, self.n_features)

    def g_val(self, x):
        return hinge_loss(self.X, self.y, x)

    def subgrad_f(self, x):
        return subgradient_l1_norm(x, self.n_features)

    def F_map(self, x):
        # F = ∂g (hinge-loss subgradient)
        return subgradient_hinge_loss(self.X, self.y, x)

    # Stepsize schedules
    def gamma_k(self, k):
        return self.gamma0 / np.sqrt(k + 1.0)

    def eta_k(self, k):
        return self.eta0 / ((k + 1.0) ** self.b)

    def solve(self, start_iter=0, end_iter=None):
        if end_iter is None:
            end_iter = self.num_iter

        # Initialization (Algorithm 3.1)
        xk = self.initial.copy()
        xbar = xk.copy()
        Sk = (self.gamma0 ** self.r)

        # log initial
        self.x_hist.append(xk.copy())
        self.f_hist.append(self.f_val(xk))
        self.g_hist.append(self.g_val(xk))
        self.f_plot.append(self.f_val(xbar))
        self.g_plot.append(self.g_val(xbar))

        for k in tqdm(range(start_iter, end_iter)):
            # Compute mapping and subgradient
            Fk = self.F_map(xk)
            dfk = self.subgrad_f(xk)

            # Stepsizes at k
            gk = self.gamma_k(k)
            ek = self.eta_k(k)

            # Update x_{k+1}
            xk1 = self._proj(xk - gk * (Fk + ek * dfk))

            # Averaging weights use γ_{k+1}
            gk1 = self.gamma_k(k + 1)
            Sk1 = Sk + (gk1 ** self.r)
            xbar = (Sk * xbar + (gk1 ** self.r) * xk1) / Sk1

            # Shift
            xk = xk1
            Sk = Sk1

            # Log
            self.x_hist.append(xk.copy())
            self.f_hist.append(self.f_val(xk))
            self.g_hist.append(self.g_val(xk))
            self.f_plot.append(self.f_val(xbar))
            self.g_plot.append(self.g_val(xbar))

        self.checkpoint = xk.copy()
        return xbar
