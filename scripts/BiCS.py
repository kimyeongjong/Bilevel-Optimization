import math
from collections import deque

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
    def __init__(
        self,
        X,
        y,
        Lg,
        L,
        R,
        bound,
        initial,
        num_iter=1000,
        eps=0.01,
        domain="box",
        mode="RL",
        beta=1.0,
    ):
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
        self.beta = beta
        self.num_iter = num_iter
        self.domain = domain.lower() if isinstance(domain, str) else domain
        self.eps = eps
        # Mode: 'RL' (know R & L), 'R' (know R only), 'N' (know none), 'ER' (unbounded domain)
        self.mode = mode.upper()
        self.initial = initial.copy()
        self.checkpoint = np.zeros(self.n_features + 1)
        self._reset_histories()
        # SOLO-FTRL state
        self._solo_initialized = False
        self._solo_c = 1.0
        self._solo_y1 = None
        self._solo_y_curr = None
        self._solo_accum = None
        self._solo_grad_sq = 0.0

    # --- Utility helpers -------------------------------------------------
    def _reset_histories(self):
        self.f_hist = [None]
        self.g_hist = [None]
        self.gy_hist = [np.inf]
        self.x_hist = [None]
        self.v_hist = [None]
        self.delta_hist = [np.inf]
        self.z_hist = [None]
        self.criterion = [None]
        self.f_plot = []
        self.g_plot = []
        self.gy_min_hist = []
        self._z_candidates = deque()
        self._z_all = deque()
        # Prefix sums for efficient N-mode ut(v) computation
        self.cum_vs = [np.zeros_like(self.initial, dtype=float)]  # Σ v_s up to index
        self.cum_vx = [0.0]  # Σ v_s^T x_s up to index

    def _proj(self, x):
        if self.domain == "box":
            return project_onto_box(x, self.bound)
        if self.domain == "ball":
            return project_onto_ball(x, self.bound)
        return x

    def subgrad_f(self, x):
        return subgradient_l1_norm(x, self.n_features)

    def subgrad_g(self, x):
        return subgradient_hinge_loss(self.X, self.y, x)

    def g_val(self, x):
        return hinge_loss(self.X, self.y, x)

    def f_val(self, x):
        return l1_norm(x, self.n_features)

    def _init_solo_state(self, y0):
        self._solo_c = math.sqrt(5.5) / self.R if self.mode in ("RL", "R") and self.R else 1.0
        self._solo_y1 = y0.copy()
        self._solo_y_curr = y0.copy()
        self._solo_accum = np.zeros_like(y0, dtype=float)
        self._solo_grad_sq = 0.0
        self._solo_initialized = True

    def _solo_current(self):
        if not self._solo_initialized:
            raise RuntimeError("SOLO-FTRL state is not initialized.")
        return self._solo_y_curr

    def _solo_update(self):
        if not self._solo_initialized:
            raise RuntimeError("SOLO-FTRL state is not initialized.")
        y_t = self._solo_y_curr
        grad = self.subgrad_g(y_t)
        self._solo_accum += grad
        self._solo_grad_sq += float(np.dot(grad, grad))
        denom = max(self._solo_c * self._solo_grad_sq, 1e-12)
        y_next = self._solo_y1 - self._solo_accum / denom
        self._solo_y_curr = self._proj(y_next)
        return grad

    def _support(self, vec):
        if self.domain == "box":
            bnd = np.broadcast_to(self.bound, vec.shape)
            return float(np.sum(np.abs(vec) * bnd))
        if self.domain == "ball":
            return float(self.bound * np.linalg.norm(vec))
        raise ValueError("Support function requested for unsupported domain")

    # --- Bounding functions ---------------------------------------------
    def _bound_u(self, t, v, G_prev):
        if self.mode == "RL":
            return 3.0 * self.L * self.R / math.sqrt(t)
        if self.mode == "R":
            term = self.beta * t + G_prev + float(np.dot(v, v))
            return 3.0 * self.R * math.sqrt(term) / t
        if self.mode == "N":
            if t <= 0:
                return 0.0
            k_t = (t + 1) // 2
            # Use prefix sums to get Σ_{s=k(t)}^{t-1} v_s and Σ v_s^T x_s in O(1)
            # cum arrays are defined so that index i stores sum up to i
            sum_vs = self.cum_vs[t - 1] - self.cum_vs[k_t - 1]
            sum_vx = self.cum_vx[t - 1] - self.cum_vx[k_t - 1]
            xt = self.x_hist[t]
            if xt is None:
                xt = self.x_hist[-1]
            # constant part: Σ v_s^T x_s + v^T x_t
            support_const = float(sum_vx + float(np.dot(v, xt)))
            # vector part inside support function: Σ v_s + v
            accum_vec = sum_vs + v
            support_val = self._support(-accum_vec) if self.bound is not None else 0.0
            max_term = support_const + support_val
            return max(2.0 * max_term / t, 1.0 / math.sqrt(t))
        if self.mode == "ER":
            return self.eps / 2.0
        raise ValueError(f"Unsupported mode '{self.mode}' for bound computation")

    def _step_size(self, t, Gsq):
        if self.mode == "RL":
            return self.R / math.sqrt(max(Gsq, 1e-16))
        if self.mode == "R":
            return self.R / math.sqrt(self.beta * t + max(Gsq, 1e-16))
        if self.mode == "N":
            return 1.0 / math.sqrt(max(Gsq, 1e-16))
        if self.mode == "ER":
            return None  # handled separately
        raise ValueError(f"Unsupported mode '{self.mode}' for step size")

    def _update_z_queues(self, t, chose_f):
        k_t = (t + 1) // 2
        while self._z_candidates and self._z_candidates[0] < k_t:
            self._z_candidates.popleft()
        while self._z_all and self._z_all[0] < k_t:
            self._z_all.popleft()

        f_t = self.f_hist[t]
        while self._z_all and self.f_hist[self._z_all[-1]] >= f_t:
            self._z_all.pop()
        self._z_all.append(t)

        if chose_f:
            while self._z_candidates and self.f_hist[self._z_candidates[-1]] >= f_t:
                self._z_candidates.pop()
            self._z_candidates.append(t)

        if self._z_candidates:
            return self._z_candidates[0]
        if self._z_all:
            return self._z_all[0]
        return t

    # --- Solvers ---------------------------------------------------------
    def solve(self, start_iter, end_iter):
        if self.mode == "ER":
            self._solve_er(start_iter, end_iter)
        else:
            self._solve_bounded(start_iter, end_iter)
        return self.checkpoint

    def _solve_bounded(self, start_iter, end_iter):
        x = self.initial.copy()
        y0 = self._proj(self.initial.copy())
        self._init_solo_state(y0)
        g_y_best = self.g_val(self._solo_current())

        self._z_candidates.clear()
        self._z_all.clear()

        Gsq = 0.0
        delta_prev = math.inf

        for t in tqdm(range(start_iter, end_iter + 1), desc=f"Bi-CS-{self.mode} Progress"):
            self.x_hist.append(x.copy())
            f_xt = self.f_val(x)
            g_xt = self.g_val(x)
            self.f_hist.append(f_xt)
            self.g_hist.append(g_xt)

            y_curr = self._solo_current()
            g_y_curr = self.g_val(y_curr)
            if g_y_curr <= g_y_best:
                g_y_best = g_y_curr
            self.gy_min_hist.append(g_y_best)
            self.gy_hist.append(g_y_curr)
            _ = self._solo_update()

            # Bounding function evaluations
            v_f = self.subgrad_f(x)
            G_prev = Gsq
            u_f = self._bound_u(t, v_f, G_prev)
            delta_tmp = min(delta_prev, u_f)
            chose_f = True

            if g_xt > g_y_best + delta_tmp:
                v_g = self.subgrad_g(x)
                u_g = self._bound_u(t, v_g, G_prev)
                gap = max(0.0, g_xt - g_y_best)
                delta_tmp = min(delta_prev, u_g, gap)
                if g_xt <= g_y_best + delta_tmp:
                    v = v_f
                    chose_f = True
                else:
                    v = v_g
                    chose_f = False
            else:
                v = v_f

            self.delta_hist.append(delta_tmp)
            self.criterion.append(chose_f)
            self.v_hist.append(v.copy())
            # Update prefix sums for N-mode ut: add current v_t, x_t
            self.cum_vs.append(self.cum_vs[-1] + v)
            self.cum_vx.append(self.cum_vx[-1] + float(np.dot(v, self.x_hist[t])))

            # Maintain z_t in O(T) via monotone queues
            z_idx = self._update_z_queues(t, chose_f)
            z = self.x_hist[z_idx]
            self.z_hist.append(z.copy())
            self.f_plot.append(self.f_val(z))
            self.g_plot.append(self.g_val(z))

            # Step-size update and primal step
            Gsq += float(np.dot(v, v))
            eta = self._step_size(t, Gsq)
            x = x - eta * v
            x = self._proj(x)

            delta_prev = delta_tmp

        self.checkpoint = x.copy()

    def _solve_er(self, start_iter, end_iter):
        x = self.initial.copy()
        y0 = self._proj(self.initial.copy())
        self._init_solo_state(y0)
        g_y_best = self.g_val(self._solo_current())

        delta_fixed = self.eps / 2.0
        total_iters = end_iter
        last_tau = None
        current_episode = None
        episode_best = None
        episode_all_best = None

        for t in tqdm(range(start_iter, total_iters + 1), desc="Bi-CS-ER Progress"):
            episode_idx = int(math.floor(math.log2(t))) + 1 if t > 0 else 1
            if current_episode is None:
                current_episode = episode_idx
                episode_start = 2 ** (current_episode - 1)
                episode_end = 2 ** current_episode - 1
                episode_best = None
                episode_all_best = None
            elif episode_idx != current_episode:
                tau = episode_best if episode_best is not None else episode_all_best
                if tau is None:
                    tau = t - 1
                last_tau = tau
                current_episode = episode_idx
                episode_start = 2 ** (current_episode - 1)
                episode_end = 2 ** current_episode - 1
                episode_best = None
                episode_all_best = None
                x = self.initial.copy()

            self.x_hist.append(x.copy())
            f_xt = self.f_val(x)
            g_xt = self.g_val(x)
            self.f_hist.append(f_xt)
            self.g_hist.append(g_xt)

            y_curr = self._solo_current()
            g_y_curr = self.g_val(y_curr)
            if g_y_curr <= g_y_best:
                g_y_best = g_y_curr
            self.gy_min_hist.append(g_y_best)
            self.gy_hist.append(g_y_best)
            _ = self._solo_update()

            if g_xt <= g_y_best + delta_fixed:
                v = self.subgrad_f(x)
                chose_f = True
            else:
                v = self.subgrad_g(x)
                chose_f = False

            self.delta_hist.append(delta_fixed)
            self.criterion.append(chose_f)
            self.v_hist.append(v.copy())

            if episode_all_best is None or f_xt < self.f_hist[episode_all_best]:
                episode_all_best = t
            if chose_f:
                if episode_best is None or f_xt < self.f_hist[episode_best]:
                    episode_best = t

            z_idx = last_tau if last_tau is not None else t
            z = self.x_hist[z_idx]
            self.z_hist.append(z.copy())
            self.f_plot.append(self.f_val(z))
            self.g_plot.append(self.g_val(z))

            norm_sq = float(np.dot(v, v))
            eta = self.eps / (2.0 * max(norm_sq, 1e-12))
            x = x - eta * v
            x = self._proj(x)

        self.checkpoint = x.copy()
