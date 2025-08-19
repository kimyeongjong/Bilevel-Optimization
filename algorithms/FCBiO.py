import math
import time
import copy
from tqdm import trange
import numpy as np
from tqdm import tqdm

class ObjectiveFunctions:
    @staticmethod
    def l1_norm(x, n_features): # x = (w, b)
        """Compute the average L1-norm of the parameter vector."""
        return np.sum(np.abs(x[:-1])) / n_features

    @staticmethod
    def subgradient_l1_norm(x, n_features): # x = (w, b)
        """
        Compute the subgradient of the L1-norm.
        For each coordinate, the subgradient is 1 if x_i > 0, -1 if x_i < 0, and 0 if x_i == 0.
        """
        sub_grad = np.ones(x.shape, dtype=np.float32)
        sub_grad[x < 0.] = -1
        sub_grad[x == 0.] = 0.
        sub_grad[-1] = 0
        return sub_grad / n_features

    @staticmethod
    def hinge_loss(A_data, b_data, x): # x = (w, b)
        """
        Compute the average hinge loss.
        A_data: data matrix, shape (num_samples, dim_param)
        b_data: labels in {-1,1}, shape (num_samples,) or (num_samples,1)
        x: parameter vector, shape (dim_param,)
        """
        num_samples = A_data.shape[0]
        x = x.ravel()
        b_data = b_data.ravel()
        predictions = A_data @ x[:-1] + x[-1]
        margins = np.maximum(0, 1 - b_data * predictions)
        return np.sum(margins) / num_samples

    @staticmethod
    def subgradient_hinge_loss(A_data, b_data, x): # x = (w, b)
        """
        Compute the subgradient of the hinge loss.
        For samples with margin > 0 the subgradient is -b_data_i * a_i.
        """
        num_samples = A_data.shape[0]
        x = x.ravel()
        b_data = b_data.ravel() # [num_samples]
        predictions = A_data @ x[:-1] + x[-1] # [num_samples]
        margin = 1 - b_data * predictions # [num_samples]
        indicator = (margin > 0).astype(float) # [num_samples]
        grad = - (A_data.T.dot(indicator * b_data))
        grad = np.concatenate([grad, [-np.dot(indicator, b_data)]])
        return grad / num_samples

    @staticmethod
    def project_onto_ball(x, radius):
        """
        Project the vector x onto the L2 ball of given radius.
        """
        norm_x = np.linalg.norm(x, 2)
        if norm_x <= radius:
            return x
        else:
            return (radius / norm_x) * x
            
    @staticmethod
    def project_onto_hypercube(x, r):
        return np.clip(x, -r, r)
    
class FCBiO:
    def __init__(self, A_data, b_data, L, radius, initial_point, T, l, u, hat_opt_g_x, eps):
        """
        Initialize FC-BiO solver
        
        Parameters:
         - initial_point: Initial parameter vector (dim_param,)
         - T: Number of iterations
         - hat_opt_g_x: approximate solution for the lower-level function
         - l: global minimum of the upper-level function
         - u: the value of upper-level function at the solution of hat_opt_g_x
         - L: Lipschitz constant
         - eps: Guarantee the epsilon approximate solution
        """
        self.A_data = A_data
        self.b_data = b_data
        self.initial_point = initial_point.copy()
        self.T = T
        
        self.l = l
        self.u = u
        self.hat_opt_g_x = hat_opt_g_x # \hat{g}^* in the paper
        self.eps = eps
        self.num_samples = A_data.shape[0]
        self.dim_param = A_data.shape[1] + 1 # (w, b)
        self.L = L
        
        self.N = math.ceil(np.log2((self.u - self.l) / (self.eps / 2)))
        self.K = math.ceil(self.T / self.N)
        
        # Trajectories for iterates (x) and \eta_t
        self.x_trajectory_mean = np.zeros((self.N * self.K, self.dim_param), dtype=np.float32)
        # parameter trajectory during the outer loop
        self.x_trajectory_mean[0] = self.initial_point.copy()
        self.w_trajectory = None
        # parameter trajectory during the inner loop
        self.w_u_trajectory = None
        # u parameter trajectory in the inner loop
        
        # Projection radius is set as the norm of the initial point
        self.radius = radius
        
        # Stepsize for \eta_t
        D = 2 * self.radius * np.sqrt(A_data.shape[1] + 1)
        self.eta = D / (L * self.K**(1/2))
        
        # To store history of objective evaluations
        self.l1_norm_history = None
        self.hinge_loss_history = None
        
        
    def subgradient_psi(self, A_data, b_data, x, t): # x = (w, b)
        # tilde_g_x = g_x - hat_opt_g_x, hat_opt_g_x = g(hat_x_g)
        
        f_x = ObjectiveFunctions.l1_norm(x, self.num_samples)
        tilde_g_x = (ObjectiveFunctions.hinge_loss(A_data, b_data, x) - self.hat_opt_g_x) / 2
        # divide by 2 in order to set hat_opt_g_x as g_opt and make the FC-BiO's error epsilon.
        
        if f_x - t >= tilde_g_x: subgrad_psi = ObjectiveFunctions.subgradient_l1_norm(x, self.num_samples)
        elif f_x - t < tilde_g_x: subgrad_psi = ObjectiveFunctions.subgradient_hinge_loss(A_data, b_data, x) / 2
        # same here
        
        return subgrad_psi
        
    def solve(self):
    
        print("FC-BiO starts.")
        start_total = time.time()  # Runtime measurement start
    
        bar_x = copy.deepcopy(self.x_trajectory_mean[0])
    
        for n in range(self.N):
            iter_start = time.time()  # binary search iteration start time
            t = (self.l + self.u) / 2
            print(f"─────────────────────────────")
            print(f"[{n+1}/{self.N}] t = {(self.l + self.u) / 2:.6f}")
    
            # 초기화
            self.w = np.zeros(shape=(self.K, self.dim_param), dtype=np.float32)
            self.w[0] = copy.deepcopy(bar_x)
            self.w_u = np.zeros(shape=(self.K, self.dim_param), dtype=np.float32)
            self.w_u[0] = copy.deepcopy(bar_x)
    
            for k in trange(self.K - 1, desc=f"   Sub-iterations (n={n+1})", leave=False):
                s = self.subgradient_psi(self.A_data, self.b_data, self.w[k], t)
                self.w[k + 1] = ObjectiveFunctions.project_onto_hypercube(self.w[k] - self.eta * s, self.radius)
    
                s_u = self.subgradient_psi(self.A_data, self.b_data, self.w_u[k], self.u)
                self.w_u[k + 1] = ObjectiveFunctions.project_onto_hypercube(self.w_u[k] - self.eta * s_u, self.radius)
    
            self.w_u_mean = np.cumsum(self.w_u, axis=0) / (np.arange(1, len(self.w_u) + 1))[:, np.newaxis]
            self.x_trajectory_mean[n * self.K:(n + 1) * self.K] = copy.deepcopy(self.w_u_mean)
    
            hat_x_t = np.mean(self.w, axis=0)
            f_hat_x_t = ObjectiveFunctions.l1_norm(hat_x_t, self.num_samples)
            tilde_g_hat_x_t = (ObjectiveFunctions.hinge_loss(self.A_data, self.b_data, hat_x_t) - self.hat_opt_g_x) / 2
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
    
        # Trajectory 평가
        self.l1_norm_history = np.array([
            ObjectiveFunctions.l1_norm(self.x_trajectory_mean[i], self.dim_param)
            for i in range(self.T)
        ])
        self.hinge_loss_history = np.array([
            ObjectiveFunctions.hinge_loss(self.A_data, self.b_data, self.x_trajectory_mean[i])
            for i in range(self.T)
        ])
    
        return self.x_trajectory_mean, self.l1_norm_history, self.hinge_loss_history
        
