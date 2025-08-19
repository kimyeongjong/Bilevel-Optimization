import numpy as np
import gurobipy as gp
from gurobipy import GRB

def hinge_loss_minimize(X, y, bound = GRB.INFINITY):
    assert X.shape[0] == y.shape[0], "Dimensions (sample numbers) of X and y do not coincide."
    n_samples, n_features = X.shape[0], X.shape[1]
    
    # Create Gurobi model
    model = gp.Model("Hinge_loss_minimization")
    print(model.ModelName)
    model.setParam("OutputFlag", 1)
    
    # Variables: weight vector w, bias b and slack variables ξ_i
    w = model.addVars(n_features, lb=-bound, ub=bound, name="w")
    b = model.addVar(lb=-bound, ub=bound, name="b")
    xi = model.addVars(n_samples, lb=0.0, name="xi")
    
    # Constraints: hinge loss slack for each sample
    for i in range(n_samples):
        xi_expr = 1 - y[i] * (gp.quicksum(w[j] * X[i, j] for j in X[i].indices) + b)
        model.addConstr(xi[i] >= xi_expr, name=f"hinge_constr_{i}")
    
    # Objective: Minimize average hinge loss
    model.setObjective((1.0 / n_samples) * gp.quicksum(xi[i] for i in range(n_samples)), GRB.MINIMIZE)
    
    # Solve
    model.optimize()
    print('Used algorithm: ', model.Params.Method)
    
    # Retrieve results
    w_opt = np.array([w[j].X for j in range(n_features)])
    b_opt = b.X
    xi_opt = np.array([xi[j].X for j in range(n_samples)])
    loss_value = model.ObjVal
    
    print(f"w_opt: {w_opt} b_opt: {b_opt} xi_opt: {xi_opt} Minimum hinge loss: {loss_value:.4f}")
    return w_opt, b_opt, loss_value


def L1_norm_second_minimize(X, y, g_opt, bound = GRB.INFINITY):
    assert X.shape[0] == y.shape[0], "Dimensions (sample numbers) of X and y do not coincide."
    n, d = X.shape[0], X.shape[1]

    
    m = gp.Model('L1_norm_minimize_among_hinge_loss_minimizers')
    print(m.ModelName)
    m.setParam("OutputFlag", 1)
    
    w = m.addVars(d, lb=-bound, ub=bound, name='w')
    u = m.addVars(d, lb=0.0, name='u') # u_i = |w_i|
    b = m.addVar(lb=-bound, ub=bound, name='b')
    xi = m.addVars(n, lb=0.0, name='xi')

    # Constraints: u_j >= |w_j|
    for j in range(d):
        m.addConstr(w[j] <= u[j], name=f"abs_pos_{j}")
        m.addConstr(-w[j] <= u[j], name=f"abs_neg_{j}")

    # Constraints: hinge loss slack for each sample xi[i] >= 1- y[i] * (w \cdot x[i] + b)
    for i in range(n):
        xi_expr = 1 - y[i] * (gp.quicksum(w[j] * X[i, j] for j in X[i].indices) + b)
        m.addConstr(xi[i] >= xi_expr, name=f"hinge_constr_{i}")

    # Hinge loss <= g_opt
    hinge_loss_expr = (1.0 / n) * gp.quicksum(xi[i] for i in range(n))
    m.addConstr(hinge_loss_expr <= g_opt, name="hinge_loss_upper_bound")

    # L1 norm
    m.setObjective(gp.quicksum(u[i] for i in range(d)) / d, GRB.MINIMIZE)

    # Solve
    m.optimize()
    #print('Constraints: ', m.getConstrs())
    print('Used algorithm: ', m.Params.Method)
    
    # Retrieve results
    w_opt = np.array([w[j].X for j in range(d)])
    b_opt = b.X
    xi_opt = np.array([xi[j].X for j in range(n)])
    u_opt = np.array([u[j].X for j in range(d)])
    loss_value = m.ObjVal

    print(f"w_opt: {w_opt} b_opt: {b_opt} xi_opt: {xi_opt} u_opt: {u_opt} Minimum L1 norm: {loss_value:.4f}")
    return w_opt, b_opt, xi_opt, u_opt, loss_value

if __name__ == "__main__":
    import os
    # Gurobi version check
    print("Gurobi version:", gp.gurobi.version())

    # Gurobi module path
    print("gurobipy module path:", os.path.dirname(gp.__file__))

    # License check
    print("Checking Gurobi license...")
    try:
        m = gp.Model("test_model")
        x = m.addVar(name="x")
        m.setObjective(x * x, gp.GRB.MINIMIZE)
        m.optimize()
        print("✅ Gurobi is working with license!")
    except gp.GurobiError as e:
        print("❌ Error:", e)