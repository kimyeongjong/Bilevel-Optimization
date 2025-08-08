import gurobipy as gp
import os










if __name__ == "__main__":
    # version check
    print("Gurobi version:", gp.gurobi.version())

    # gurobipy module path
    print("gurobipy module path:", os.path.dirname(gp.__file__))

    # license check
    try:
        m = gp.Model("test_model")
        x = m.addVar(name="x")
        m.setObjective(x * x, gp.GRB.MINIMIZE)
        m.optimize()
        print("✅ Gurobi is working with license!")
    except gp.GurobiError as e:
        print("❌ Error:", e)