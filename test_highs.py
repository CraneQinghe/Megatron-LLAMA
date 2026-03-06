import cvxpy as cp
import numpy as np

n = 5
m = 10
Throughput = np.random.rand(n, m)
prior = np.random.rand(n, m)
x = cp.Variable(shape=(n * m,), boolean=True)
x_reshaped = cp.reshape(x, (n, m))
objective = cp.Maximize(cp.sum(cp.multiply(cp.multiply(Throughput, x_reshaped), prior)))
constraints = [cp.sum(x_reshaped, axis=1) <= 1]

problem = cp.Problem(objective, constraints)
# Try standard HIGHS opts
solver_opts = {
    "time_limit": 10,
    "mip_rel_gap": 0.05
}
try:
    problem.solve(solver=cp.HIGHS, **solver_opts)
    print("Success with HIGHS")
    print("Status:", problem.status)
    print("Value:", problem.value)
except Exception as e:
    print("Failed with HIGHS standard opts:", e)
    # Try without opts
    try:
        problem.solve(solver=cp.HIGHS)
        print("Success with HIGHS (no opts)")
    except Exception as e2:
        print("Failed with HIGHS completely:", e2)
