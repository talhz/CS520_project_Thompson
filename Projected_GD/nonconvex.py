import torch
from PGD import PGD

## Example: Nonconvex Problem (Not stable)

# Define the log-distance function to minimize.
def f(X):
    n = X.shape[0]
    energy = 0
    for i in range(n):
        for j in range(i+1, n):
            dist = torch.log(torch.norm(X[i,:] - X[j,:]))
            energy += dist
    return energy*2

# Generate some random data for X0.
k, n = 3, 10
X0 = torch.randn(n, k)

learner = PGD(f, X0)
X_opt, f_val, n_iters = learner.train()
learner.result()

# 3d plot
learner.plotX()