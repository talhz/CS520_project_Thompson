import torch
from PGD import PGD

## Example: Thompson Problem    

# Define the quadratic function to minimize.
def f(X):
    n = X.shape[0]
    energy = 0
    for i in range(n):
        for j in range(i+1, n):
            dist = torch.norm(X[i,:] - X[j,:])
            energy += 1 / dist
    return energy

# Generate some random data for X0.
k, n = 3, 40
X0 = torch.randn(n, k)
learner = PGD(f, X0)
T = 2000
res = []
for t in range(1, T + 1):
    f_val = learner.train(t)
    print(f_val)
    res.append(f_val)
learner.result()

# 3d plot
if k == 3:
    learner.plotX()
