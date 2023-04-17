import torch
import numpy as np
from general_gd import GeneralGD

## Example: Thompson Problem    

# The energy is the negative minimal distance
def f(X):
    n = X.shape[0]
    energy = np.inf
    for i in range(n):
        for j in range(i+1, n):
            dist = torch.norm(X[i,:] - X[j,:])
            if dist < energy: 
                energy = dist
    return -energy

# Generate some random data for X0.
k, n = 3, 20
X0 = torch.randn(n, k)

learner = GeneralGD(f, X0)
X_opt, f_val, n_iters = learner.train()
learner.result()

# 3d plot
learner.plotX()
