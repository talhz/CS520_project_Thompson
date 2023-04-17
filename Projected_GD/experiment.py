import torch
from general_gd import GeneralGD

torch.manual_seed(0)
torch.set_printoptions(precision=9)

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


k = 3
res = []
for n in range(2, 31):
    X0 = torch.randn(n, k)
    
    learner = GeneralGD(f, X0)
    X_opt, f_val, n_iters = learner.train()
    res.append(f_val)

print(res)
