# Use finite difference to check gradient
import torch

def gradient_check(f, x, g):
    n, k = x.shape[0], x.shape[1]
    fx = f(x) # computed function 
    gx = g(x) # putative gradient
    
    h = 1e-6
    xi = x.clone().detach()
    gxd = gx.clone().detach() 
    for i in range(n):
        for j in range(k):
            xi[i, j] += h
            gxd[i, j] = (f(xi) - fx) / h
            xi[i, j] = x[i, j]
            
    absdiff = torch.abs(gxd - gx)
    return {'g': gx, 'gfd': gxd, 
            'maxdiff': torch.max(absdiff), 
            'normdiff': torch.norm(gxd - gx)}

# Gradient check for Thompson problem 
def f(X):
    n = X.shape[0]
    energy = 0
    for i in range(n):
        for j in range(i+1, n):
            dist = torch.norm(X[i,:] - X[j,:])**2
            energy += 1 / dist
    return energy*2

def g(X):
    n = X.shape[0]
    k = X.shape[1]
    res = torch.zeros((n, k))
    for i in range(n):
        s = 0
        for j in range(n):
            if i != j:
                s += (X[j,:] - X[i,:]) / torch.norm(X[i,:] - X[j,:])**4
        res[i, :] = -2 * s
    return res

k, n = 1, 5
X0 = torch.randn(n, k)
print(gradient_check(lambda x: x.T@x, X0, lambda x: 2 * x))

k, n = 3, 3
X0 = torch.randn(n, k)
print(gradient_check(f, X0, g))