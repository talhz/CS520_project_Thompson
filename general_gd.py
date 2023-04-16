import torch

def project_sphere(X):
    """Project each row of X onto a unit norm sphere."""
    norms = torch.norm(X, dim=1, keepdim=True)
    X = X / norms
    return X

def projected_gradient_descent(f, X0, lr=1e-2, max_iters=1000, tol=1e-6):
    """
    Minimize f(X) subject to diag(X*X^T) = e using projected gradient descent.

    Args:
        f (function): Function to minimize, taking X as input.
        X0 (torch.Tensor): Initial guess for X, n x k matrix with each row having unit norm.
        lr (float): Learning rate for gradient descent.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for convergence based on relative change in f.

    Returns:
        X (torch.Tensor): Optimal solution, n x k matrix with each row having unit norm.
        f_val (float): Value of f(X) at the optimal solution.
        n_iters (int): Number of iterations until convergence.
    """
    X = X0.clone().requires_grad_(True)
    n_iters = 0
    f_val = f(X)
    while n_iters < max_iters:
        # Compute the gradient of f(X) w.r.t. X.
        grad_f = torch.autograd.grad(f_val, X)[0]
        # Project each row of X onto a unit norm sphere.
        X = project_sphere(X - lr * grad_f)
        # Update the function value and check for convergence.
        f_old = f_val
        f_val = f(X)
        if torch.abs((f_val - f_old) / f_old) < tol:
            break
        n_iters += 1
    return X.detach(), f_val.detach(), n_iters

# Define the quadratic function to minimize.
def f(X):
    n = X.shape[0]
    energy = 0
    for i in range(n):
        for j in range(i+1, n):
            dist = torch.norm(X[i,:] - X[j,:])**2
            energy += 1 / dist
    return energy*2

# Generate some random data for X0.
k, n = 10, 10
X0 = torch.randn(n, k)

# Define the constraint: diag(X*X^T) = e.
def constraint(X):
    return torch.norm(torch.diagonal(X @ X.t()) - torch.ones(n), p=2)

# Minimize f(X) subject to the constraint using projected gradient descent.
X_opt, f_val, n_iters = projected_gradient_descent(lambda X: f(X) + 1e6 * constraint(X), X0)

# Print the results.
print("Optimal solution X:\n", X_opt)
print("Value of f(X) at the optimal solution:", f_val)
print("Number of iterations until convergence:", n_iters)
print("Constraint violation at the optimal solution:", constraint(X_opt))
