# Implementation of Penalty Method for Thompson Problem

import torch
import numpy as np
import matplotlib.pyplot as plt

class Penalty:
    def __init__(self, f, X0, lr=1e-3, steps=30, max_iters=500, tol=1e-7):
        """
        Initialization of augmented Lagragian algorithm 
        
        Args:
            f (function): Function to minimize, taking X as input.
            X0 (torch.Tensor): Initial guess for X, n x k matrix with each row having unit norm.
            lr (float): Learning rate for gradient descent.
            max_iters (int): Maximum number of iterations.
            tol (float): Tolerance for convergence based on relative change in f.
        """
        self.f = f
        self.X0 = X0
        self.lr = lr
        self.steps = steps
        self.max_iters = max_iters
        self.tol = tol
    
    def penalty(self, X):
        n = X.shape[0]
        sum = 0
        for i in range(n):
            sum += torch.abs(torch.norm(X[i])**2 - 1)
        return sum
    
    def objective(self, X, Mu):
        return self.f(X) + Mu * self.penalty(X)
    
    def constraint(self, X):
        """
        Define the constraint: diag(X*X^T) = e.
        """
        return torch.norm(torch.diagonal(self.X.detach() @ self.X.detach().t()) - torch.ones(self.X.detach().shape[0]), p=2)
    
    def train(self):
        self.X = self.X0.clone().double().requires_grad_(True)
        self.step = 0
        n = self.X.shape[0]
        self.Mu = 100
        self.f_val = self.objective(self.X, self.Mu)
        while self.step < self.steps:
            self.n_iters = 0
            while self.n_iters < self.max_iters:
            # Compute the gradient of f(X) w.r.t. X.
                grad_f = torch.autograd.grad(self.f_val, self.X)[0]
                # Check gradient TODO: only check gradient for certain iteration. 
                # torch.autograd.gradcheck(self.objective, inputs=[self.X, self.Lambda, self.Mu], eps=1e-4, atol=1e-2)
                self.X = self.X - self.lr * grad_f
                # Update the function value and check for convergence.
                f_old = self.f_val
                self.f_val = self.objective(self.X, self.Mu)
                if torch.norm(grad_f[0])**2 < self.tol:
                    break
                self.n_iters += 1
            self.Mu += 5
            self.step += 1
            print(self.f(self.X))
        return self.X.detach(), self.f_val.detach(), self.step
    
    def result(self):
        
        print("Optimal solution X:\n", self.X.detach())
        print("Value of f(X) at the optimal solution:", self.f(self.X))
        print("Number of iterations until convergence:", self.step)
        print("Mu", self.Mu)
        print("Constraint violation at the optimal solution:", self.constraint(self.X.detach()))
        
    def plotX(self, save=False):
        """
        For 3d case, plot X on unit sphere
        """
        if self.X.shape[1] != 3: 
            raise ValueError("Only support 3D plot!")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # plot the unit sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z, color='w', alpha = 0.3)

        # plot a point on the sphere
        for i in range(self.X.shape[0]):
            x = self.X.detach().numpy()[i]
            ax.scatter(x[0], x[1], x[2], color='r', s=100)

        plt.show()
        
## Example: Thompson Problem    

# Define the quadratic function to minimize.
if __name__ == "__main__":
    def f(X):
        n = X.shape[0]
        energy = 0
        for i in range(n):
            for j in range(i+1, n):
                dist = torch.norm(X[i,:] - X[j,:])
                energy += 1 / dist
        return energy

    # Generate some random data for X0.
    k, n = 3, 30
    X0 = torch.randn(n, k)
    learner = Penalty(f, X0)
    X_opt, f_val, n_iters = learner.train()
    learner.result()

    # 3d plot
    if k == 3:
        learner.plotX()