import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

torch.set_printoptions(precision=12)

class PGD:
    def __init__(self, f, X0, lr=1e-2, max_iters=1000, tol=1e-7):
        """
        Initialization of projected gradient descent algorithm 
        
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
        self.f = f
        self.X0 = X0
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol
        self.X = self.X0.clone().requires_grad_(True)
        self.converge = False
        
    @staticmethod    
    def project_sphere(X):
        """Project each row of X onto a unit norm sphere."""
        norms = torch.norm(X, dim=1, keepdim=True)
        X = X / norms
        return X
    
    # def check_grad(self, grad, tol=1e-2):
    #     """
    #     Function to check gradient
        
    #     Args:
    #         eps (float): the gap in finite difference
    #         tol (float): Tolerance for finite difference and gradient
            
    #     Returns:
    #         res (Boolean): indicates if passes the check
    #     """
    #     n, k = self.X0.shape
    #     X = self.X.clone().detach()
    #     Xcopy = X.clone()
    #     finite_grad = torch.zeros((n, k))
    #     eps = np.sqrt(sys.float_info.epsilon * self.f_val.detach())
    #     print(self.f_val.detach(), eps)
    #     print("Conducting gradient check... tolerance", tol)
    #     for i in range(n):
    #         for j in range(k):
    #             X[i][j] += eps
    #             # print(X, self.X.detach())
    #             f_val = self.f(X)
    #             print(f_val, self.f_val.detach())
    #             diff = f_val - self.f_val.detach()
    #             finite_grad[i][j] = diff / eps
    #             # print(diff, diff / eps)
    #             X = Xcopy.clone()
                
        # if torch.norm(finite_grad - grad.detach()) < tol:
        #     print("Gradient Check Passed\n")
        # else:
        #     # print(finite_grad)
        #     # print(grad)
        #     print(torch.norm(finite_grad - grad.detach()))
        #     raise ValueError("Cannot pass gradient check!")
            
    
    def constraint(self):
        """
        Define the constraint: diag(X*X^T) = e.
        """
        return torch.norm(torch.diagonal(self.X.detach() @ self.X.detach().t()) - torch.ones(self.X.detach().shape[0]), p=2)

    def train(self, t):
        """
        Minimize f(X) subject to diag(X*X^T) = e using projected gradient descent.

        Args:
            f (function): Function to minimize, taking X as input.
            X0 (torch.Tensor): Initial guess for X, n x k matrix with each row having unit norm.
            lr (float): Learning rate for gradient descent.
            max_iters (int): Maximum number of iterations.
            tol (float): Tolerance for convergence based on relative change in f.

        Returns
        
        """
        self.f_val = self.f(self.X)
        if t < self.max_iters:
            if not self.converge:
                # Compute the gradient of f(X) w.r.t. X.
                grad_f = torch.autograd.grad(self.f_val, self.X)[0]
                # Check gradient
                # torch.autograd.gradcheck(self.f, inputs=self.X, eps=1e-2, atol=1e-2)
                
                # Project each row of X onto a unit norm sphere.
                self.X = self.project_sphere(self.X - self.lr * grad_f)
                # Update the function value and check for convergence.
                f_old = self.f_val
                self.f_val = self.f(self.X)
                if torch.abs((self.f_val - f_old) / f_old) < self.tol:
                    self.converge = True
                    self.n_iters = t
            return self.f_val.detach()
        return self.f_val.detach().numpy()
    
    def result(self):
        
        print("Optimal solution X:\n", self.X.detach())
        print("Value of f(X) at the optimal solution:", self.f_val.detach())
        print("Number of iterations until convergence:", self.n_iters)
        print("Constraint violation at the optimal solution:", self.constraint())
        
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
        
            
    



