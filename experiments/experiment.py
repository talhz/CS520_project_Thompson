import torch
from PGD import PGD
from PGD_nesterov import PGD_Nesterov
from penalty import Penalty
from augmented_Lagrange import Augmented_Lagragian
import numpy
import matplotlib.pyplot as plt

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


k, n = 3, 30
res_PGD, res_PGD_nes, res_penal, res_lag = [], [], [], []
X0 = torch.randn(n, k)
learner_PGD = PGD(f, X0)
learner_PGD_nes = PGD_Nesterov(f, X0)
learner_penal = Penalty(f, X0)
learner_lag = Augmented_Lagragian(f, X0)
ax = []
T = 1000
plt.figure(figsize=(10, 6))

for t in range(1, T):
    res_PGD.append(learner_PGD.train(t))
    res_PGD_nes.append(learner_PGD_nes.train(t))
    res_penal.append(learner_penal.train(t))
    res_lag.append(learner_lag.train(t))
    ax.append(t)
    plt.clf()
    colors = ['red', 'blue', 'purple', 'green']
    plt.plot(ax, res_PGD, color=colors[0], linestyle='--')
    plt.plot(ax, res_PGD_nes, color=colors[1], linestyle='--')
    plt.plot(ax, res_penal, color=colors[2], linestyle='--')
    plt.plot(ax, res_lag, color=colors[3], linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.legend(['PGD', 'PGD Nesterov', 'Penalty', 'Augmented Lagrangian'])
    plt.pause(0.05)
    plt.ioff()
    
plt.show()