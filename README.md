# CS520_project_Thompson
This is the repository for CS520 project 1. In this repo, there are several algorithms that can solve Thompson-type problem. Current algorithms include [projected gradient descent (PGD)](https://github.com/talhz/CS520_project_Thompson/blob/main/Projected_GD/general_gd.py), [nesterov accelerated PGD](https://github.com/talhz/CS520_project_Thompson/blob/main/Projected_GD/PGD_nesterov.py), [penalty method](https://github.com/talhz/CS520_project_Thompson/blob/main/Projected_GD/penalty.py), [augmented Lagrange method](https://github.com/talhz/CS520_project_Thompson/blob/main/Projected_GD/augmented_Lagrange.py) and [stochastic gradient descent (SGD) method](https://github.com/talhz/CS520_project_Thompson/blob/main/Projected_GD/SGD.py).

## Week1 Update
Nelder-Mead Algorithm and Projected Gradient Descent are added.

## Week2 Update
We used the minimizer in cvxpy package to solve eq. (5) and (*) in the report.

## Week3 Update
We explored the projected gradient descent algorithm and found that it attained the theoretical infimum on a large scale of problems. In the meantime, it gives the coordinates of $\mathbf{X}$ automatically after the iteration, without conducting matrix decomposition. The following figure is the distribution of points on the 3-sphere for 2 to 30 points.
<p align='center'>
  <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/gif.gif?raw=true" alt="thompsonL2" width="90%"/>  
</p>

## Quick Example

```python3
def f(X):
    n = X.shape[0]
    energy = 0
    for i in range(n):
        for j in range(i+1, n):
            dist = torch.norm(X[i,:] - X[j,:])
            energy += 1 / dist
    return energy

k, n = 3, 30
X0 = torch.randn(n, k)
learner_PGD = PGD(f, X0)
T = 50
for t in range(1, T + 1):
  learner_PGD.train(t)
learner_PGD.result()
```
In each iteration, the learner returns current value of the enegy function `f(X)` and the result will be printed in the end. 

## Results

For each $n = 2, 3, \dots, 10$, the distribution of electrons are plotted. Check [result](https://github.com/talhz/CS520_project_Thompson/tree/main/figs/result) for more figures.
| $n$ <img width=100/>| Image <img width =500/> | 
| --- |--- |
| $n = 2$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_2.png?raw=true" alt="Thompson_2" width="500"/> |
| $n = 3$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_3.png?raw=true" alt="Thompson_3" width="500"/> |
| $n = 4$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_4.png?raw=true" alt="Thompson_4" width="500"/> |
| $n = 5$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_5.png?raw=true" alt="Thompson_5" width="500"/> |
| $n = 6$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_6.png?raw=true" alt="Thompson_6" width="500"/> |
| $n = 7$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_7.png?raw=true" alt="Thompson_7" width="500"/> |
| $n = 8$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_8.png?raw=true" alt="Thompson_8" width="500"/> |
| $n = 9$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_9.png?raw=true" alt="Thompson_9" width="500"/> |
| $n = 10$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_10.png?raw=true" alt="Thompson_10" width="500"/> |
| $n = 100$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/SGD_100.png?raw=true" alt="SGD_100" width="500"/> |
