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

For each $n = 2, 3, \dots, 10$, the distribution of electrons are plotted: 
| $n$ <img width=100/>| Image <img width =500/> | 
| --- |--- |
| $n = 2$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_2.png" alt="Thompson_2" width="500"/> |
| $n = 3$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_3.png" alt="Thompson_3" width="500"/> |
| $n = 4$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_4.png" alt="Thompson_4" width="500"/> |
| $n = 5$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_5.png" alt="Thompson_5" width="500"/> |
| $n = 6$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_6.png" alt="Thompson_6" width="500"/> |
| $n = 7$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_7.png" alt="Thompson_7" width="500"/> |
| $n = 8$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_8.png" alt="Thompson_8" width="500"/> |
| $n = 9$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_9.png" alt="Thompson_9" width="500"/> |
| $n = 10$ | <img src="https://github.com/talhz/CS520_project_Thompson/blob/main/figs/Thompson_10.png" alt="Thompson_10" width="500"/> |
