"""Sanity checks for stats testing.

"""

import numpy as np
from functions import calc_and_plot

N = np.random.normal

# Simulation 1
x = N(size=2000);        y = N(size=2000)
X = N(0, 0.1, size=200); Y = N(0, 0.1, size=200)
calc_and_plot(x, y, X, Y, W=5, concat=False, filename="plots/stats-01.png")

# Simulation 2
X = N(2, 0.1, size=200); Y = N(2, 0.1, size=200)
calc_and_plot(x, y, X, Y, W=5, concat=False, filename="plots/stats-02.png")

# Simulation 3
X = N(3, 0.1, size=200); Y = N(3, 0.1, size=200)
calc_and_plot(x, y, X, Y, W=5, concat=False, filename="plots/stats-03.png")

# Simulation 4
X = N(0, 0.1, size=200); Y = N(0.5*X, 0.1, size=200)
calc_and_plot(x, y, X, Y, W=5, concat=False, filename="plots/stats-04.png")

# Simulation 5
X = N(2, 0.1, size=200); Y = N(0.5*(X-2) + 2, 0.1, size=200)
calc_and_plot(x, y, X, Y, W=5, concat=False, filename="plots/stats-05.png")

# Simulation 6
X = N(3, 0.1, size=200); Y = N(0.5*(X-3) + 3, 0.1, size=200)
calc_and_plot(x, y, X, Y, W=5, concat=False, filename="plots/stats-06.png")

# Simulation 7
x = N(size=2000);        y = N(0.5*x, 0.1, size=2000)
X = N(0, 0.1, size=200); Y = N(0, 0.1, size=200)
calc_and_plot(x, y, X, Y, W=5, concat=False, filename="plots/stats-07.png")

# Simulation 8
X = N(2, 0.1, size=200); Y = N(1, 0.1, size=200)
calc_and_plot(x, y, X, Y, W=5, concat=False, filename="plots/stats-08.png")

# Simulation 9
X = N(3, 0.1, size=200); Y = N(1.5, 0.1, size=200)
calc_and_plot(x, y, X, Y, W=5, concat=False, filename="plots/stats-09.png")

# Simulation 10
X = N(0, 0.1, size=200); Y = N(0.5*X, 0.1, size=200)
calc_and_plot(x, y, X, Y, W=5, concat=False, filename="plots/stats-10.png")

# Simulation 11
X = N(2, 0.1, size=200); Y = N(0.5*X, 0.1, size=200)
calc_and_plot(x, y, X, Y, W=5, concat=False, filename="plots/stats-11.png")

# Simulation 12
X = N(3, 0.1, size=200); Y = N(0.5*X, 0.1, size=200)
calc_and_plot(x, y, X, Y, W=5, concat=False, filename="plots/stats-12.png")
