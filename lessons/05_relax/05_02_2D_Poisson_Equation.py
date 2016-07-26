from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

from laplace_helper import L2_rel_error, plot_3D


# Grid parameters
nx = 41
ny = 41
xmin = 0
xmax = 1
ymin = -0.5
ymax = 0.5

l2_target = 2e-7


def poisson_IG(nx, ny, xmax, xmin, ymax, ymin):
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    # Mesh
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    X, Y = np.meshgrid(x, y)

    # Source
    L = xmax - xmin
    b = -2 * (np.pi / L)**2 * np.sin(np.pi * X / L) * np.cos(np.pi * Y / L)

    # Initialize
    p_i = np.zeros((ny, nx))

    return X, Y, x, y, p_i, b, dx, dy, L


def poisson_2d(p, b, dx, dy, l2_target):
    l2_norm = 1
    iterations = 0
    l2_conv = []

    while l2_norm > l2_target:
        pd = p.copy()

        p[1: -1, 1:-1] = (1 / (2 * (dx**2 + dy**2)) *
                          ((pd[1: -1, 2:] + pd[1: -1, :-2]) * dy**2 +
                           (pd[2:, 1: -1] + pd[:-2, 1: -1]) * dx**2 -
                           b[1: -1, 1: -1] * dx**2 * dy**2))

        l2_norm = L2_rel_error(pd, p)
        iterations += 1
        l2_conv.append(l2_norm)

    print('Number of Jacobi iterations: {0:d}'.format(iterations))
    return p, l2_conv


X, Y, x, y, p_i, b, dx, dy, L = poisson_IG(nx, ny, xmax, xmin, ymax, ymin)
plot_3D(x, y, p_i)
plt.show()

p, l2_conv = poisson_2d(p_i.copy(), b, dx, dy, l2_target)
plot_3D(x, y, p)
plt.show()


def p_analytical(X, Y, L):
    return np.sin(X * np.pi / L) * np.cos(Y * np.pi / L)

p_an = p_analytical(X, Y, L)

error = L2_rel_error(p, p_an)

# Algebraic convergence
plt.figure(figsize=(8, 6))
plt.grid(True)
plt.xlabel(r'iterations', fontsize=18)
plt.ylabel(r'$L_2$-norm', fontsize=18)
plt.semilogy(np.arange(len(l2_conv)), l2_conv, lw=2, color='k')


# Spatial convergence
# Checking if we are in second orther convergence in space
nx_values = [11, 21, 41, 81]

error = np.zeros_like(nx_values, dtype=np.float)

for i, nx in enumerate(nx_values):
    ny = nx

    X, Y, c, y, p_i, b, dx, dy, L = poisson_IG(nx, ny, xmax, xmin, ymax, ymin)

    p, l2_conv = poisson_2d(p_i.copy(), b, dx, dy, l2_target)

    p_an = p_analytical(X, Y, L)

    error[i] = L2_rel_error(p, p_an)


plt.figure(figsize=(6, 6))
plt.grid(True)
plt.xlabel(r'$n_x$', fontsize=18)
plt.ylabel(r'$L_2$-norm of the error', fontsize=18)
plt.loglog(nx_values, error, color='k', ls='--', lw=2, marker='o')
plt.axis('equal')

plt.show()
