
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

from laplace_helper import p_analytical, plot_3D, L2_rel_error

import numba
from numba import jit


nx = 128
ny = 128

L = 5
H = 5

x = np.linspace(0, L, nx)
y = np.linspace(0, H, ny)

dx = L / (nx - 1)
dy = H / (ny - 1)

p0 = np.zeros((ny, nx))

p0[-1, :] = np.sin(1.5 * np.pi * x / x[-1])


def laplace2d(p, l2_target):
    l2norm = 1
    pn = np.empty_like(p)
    iterations = 0

    while l2norm > l2_target:
        pn = p.copy()
        p[1: -1, 1: -1] = 0.25 * (pn[1: -1, 2:] + pn[1: -1, :-2] +
                                  pn[2:, 1: -1] + pn[:-2, 1: -1])

        # Neumann BC
        p[1:-1, -1] = 0.25 * (2 * pn[1:-1, -2] + pn[2:, -1] + pn[:-2, -1])

        l2norm = np.sqrt(np.sum((p - pn)**2) / np.sum(pn**2))
        iterations += 1

    return p, iterations

l2_target = 1e-8
p, iterations = laplace2d(p0.copy(), l2_target)

print("Jacobi method took {} iterations at tolerance {}".
      format(iterations, l2_target))


@jit(nopython=True)
def laplace2d_jacobi(p, pn, l2_target):
    iterations = 0
    iter_diff = l2_target + 1  # init iter_diff to be larger than l2_target
    denominator = 0.0
    ny, nx = p.shape
    l2_diff = np.zeros(20000)  # Numba doesn't handle mutable objects

    while iter_diff > l2_target:
        for j in range(ny):
            for i in range(nx):
                pn[j, i] = p[j, i]

        iter_diff = 0.0
        denominator = 0.0

        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                p[j, i] = .25 * (pn[j, i - 1] + pn[j, i + 1] +
                                 pn[j - 1, i] + pn[j + 1, i])

        # Neumann 2nd-order BC
        for j in range(1, ny - 1):
            p[j, -1] = .25 * (2 * pn[j, -2] + pn[j + 1, -1] + pn[j - 1, -1])

        for j in range(ny):
            for i in range(nx):
                iter_diff += (p[j, i] - pn[j, i])**2
                denominator += (pn[j, i] * pn[j, i])

        iter_diff /= denominator
        iter_diff = iter_diff**0.5
        l2_diff[iterations] = iter_diff
        iterations += 1

    return p, iterations, l2_diff


p, iterations, l2_diffJ = laplace2d_jacobi(p0.copy(), p0.copy(), 1e-8)

print("Numba Jacobi method took {} iterations at tolerance {}".format(
    iterations, l2_target))


@jit(nopython=True)
def laplace2d_gauss_seidel(p, pn, l2_target):
    iterations = 0
    iter_diff = l2_target + 1  # initialize iter_diff to be larger than l2_target
    denominator = 0.0
    ny, nx = p.shape
    l2_diff = np.zeros(20000)

    while iter_diff > l2_target:
        for j in range(ny):
            for i in range(nx):
                pn[j, i] = p[j, i]

        iter_diff = 0.0
        denominator = 0.0

        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                p[j, i] = .25 * (p[j, i - 1] + p[j, i + 1] +
                                 p[j - 1, i] + p[j + 1, i])

        # Neumann 2nd-order BC
        for j in range(1, ny - 1):
            p[j, -1] = .25 * (2 * p[j, -2] + p[j + 1, -1] + p[j - 1, -1])

        for j in range(ny):
            for i in range(nx):
                iter_diff += (p[j, i] - pn[j, i])**2
                denominator += (pn[j, i] * pn[j, i])

        iter_diff /= denominator
        iter_diff = iter_diff**0.5
        l2_diff[iterations] = iter_diff
        iterations += 1

    return p, iterations, l2_diff

p, iterations, l2_diffGS = laplace2d_gauss_seidel(p0.copy(), p0.copy(), 1e-8)

print("Numba Gauss-Seidel method took {} iterations at tolerance {}".format(iterations, l2_target))


@jit(nopython=True)
def laplace2d_SOR(p, pn, l2_target, omega):

    iterations = 0
    # initialize iter_diff to be larger than l2_target
    iter_diff = l2_target + 1
    denominator = 0.0
    ny, nx = p.shape
    l2_diff = np.zeros(20000)

    while iter_diff > l2_target:
        for j in range(ny):
            for i in range(nx):
                pn[j, i] = p[j, i]

        iter_diff = 0.0
        denominator = 0.0

        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                p[j, i] = ((1 - omega) * p[j, i] + omega * .25 *
                           (p[j, i - 1] + p[j, i + 1] +
                            p[j - 1, i] + p[j + 1, i]))

        # Neumann 2nd-order BC
        for j in range(1, ny - 1):
            p[j, -1] = .25 * (2 * p[j, -2] + p[j + 1, -1] + p[j - 1, -1])

        for j in range(ny):
            for i in range(nx):
                iter_diff += (p[j, i] - pn[j, i])**2
                denominator += (pn[j, i] * pn[j, i])

        iter_diff /= denominator
        iter_diff = iter_diff**0.5
        l2_diff[iterations] = iter_diff
        iterations += 1

    return p, iterations, l2_diff


l2_target = 1e-8
omega = 1.5

p, iterations, l2_diffSOR = laplace2d_SOR(
    p0.copy(), p0.copy(), l2_target, omega)

s = "Numba SOR method took {} iterations at tolerance {} with omega = {}"

print(s.format(iterations, l2_target, omega))


l2_target = 1e-8
omega = 2. / (1 + np.pi / nx)
p, iterations, l2_diffSORopt = laplace2d_SOR(
    p0.copy(), p0.copy(), l2_target, omega)

print("Numba SOR method took {} iterations\
 at tolerance {} with omega = {:.4f}".format(iterations, l2_target, omega))


plt.figure(figsize=(8, 8))
plt.xlabel(r'iterations', fontsize=18)
plt.ylabel(r'$L_2$-norm', fontsize=18)
plt.semilogy(np.trim_zeros(l2_diffJ, 'b'),
             'k-', lw=2, label='Jacobi')
plt.semilogy(np.trim_zeros(l2_diffGS, 'b'),
             'k--', lw=2, label='Gauss-Seidel')
plt.semilogy(np.trim_zeros(l2_diffSOR, 'b'),
             'g-', lw=2, label='SOR')
plt.semilogy(np.trim_zeros(l2_diffSORopt, 'b'),
             'g--', lw=2, label='Optimized SOR')
plt.legend(fontsize=16)

plt.show()
