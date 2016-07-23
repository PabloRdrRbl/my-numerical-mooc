import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import solve

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

def generateMatrix(N, sigma):
    # Diagonal
    d = 2 * np.diag(np.ones(N - 2) * (1 + 1 / sigma))

    # Neumann BC
    d[-1, -1] = 1 + 2 / sigma

    # Upper diagonal
    ud = np.diag(np.ones(N - 3) * -1, 1)

    # Lower diagonal
    ld = np.diag(np.ones(N - 3) * -1, -1)

    A = d + ud + ld

    return A


def generateRHS(T, sigma):
    b = T[1: -1] * 2 * (1 / sigma - 1) + T[:-2] + T[2:]

    # Dirichlet BC
    b[0] += T[0]

    return b

def CrankNicolson(T, A, nt, sigma):
    for t in range(nt):
        Tn = T.copy()
        b = generateRHS(Tn, sigma)

        T_interior = solve(A, b)
        T[1: -1] = T_interior

        # Neumann BC
        T[-1] = T[-2]

    return T


L = 1
nx = 21
alpha = 1.22e-3

dx = L / (nx - 1)

Ti = np.zeros(nx)
Ti[0] = 100

sigma = 0.5
dt = sigma * dx * dx / alpha
nt = 10

A = generateMatrix(nx, sigma)

T = CrankNicolson(Ti.copy(), A, nt, sigma)

# Plot
x = np.linspace(0, L, nx)

plt.plot(x, T, color='#003366', ls='-', lw=3)

plt.show()
