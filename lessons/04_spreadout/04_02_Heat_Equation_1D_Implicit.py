import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import solve

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# Grid parameters
L = 1.0
nt = 100
nx = 51
alpha = 1.22e-3

q = 0.0

dx = L / (nx - 1)

qdx = q * dx

# Initial conditions
Ti = np.zeros(nx)
Ti[0] = 100


def generateMatrix(N, sigma):
    """
    Computes the matrix for the diffusion equation with backward Euler
    Dirichlet condition at i=0, Neumann at i=-1
    """

    # Setup the diagonal
    d = np.diag(np.ones(N - 2) * (2 + 1 / sigma))

    # Consider Neumann BC
    d[-1, -1] = 1 + 1 / sigma

    # Setup upper diagonal
    ud = np.diag(np.ones(N - 3) * -1, 1)

    # Setup lower diagonal
    ld = np.diag(np.ones(N - 3) * -1, -1)  # Last -1 indicates the diagonal

    A = d + ud + ld

    return A


def generateRHS(T, sigma, qdx):
    """
    Computes right-hand side of linear system for diffusion equation
    with backward Euler
    """

    b = T[1: -1] * 1 / sigma

    # Consider Dirichlet BC
    b[0] += T[0]

    # Consider Neumann BC
    b[-1] += qdx

    return b


def implicit_btcs(T, A, nt, sigma, qdx):
    """
    Advances diffusion equation in time with implicit central scheme
    """

    for t in range(nt):
        Tn = T.copy()
        b = generateRHS(Tn, sigma, qdx)

        # Solve
        T_interior = solve(A, b)
        T[1:-1] = T_interior

        # Enforce Neumann BC (Dirichlet is enforced automatically)
        T[-1] = T[-2] + qdx

    return T


sigma = 0.5
dt = sigma * dx * dx / alpha
nt = 1000

A = generateMatrix(nx, sigma)

T = implicit_btcs(Ti.copy(), A, nt, sigma, qdx)

plt.plot(np.linspace(0, 1, nx), T, color='#003366', ls='-', lw=3)
plt.show()

# Violates the stability condition of the explicit scheme
sigma = 5.0

A = generateMatrix(nx, sigma)

T = implicit_btcs(T.copy(), A, nt, sigma, qdx)

plt.plot(np.linspace(0, 1, nx), T, color='#003366', ls='-', lw=3)
plt.show()
