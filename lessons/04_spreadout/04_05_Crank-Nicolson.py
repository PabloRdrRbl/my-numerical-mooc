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


# Accuracy and convergence
def T_analytical(x, t, n_max, alpha, L):
    T = 100
    for n in range(1, n_max + 1):
        k = (2 * n - 1) * np.pi / (2 * L)

        summation = (400 / ((2 * n - 1) * np.pi) * np.sin(k * x) *
                     np.exp(- alpha * k * k * t))

        T -= summation

    return T

T_exact = T_analytical(x, dt * nt, 100, alpha, L)
plt.plot(x, T_exact, color='#003366', ls='-', lw=3)

plt.show()


def L2_error(T, T_exact):
    e = np.sqrt(np.sum((T - T_exact) ** 2) / np.sum(T_exact) ** 2)

    return e


def generateMatrix_btcs(N, sigma):
    d = np.diag(np.ones(N - 2) * (2 + 1 / sigma))

    d[-1, -1] = 1 + 1 / sigma

    ud = np.diag(np.ones(N - 3) * -1, 1)

    ld = np.diag(np.ones(N - 3) * -1, -1)

    A = d + ud + ld

    return A


def generateRHS_btcs(T, sigma):
    b = np.zeros_like(T)

    b = T[1: -1] * 1 / sigma

    b[0] += T[0]

    return b


def implicit_btcs(T, A, nt, sigma):
    for t in range(nt):
        Tn = T.copy()
        b = generateMatrix_btcs(Tn, sigma)
        T_interior = solve(A, b)
        T[1: -1] = T_interior
        T[-1] = T[-2]

    return T

nx = 1001
dx = L / (nx - 1)

dt_values = np.asarray([1.0, 0.5, 0.25, 0.125])
error = np.zeros(len(dt_values))
error_btcs = np.zeros(len(dt_values))

t_final = 10
t_inicial = 1

x = np.linspace(0, L, nx)

Ti = T_analytical(x, t_inicial, 100, alpha, L)
T_exact = T_analytical(x, t_final, 100, alpa, L)

for i, dt in enumerate(dt_values):
    sigma = alpha * dt / dx**2

    nt = int((t_final - t_inicial) / dt)

    A = generateMatrix(nx, sigma)

    A_btcs = generateMatrix_btcs(nx, sigma)

    T = CrankNicolson(Ti.copy(), A, nt, sigma)

    error[i] = L2_error(T, T_exact)

    T = implicit_btcs(Ti.copy(), A_btcs, nt, sigma)

    error_btcs[i] = L2_error(T, T_exact)


plt.figure(figsize=(8, 8))
plt.grid(True)
plt.xlabel(r'$\Delta t$', fontsize=18)
plt.ylabel(r'$L_2$-norm of the error', fontsize=18)
plt.axis('equal')
plt.loglog(dt_values, error, color='k', ls='--', lw=2, marker='o')
plt.loglog(dt_values, error_btcs, color='k', ls='--', lw=2, marker='s')
plt.legend(['Crank-Nicolson', 'BTCS'])

plt.show()


nx_values = np.asarray([11, 21, 41, 81, 161])

dt = 0.1
error = np.zeros(len(nx_values))

t_final = 20

x = np.linspace(0, L, nx)

for i, nx in enumerate(nx_values):

    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    sigma = alpha * dt / dx**2

    nt = int(t_final / dt)

    A = generateMatrix(nx, sigma)

    Ti = np.zeros(nx)
    Ti[0] = 100

    T = CrankNicolson(Ti.copy(), A, nt, sigma)

    T_exact = T_analytical(x, t_final, 100, alpha, L)

    error[i] = L2_error(T, T_exact)


plt.figure(figsize=(8, 8))
plt.grid(True)
plt.xlabel(r'$n_x$', fontsize=18)
plt.ylabel(r'$L_2$-norm of the error', fontsize=18)
plt.axis('equal')
plt.loglog(nx_values, error, color='k', ls='--', lw=2, marker='o')

plt.show()


nx_values = np.asarray([11, 21, 41, 81, 161])

dt = 0.1
error = np.zeros(len(nx_values))

t_final = 1000

x = np.linspace(0, L, nx)

for i, nx in enumerate(nx_values):
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    sigma = alpha * dt / dx**2

    nt = int(t_final / dt)

    A = generateMatrix(nx, sigma)

    Ti = np.zeros(nx)
    Ti[0] = 100

    T = CrankNicolson(Ti.copy(), A, nt, sigma)

    T_exact = T_analytical(x, t_final, 100, alpha, L)

    error[i] = L2_error(T, T_exact)


plt.figure(figsize=(8, 8))
plt.grid(True)
plt.xlabel(r'$n_x$', fontsize=18)
plt.ylabel(r'$L_2$-norm of the error', fontsize=18)
plt.xlim(1, 1000)
plt.ylim(1e-5, 1e-2)
plt.loglog(nx_values, error, color='k', ls='--', lw=2, marker='o')

plt.show()
