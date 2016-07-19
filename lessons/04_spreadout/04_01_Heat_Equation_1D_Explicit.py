import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# Grid parameters
L = 1
nt = 100
nx = 51
alpha = 1.22e-3

dx = L / (nx - 1)

sigma = 1 / 2.0
dt = sigma * dx * dx / alpha

Ti = np.zeros(nx)  # (ºC)
Ti[0] = 100  # (ºC)

x = np.linspace(0, 1, nx)


def ftcs(T, nt, dt, dx, alpha):
    Tn = T.copy()
    for n in range(nt):
        Tn[1: -1] = (Tn[1: -1] + alpha * dt / dx**2 *
                     (Tn[2:] - 2 * Tn[1: -1] + Tn[0: -2]))

    return Tn


T = ftcs(Ti.copy(), nt, dt, dx, alpha)

plt.plot(x, T, color='#003366', ls='-', lw=3)
plt.ylim(0, 100)
plt.xlabel('Length of Rod')
plt.ylabel('Temperature')

plt.show()


def ftcs_mixed(T, nt, dt, dx, alpha):
    Tn = T.copy()
    for n in range(nt):
        Tn[1: -1] = (Tn[1:-1] + alpha * dt / dx**2 *
                     (Tn[2:] - 2 * Tn[1:-1] + Tn[0:-2]))
        Tn[-1] = Tn[-2]  # Neumann condition

    return Tn

nt = 1000
T = ftcs_mixed(Ti, nt, dt, dx, alpha)

plt.plot(x, T, color='#003366', ls='-', lw=3)
plt.ylim(0, 100)
plt.title('Rod with LHS Dirichlet B.C. and RHS Neumann B.C.\n')
plt.xlabel('Length of Rod')
plt.ylabel('Temperature')

plt.show()
