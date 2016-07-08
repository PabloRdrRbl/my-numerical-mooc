import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

nx = 41
dx = 2. / (nx - 1)
nt = 20
nu = 0.3
sigma = 0.2
dt = sigma * dx**2 / nu

x = np.linspace(0, 2, nx)
ubound = np.where(x >= 0.5)
lbound = np.where(x <= 1)

u = np.ones(nx)
u[np.intersect1d(lbound, ubound)] = 2
plt.plot(x, u, color='r', ls='--', lw=3)

for n in range(nt):
    un = u.copy()
    u[1: -1] = (un[1:-1] + nu * dt / dx**2 *
                (un[2:] - 2 * un[1: -1] + un[0: -2]))

plt.plot(x, u, color='#003366', ls='--', lw=3)
plt.ylim(0, 2.5)

plt.savefig('hola.png')
