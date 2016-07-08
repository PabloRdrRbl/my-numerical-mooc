import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


nx = 41
dx = 2 / (nx - 1)
nt = 25
dt = 0.02
c = 1
x = np.linspace(0, 2, nx)

u = np.ones(nx)
lbound = np.where(x >= 0.5)
ubound = np.where(x <= 1)

bounds = np.intersect1d(lbound, ubound)
u[bounds] = 2

plt.figure(figsize=(11, 8))
plt.subplot(121)
plt.plot(x, u, color='#003366', ls='--', lw=3)
plt.ylim(0, 2.5)
plt.xlim

for n in range(1, nt):
    un = u.copy()
    for i in range(1, nx):
        u[i] = un[i] - c * dt / dx * (un[i] - un[i - 1])

plt.plot(x, u, color='r', ls='--', lw=3)

# Problem parameters
nx = 41
dx = 2 / (nx - 1)
nt = 10
dt = 0.02

# Initial conditions
u = np.ones(nx)
u[np.intersect1d(lbound, ubound)] = 2

plt.subplot(122)
plt.plot(x, u, color='#003366', ls='--', lw=3)
plt.ylim(0, 2.5)

for n in range(1, nt):
    un = u.copy()
    u[1:] = un[1:] - un[1:] * dt / dx * (un[1:] - un[: -1])
    u[0] = 1.0  # Boundary condition

plt.plot(x, u, color='r', ls='--', lw=3)

plt.show()
