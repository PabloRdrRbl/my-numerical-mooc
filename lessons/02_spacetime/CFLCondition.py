import numpy as np
import matplotlib.pyplot as plt

from matplo import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


def linearconv(nx):
    dx = 2 / (nx - 1)
    nt = 20
    dt = 0.025
    c = 1

    x = np.linspace(0, 2, nx)

    u = np.ones(nx)
    lbound = np.where(x >= 0.5)
    ubound = np.where(x <= 1)
    u[np.intersect1d(lbound, ubound)] = 2

    for n in range(nt):
        un = u.copy()
        u[1:] = un[1:] - c * dt / dx * (un[1:] - un[0:-1])
    u[0] = 1.0  # Boundary condition

    plt.plot(x, u, color='#003366', ls='--', lw=3)
    plt.ylim(0, 2.5)

    plt.show()


for nx in [41, 61, 71, 85]:
    linearconv(nx)
