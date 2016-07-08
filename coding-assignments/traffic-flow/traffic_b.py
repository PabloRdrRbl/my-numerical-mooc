# Coding assignment: traffic flow

import numpy as np
import matplotlib.pyplot as plt

# matplotlib style
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


# Problem parameters
Vmax = 136 / 3.6  # speed of traffic (m/s)
rhomax = 250 / 1000  # the number of cars per unit length of highway (cars/m)
L = 11 * 1000  # logitude of the highway (m)

nx = 51  # Points but not subintervals
dx = L / (nx - 1)  # (m)
dt = 0.001 * 3600  #  (s)

# Initial conditions
x = np.linspace(0, L, nx)
rho0 = np.ones(nx) * (20 / 1000)
rho0[10:20] = 50 / 1000

# Boundary condition
# rho(0, t) = 20 / 1000  (cars/m)


def V(rho_t):
    """
    Average velocity in (m/s)
    """
    return Vmax * (1 - rho_t / rhomax)


def F(rho):
    """
    F rate of cars per hour (cars/s)
    """
    return Vmax * rho * (1 - rho / rhomax)


def rho(t):
    """
    Solves for density (cars/m) with time in (s)
    """
    nt = int(t / dt)

    rho = np.copy(rho0)

    for n in range(nt):
        nrho = rho.copy()
        rho[1:] = nrho[1:] - dt / dx * (F(nrho[1:]) - F(nrho[: -1]))
        rho[0] = 20 / 1000  # Boundary condition (cars/m)

    return rho


if __name__ == '__main__':

    print('The minimum velocity at time t = 0 in (m/s) is: %.2f (m/s)' %
          (V(np.max(rho(0)))))

    print()

    print('The average velocity at time = 3 (min) in (m/s) is: %.2f (m/s)' %
          (V(np.mean(rho(3 * 60)))))

    print()

    print('The minimum velocity at time t = 3 (min) in (m/s) is: %.2f (m/s)' %
          (V(np.max(rho(3 * 60)))))

    plt.plot(x, rho(3 * 60), color='#003366', ls='--', lw=3)

    plt.show()
