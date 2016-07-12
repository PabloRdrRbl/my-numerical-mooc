import numpy as np
import matplotlib.pyplot as plt

# matplotlib setup
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


def rho_green_light(nx, rho_light):
    """
    Returns the initial distribution of the density as a function of the
    density at the stoplight.

    Before the stop light (x=2) rho = rho_max * x/2
    Afeter the stop light (x=2) rho = 0

    rho_light is the density of cars (car/km) at stoplight
    """
    # We take (0, 1) as the general case
    rho = np.linspace(0, 1, nx) * 2 * rho_light
    rho[int((nx - 1) / 2):] = 0

    return rho

nx = 81
nt = 30
dx = 4.0 / (nx - 1)

x = np.linspace(0, 4, nx)

rho_max = 10
u_max = 1
rho_light = 5

rho = rho_green_light(nx, rho_light)

plt.plot(x, rho, color='#003366', ls='-', lw=3)
plt.ylabel('Traffic density')
plt.xlabel('Distance')
plt.ylim(-0.5, 11.)

plt.show()


def computeF(u_max, rho_max, rho):
    """
    Computes the flux F = V * rho, at any point
    """
    return u_max * rho * (1 - rho / rho_max)


def ftbs(rho, nt, dt, dx, rho_max, u_max):
    """Computes the solution with forward in time, backward in space
    """
    rho_n = np.zeros((nt, len(rho)))  # Density for each point and time

    rho_n[0, :] = rho.copy()

    for t in range(1, nt):  # t=0 are the initial conditions
        F = computeF(u_max, rho_max, rho)
        rho_n[t, 1:] = rho[1:] - dt / dx * (F[1:] - F[:-1])
        rho_n[t, 0] = rho[0]  # rho.size --> (1, nx)
        rho = rho_n[t].copy()

    return rho_n


sigma = 1
dt = sigma * dx

rho_n = ftbs(rho, nt, dt, dx, rho_max, u_max)
for t in range(nt):
    plt.plot(x, rho_n[t], color='#003366', ls='-', lw=3)
    plt.ylabel('Traffic density')
    plt.xlabel('Distance')
    plt.ylim(-0.5, 11.)

    plt.show()
