from traffic import rho_red_light, computeF

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# Grid parameters
nx = 101
nt = 30
dx = 4.0 / (nx - 1)

rho_in = 5.0
rho_max = 10.0

V_max = 1.0

x = np.linspace(0, 4, nx - 1)

rho = rho_red_light(nx - 1, rho_max, rho_in)


def godunov(rho, nt, dt, dx, rho_max, V_max):
    """
    Computes the solution with the Godunov scheme
    using the Lax-Friedrichs flux.
    """
    rho_n = np.zeros((nt, len(rho)))
    rho_n[:] = rho.copy()

    rho_plus = np.zeros_like(rho)
    rho_minus = np.zeros_like(rho)
    flux = np.zeros_like(rho)

    for t in range(1, nt):
        rho_plus[: -1] = rho[1:]  # Can't do i+1/2 indices, so cell boundary
        rho_minus = rho.copy()  # arrays at index i are at location i+1/2
        flux = 0.5 * (computeF(V_max, rho_max, rho_minus) +
                      computeF(V_max, rho_max, rho_plus) +
                      dx / dt * (rho_minus - rho_plus))

        rho_n[t, 1:-1] = rho[1:-1] + dt / dx * (flux[:-2] - flux[1: -1])
        rho_n[t, 0] = rho[0]
        rho_n[t, -1] = rho[-1]
        rho = rho_n[t].copy()

    return rho_n

sigma = 1.0
dt = sigma * dx / V_max

rho = rho_red_light(nx - 1, rho_max, rho_in)
rho_n = godunov(rho, nt, dt, dx, rho_max, V_max)
