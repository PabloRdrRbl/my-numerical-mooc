import numpy as np
import matplotlib.pyplot as plt

# matplotlib settings
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


def rho_red_ligth(nx, rho_max, rho_in):
    """
    Computes the red light initial condition with shock
    """
    rho = np.ones(nx) * rho_max
    rho[: int((nx - 1) * 3 / 4)] = rho_in

    return rho


def wave_speed(u_max, rho_max, rho):
    return u_max * (1 - 2 * (rho / rho_max))


def computeF(u_max, rho_max, rho):
    return u_max * rho * (1 - rho / rho_max)


# Basic parameters
nx = 81
nt = 30
dx = 4 / (nx - 1)

rho_in = 5
rho_max = 10

u_max = 1

x = np.linspace(0, 4, nx)

rho = rho_red_ligth(nx, rho_max, rho_in)

# Initial plot
plt.plot(x, rho, color='#003366', ls='-', lw=3)
plt.plot(x, wave_speed(u_max, rho_max, rho), color='r', ls='-', lw=3)
plt.ylabel('Traffic density')
plt.xlabel('Distance')
plt.ylim(-2, 11.)

plt.show()


def laxfriedrichs(rho, nt, dt, dx, rho_max, u_max):
    """
    Computes the solution with Lax-Friedrichs scheme

    Keyword Arguments:
    rho     -- array of floats. Density at current time-step
    nt      -- Number of time steps
    dt      -- Time-step size
    dx      -- Mesh spacing
    rho_max -- Maximum allowed car density
    u_max   -- Speed limit
    """
    rho_n = np.zeros((nt, len(rho)))
    rho_n[:] = rho.copy()

    for t in range(1, nt):
        F = computeF(u_max, rho_max, rho)
        rho_n[t, 1:-1] = (0.5 * (rho[2:] + rho[:-2]) -
                          0.5 * dt / dx * (F[2:] - F[:-2]))
        # Boundary conditions
        rho_n[t, 0] = rho[0]
        rho_n[t, -1] = rho[-1]
        rho = rho_n[t].copy()

    return rho_n


def Jacobian(u_max, rho_max, rho):
    return u_max * (1 - 2 * rho / rho_max)


def laxwendroff(rho, nt, dt, dx, rho_max, u_max):
    rho_n = np.zeros((nt, len(rho)))
    rho_n[:] = rho.copy()

    for t in range(1, nt):
        F = computeF(u_max, rho_max, rho)
        J = Jacobian(u_max, rho_max, rho)

        rho_n[t, 1:-1] = (rho[1:-1] - dt / (2 * dx) * (F[2:] - F[:-2]) +
                          dt**2 / (4 * dx**2) *
                          ((J[2:] + J[1:-1]) * (F[2:] - F[1:-1]) -
                           (J[1:-1] + J[:-2]) * (F[1:-1] - F[:-2])))

        rho_n[t, 0] = rho[0]
        rho_n[t, -1] = rho[-1]
        rho = rho_n[t].copy()

    return rho_n


def maccormack(rho, nt, dt, dx, u_max, rho_max):
    rho_n = np.zeros((nt, len(rho)))
    rho_n[:, :] = rho.copy()
    rho_star = rho.copy()

    for t in range(1, nt):
        F = computeF(u_max, rho_max, rho)
        rho_star[:-1] = rho[:-1] - dt / dx * (F[1:] - F[:-1])
        Fstar = computeF(u_max, rho_max, rho_star)
        rho_n[t, 1:] = (0.5 * (rho[1:] + rho_star[1:] -
                               dt / dx * (Fstar[1:] - Fstar[:-1])))
        rho = rho_n[t].copy()

    return rho_n
