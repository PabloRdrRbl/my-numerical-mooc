import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

rho_max = 10.0
u_max = 1.0

# Coming from the symbolic calculations
aval = 0.0146107219255619
bval = 0.00853892780744381


def computeF(u_max, rho, aval, bval):
    return u_max * rho * (1 - aval * rho - bval * rho**2)


def rho_green_light(nx, rho_light):
    rho_initial = np.arange(nx) * 2 / nx * rho_light  # Before stoplight
    rho_initial[int((nx - 1) / 2):] = 0

    return rho_initial


# Grid parameters
nx = 81
nt = 30
dx = 4.0 / (nx - 1)

x = np.linspace(0, 4, nx)

rho_light = 5.5

rho_initial = rho_green_light(nx, rho_light)

plt.plot(x, rho_initial, color='#003366', ls='-', lw=3)
plt.ylim(-0.5, 11.0)

# plt.show()


def ftbs(rho, nt, dt, dx, rho_max, u_max):
    """
    Forward in time and backward in space.
    """
    rho_n = np.zeros((nt, len(rho)))
    rho_n[0, :] = rho.copy()

    for t in range(1, nt):
        F = computeF(u_max, rho, aval, bval)
        rho_n[t, 1:] = rho[1:] - dt / dx * (F[1:] - F[:-1])
        rho_n[t, 0] = rho[0]
        rho_n[t, -1] = rho[-1]
        rho = rho_n[t].copy()

    return rho_n

sigma = 1.
dt = sigma * dx / u_max

rho_n = ftbs(rho_initial, nt, dt, dx, rho_max, u_max)
x = np.linspace(0, 4, nx)

for t in range(nt):
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 4), ylim=(-1, 8), xlabel=('Distance'),
                  ylabel=('Traffic density'))
    ax.plot(x, rho_n[t], color='#003366', lw=2)
    plt.savefig('output/image-%02d.png' % t)
    plt.close()
