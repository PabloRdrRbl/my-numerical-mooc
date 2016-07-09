import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


def rho_green_light(nx, rho_light):
    rho = np.arange(nx) * 2. / nx * rho_light  # Before stop light
    rho[int((nx - 1) / 2):] = 0

    return rho

nx = 81
nt = 30
dx = 4.0 / (nx - 1)

x = np.linspace(0, 4, nx)

rho_max = 10
u_max = 1
rho_light = 10

rho = rho_green_light(nx, rho_light)

plt.plot(x, rho, color='#003366', ls='-', lw=3)
plt.ylabel('Traffic density')
plt.xlabel('Distance')
plt.ylim(-0.5, 11.)
