import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams, cm
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


def ftcs(T, nt, alpha, dt, dx, dy):
    j_mid = int((np.shape(T)[0]) / 2)  # row number
    i_mid = int((np.shape(T)[1]) / 2)  # column number

    for n in range(nt):
        Tn = T.copy()

        T[1: -1, 1: -1] = (Tn[1:-1, 1:-1] + alpha *
                           (dt / dy**2 * (Tn[2:, 1:-1] - 2 * Tn[1:-1, 1:-1] +
                                          Tn[:-2, 1:-1]) + \
                            dt / dx**2 * (Tn[1:-1, 2:] - 2 * Tn[1:-1, 1:-1] + \
                                          Tn[1:-1, :-2])))

        # Enforce Neumann BCs
        T[-1, :] = T[-2, :]
        T[:, -1] = T[:, -2]

        # Check if we reached T = 70ºC
        if T[j_mid, i_mid] >= 70:
            print("Center of plate reached 70ºC at time {0:.2f}s.".format(dt *
                                                                           n))
            break

    if T[j_mid, i_mid] < 70:
        print("Center has not reached 70ºC yet, it is only {0:.2f}ºC.".format(
            T[j_mid, i_mid]))

    return T


L = 1.0e-2
H = 1.0e-2

nx = 21
ny = 21
nt = 500

dx = L / (nx - 1)
dy = H / (ny - 1)

x = np.linspace(0, L, nx)
y = np.linspace(0, H, ny)

alpha = 1e-4

Ti = np.ones((ny, nx)) * 20
Ti[0, :] = 100
Ti[:, 0] = 100

# Time step
sigma = 0.25
dt = sigma * min(dx, dy)**2 / alpha
T = ftcs(Ti.copy(), nt, alpha, dt, dx, dy)

# Visualization
plt.figure(figsize=(8, 5))
plt.contourf(x, y, T, 20, cmap=cm.viridis)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.colorbar()

plt.show()
