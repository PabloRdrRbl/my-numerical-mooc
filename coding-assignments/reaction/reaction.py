import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm


# Grid points with dimension 192x192
n = 192


# Du, Dv, F, k = 0.00016, 0.00008, 0.035, 0.065 # Bacteria 1
# Du, Dv, F, k = 0.00014, 0.00006, 0.035, 0.065 # Bacteria 2
# Du, Dv, F, k = 0.00016, 0.00008, 0.060, 0.062  # Coral
# Du, Dv, F, k = 0.00019, 0.00005, 0.060, 0.062  # Fingerprint
Du, Dv, F, k = 0.00010, 0.00010, 0.018, 0.050  # Spirals
# Du, Dv, F, k = 0.00012, 0.00008, 0.020, 0.050 # Spirals Dense
# Du, Dv, F, k = 0.00010, 0.00016, 0.020, 0.050 # Spirals Fast
# Du, Dv, F, k = 0.00016, 0.00008, 0.020, 0.055  # Unstable
# Du, Dv, F, k = 0.00016, 0.00008, 0.050, 0.065 # Worms 1
# Du, Dv, F, k = 0.00016, 0.00008, 0.054, 0.063 # Worms 2
# Du, Dv, F, k = 0.00016, 0.00008, 0.035, 0.060 # Zebrafish
# Domain is 5m x 5m
dh = 5 / (n - 1)

# Final time (s)
T = 8000

# Timestep
dt = 0.9 * dh**2 / (4 * max(Du, Dv))

# Number of time steps
nt = int(T / dt)

# Initial conditions
uvinitial = np.load('./data/uvinitial.npz')
U = uvinitial['U']
V = uvinitial['V']

# Ploting the initial conditions
fig = plt.figure(figsize=(8, 5))
plt.imshow(U, cmap=cm.RdBu)
plt.xticks([]), plt.yticks([])
plt.imshow(V, cmap=cm.RdBu)
plt.xticks([]), plt.yticks([])

plt.savefig('./image/image_00.png')
plt.close()


def reaction(U, V, Du, Dv, F, k, dh, dt, nt):

    for t in range(1, nt):
        Un = U.copy()
        Vn = V.copy()

        U[1: -1, 1: -1] = ((dt * (Du / dh**2 *
                                  (Un[1: -1, : -2] - 2 * Un[1: -1, 1: -1] +
                                   Un[1: -1, 2:] +
                                   Un[: -2, 1: -1] - 2 * Un[1: -1, 1: -1] +
                                   Un[2:, 1: -1]) -
                                  Un[1: -1, 1: -1] * Vn[1: -1, 1: -1] ** 2 +
                                  F * (1 - Un[1: -1, 1: -1]))) +
                           Un[1: -1, 1: -1])

        V[1: -1, 1: -1] = ((dt * (Dv / dh**2 *
                                  (Vn[1: -1, : -2] - 2 * Vn[1: -1, 1: -1] +
                                   Vn[1: -1, 2:] +
                                   Vn[: -2, 1: -1] - 2 * Vn[1: -1, 1: -1] +
                                   Vn[2:, 1: -1]) +
                                  Un[1: -1, 1: -1] * Vn[1: -1, 1: -1] ** 2 -
                                  (F + k) * Vn[1: -1, 1: -1])) +
                           Vn[1: -1, 1: -1])

        # Neumann BC
        U[0, :] = U[1, :]
        U[-1, :] = U[-2, :]
        U[:, 0] = U[:, 1]
        U[:, -1] = U[:, -2]

        V[0, :] = V[1, :]
        V[-1, :] = V[-2, :]
        V[:, 0] = V[:, 1]
        V[:, -1] = V[:, -2]

        if (t % 100 == 0):
            fig = plt.figure(figsize=(8, 5))
            plt.imshow(U, cmap=cm.RdBu)
            plt.xticks([]), plt.yticks([])
            plt.imshow(V, cmap=cm.RdBu)
            plt.xticks([]), plt.yticks([])

            plt.savefig('./image/image_%02.d.png' % (t / 100))
            plt.close()

    return U, V

U, V = reaction(U.copy(), V.copy(), Du, Dv, F, k, dh, dt, nt)

print(U[100, ::40])
