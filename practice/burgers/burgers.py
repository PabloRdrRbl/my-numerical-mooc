import numpy as np
import matplotlib.pyplot as plt


def u_initial(nx):
    u = np.ones(nx)
    u[nx / 2:] = 0

    return u


def computeF(u):
    return 0.5 * u**2


def maccormack(u, nt, dt, dx):
    un = np.zeros((nt, len(u)))
    un[:] = u.copy()
    ustar = u.copy()

    for n in range(1, nt):
        F = computeF(u)
        ustar[: -1] = u[: -1] - dt / dx * (F[1:] - F[:-1])
        Fstar = computeF(ustar)
        un[n, 1:] = 0.5 * (u[1:] + ustar[1:] - dt /
                           dx * (Fstar[1:] - Fstar[:-1]))
        u = un[n].copy()

    return un

nx = 81
nt = 70
dx = 4.0 / (nx - 1)

u = u_initial(nx)
sigma = .5
dt = sigma * dx

un = maccormack(u, nt, dt, dx)
x = np.linspace(0, 4, nx)

for i in range(nt):
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 4), ylim=(-.5, 2))
    ax.plot(x, un[i], lw=2)

    plt.savefig('output/image-%02.d' % i)

    plt.close()
