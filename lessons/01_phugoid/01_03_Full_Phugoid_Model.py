from math import sin, cos, log, ceil
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# Parameter values
g = 9.8  # gravity in m s^{-2}
v_t = 30.0  # trim velocity in m s^{-1}
C_D = 1 / 40  # drag coefficient --- or D/L if C_L=1
C_L = 1  # for convenience, use C_L = 1

# Initial conditions
v0 = v_t  # start at the trim velocity (or add a delta)
theta0 = 0  # initial angle of trajectory
x0 = 0  # horizotal position is arbitrary
y0 = 1000  # initial altitude


def f(u):
    v = u[0]
    theta = u[1]
    # x = u[2]
    # y = u[3]

    return np.array([-g * sin(theta) - C_D / C_L * g / v_t**2 * v**2,
                     -g * cos(theta) / v + g / v_t**2 * v,
                     v * cos(theta),
                     v * sin(theta)])


def euler_step(u, f, dt):
    return u + dt * f(u)


def plot_trajectory(u):
    x = u[:, 2]
    y = u[:, 3]

    plt.figure(figsize=(8, 6))
    plt.grid(True)
    plt.xlabel(r'x', fontsize=18)
    plt.ylabel(r'y', fontsize=18)
    plt.title('Glider trajectory, flight time = %.2f' % T, fontsize=18)
    plt.plot(x, y, 'k-', lw=2)

    plt.show()


T = 100
dt = 0.1
N = int(T / dt) + 1
t = np.linspace(0, T, N)

u = np.empty((N, 4))
u[0] = np.array([v0, theta0, x0, y0])

for n in range(N - 1):
    u[n + 1] = euler_step(u[n], f, dt)

plot_trajectory(u)

# Testing the convergence
dt_values = np.array([0.1, 0.05, 0.01, 0.005, 0.001])

u_values = np.empty_like(dt_values, dtype=np.ndarray)

for i, dt in enumerate(dt_values):
    N = int(T / dt) + 1

    t = np.linspace(0.0, T, N)

    u = np.empty((N, 4))
    u[0] = np.array([v0, theta0, x0, y0])

    for n in range(N - 1):
        u[n + 1] = euler_step(u[n], f, dt)

        u_values[i] = u


def get_diffgrid(u_current, u_fine, dt):
    N_current = len(u_current[:, 0])
    N_fine = len(u_fine[:, 0])

    grid_grid_ratio = ceil(N_fine / N_current)

    diffgrid = dt * \
        np.sum(np.abs(u_current[:, 2] - u_fine[::grid_grid_ratio, 2]))

    return diffgrid

diffgrid = np.empty_like(dt_values)

for i, dt in enumerate(dt_values):
    print('dt = {}'.format(dt))

    diffgrid[i] = get_diffgrid(u_values[i], u_values[-1], dt)

# log-log plot of the grid differences
plt.figure(figsize=(6, 6))
plt.grid(True)
plt.xlabel('$\Delta t$', fontsize=18)
plt.ylabel('$L_1$-norm of the grid differences', fontsize=18)
plt.axis('equal')
plt.loglog(dt_values[:-1], diffgrid[:-1], color='k', ls='-', lw=2, marker='o')
plt.show()


r = 2
h = 0.001

dt_values2 = np.array([h, r * h, r**2 * h])

u_values2 = np.empty_like(dt_values2, dtype=np.ndarray)

diffgrid2 = np.empty(2)

for i, dt in enumerate(dt_values2):
    N = int(T / dt) + 1

    t = np.linspace(0.0, T, N)

    u = np.empty((N, 4))
    u[0] = np.array([v0, theta0, x0, y0])

    for n in range(N - 1):
        u[n + 1] = euler_step(u[n], f, dt)

    u_values2[i] = u

diffgrid2[0] = get_diffgrid(u_values2[1], u_values2[0], dt_values2[1])

diffgrid2[1] = get_diffgrid(u_values2[2], u_values2[1], dt_values2[2])

p = (log(diffgrid2[1]) - log(diffgrid2[0])) / log(r)

print('The order of convergence is p = {:.3f}'.format(p))
