from math import sin, cos, log
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


# Model parameters
g = 9.8        # gravity in m s^{-2}
v_t = 4.9      # trim velocity in m s^{-1}
C_D = 1 / 5.0  # drag coefficient --- or D/L if C_L=1
C_L = 1.0      # for convenience, use C_L = 1

# set initial conditions
v0 = 6.5       # start at the trim velocity (or add a delta)
theta0 = -0.1  # initial angle of trajectory
x0 = 0.0       # horizotal position is arbitrary
y0 = 2.0       # initial altitude


def f(u):
    v = u[0]
    theta = u[1]
    x = u[2]
    y = u[3]

    return np.array([-g * sin(theta) - C_D / C_L * g / v_t**2 * v**2,
                     -g * cos(theta) / v + g / v_t**2 * v,
                     v * cos(theta),
                     v * sin(theta)])


def euler_step(u, f, dt):
    return u + dt * f(u)


def get_diffgrid(u_current, u_fine, dt):
    N_current = len(u_current[:, 0])
    N_fine = len(u_fine[:, 0])

    grid_size_ratio = np.ceil(N_fine / N_current)

    diffgrid = dt * \
        np.sum(np.abs(u_current[:, 2] - u_fine[::grid_size_ratio, 2]))

    return diffgrid


def rk2_step(u, f, dt):
    u_star = u + 0.5 * dt * f(u)
    return u + dt * f(u_star)


# set time-increment and discretize the time
T = 15.0                           # final time
dt = 0.01                           # set time-increment
N = int(T / dt) + 1                  # number of time-steps


# set initial conditions
u_euler = np.empty((N, 4))
u_rk2 = np.empty((N, 4))


# initialize the array containing the solution for each time-step
u_euler[0] = np.array([v0, theta0, x0, y0])
u_rk2[0] = np.array([v0, theta0, x0, y0])

for n in range(N - 1):
    u_euler[n + 1] = euler_step(u_euler[n], f, dt)
    u_rk2[n + 1] = rk2_step(u_rk2[n], f, dt)


# Position of the glider in time
x_euler = u_euler[:, 2]
y_euler = u_euler[:, 3]
x_rk2 = u_rk2[:, 2]
y_rk2 = u_rk2[:, 3]

# get the index of element of y where altitude becomes negative
idx_negative_euler = np.where(y_euler < 0.0)[0]
if len(idx_negative_euler) == 0:
    idx_ground_euler = N - 1
    print('Euler integration has not touched ground yet!')
else:
    idx_ground_euler = idx_negative_euler[0]

idx_negative_rk2 = np.where(y_rk2 < 0.0)[0]
if len(idx_negative_rk2) == 0:
    idx_ground_rk2 = N - 1
    print('Runge-Kutta integration has not touched ground yet!')
else:
    idx_ground_rk2 = idx_negative_rk2[0]

# check to see if the paths match
print('Are the x-values close? {}'.format(np.allclose(x_euler, x_rk2)))
print('Are the y-values close? {}'.format(np.allclose(y_euler, y_rk2)))

# plot the glider path
plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.grid(True)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.plot(x_euler[:idx_ground_euler],
         y_euler[:idx_ground_euler], 'k-', label='Euler')
plt.plot(x_rk2[:idx_ground_rk2], y_rk2[:idx_ground_rk2], 'r--', label='RK2')
plt.title('distance traveled: {:.3f}'.format(x_rk2[idx_ground_rk2 - 1]))
plt.legend()

# Let's take a closer look!
plt.subplot(122)
plt.grid(True)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.plot(x_euler, y_euler, 'k-', label='Euler')
plt.plot(x_rk2, y_rk2, 'r--', label='RK2')
plt.xlim(0, 5)
plt.ylim(1.8, 2.5)

plt.savefig('euler-rk2.png')
plt.close()
