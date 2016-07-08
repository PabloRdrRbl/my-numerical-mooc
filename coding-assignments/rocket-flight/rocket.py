import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


# PHYSICAL PARAMETERS
# h --> the altitude of the rocket
ms = 50  # mass of the rocket shell (kg)
mpo = 100  # mass of the rocket propellant at time t=0 (kg)
ve = 325  # exhaust speed (m/s)
g = 9.81  # gravity acceleration (m/s^2)
r = 0.5  # radio of the rocket (m)
A = np.pi * r**2  # cross sectional area of the rocket (m^2)
rho = 1.091  # average density, cte (kg/m^3)
CD = 0.15  # drag coefficient (adim)

v0 = 0  # At the begining the rocket is at rest
h0 = 0  # At the begining the rocket is on the ground

# GRID PARAMETERS
dt = 0.1  # timestep (s)
T = 50  # maximum time simulated (s)
N = int(T / dt) + 1  # Time steps taken
t = np.linspace(0, T, N)


def mp_rate(t):
    """
    Calculates the propellant burn rate (kg/s).

    t = n * dt
    """
    if (t < 5):
        mp = 20  # me: mass exhausted and 20 (kg/s) propellant burn rate
    else:
        mp = 0  # before 5 (s) the rocket shuts down

    return mp


def me(t):
    if (t < 5):
        me = 20 * t  # me: mass exhausted and 20 (kg/s) propellant burn rate
    else:
        me = 20 * 5  # before 5 (s) the rocket shuts down

    return me


def mp(t):
    """
    Calculates the remaining propellant in the rocket.
    """
    return mpo - me(t)


def f(u, t):
    v = u[0]
    # h = u[1]

    k1 = ms + mp(t)
    k2 = mp_rate(t) * ve
    k3 = 0.5 * rho * v * abs(v) * A * CD

    return np.array([(- k1 * g + k2 - k3) / k1,
                     v])


def euler_step(u, f, n, dt):
    return u + dt * f(u, n * dt)


def solve():
    u = np.empty((N, 2))
    u[0] = np.array([v0, h0])

    for n in range(N - 1):
        u[n + 1] = euler_step(u[n], f, n, dt)

    return u


def plot_solutions(u):
    plt.figure(1)
    plt.plot(t, u[:, 0], 'k-', lw=2)
    plt.title('velocity vs. time')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$v$')
    plt.savefig('velocity.png')

    plt.figure(2)
    plt.plot(t, u[:, 1], 'k-', lw=2)
    plt.ylim(0, 1400)
    plt.title('height vs. time')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$h$')
    plt.savefig('height.png')


if __name__ == '__main__':
    print('-- Remaining fuel --')

    print('At a time t = 3.2 (s) the mass (kg) of the rocket '
          'propellant remaining in the roket is: %.4f (kg)' %
          (mp(3.2)))

    print('')

    print('-- Maximum velocity --')

    u = solve()

    print('The maximum speed of the rocket (m/s) is: %.4f (m/s)' %
          (np.max(u[:, 0])))
    print('The maximum speed occurs at: %.4f (s)' % (t[np.argmax(u[:, 0])]))
    print('The altitude (m) at this time is: %.4f (m)' %
          u[np.argmax(u[:, 0]), 1])

    print('')

    print('-- Maximum height --')

    print('The maximum height of the rocket (m) is: %.4f (m)' %
          (np.max(u[:, 1])))
    print('The maximum height occurs at: %.4f (s)' % (t[np.argmax(u[:, 1])]))

    print('')

    print('-- Impact --')

    print('The impact occurs at: %.4f (s)' % (t[np.where(u[:, 1] < 0)[0][0]]))
    print('The speed of the rocket (m/s) at time of impact is: %.4f (m/s)' %
          (u[np.where(u[:, 1] < 0)[0][0], 0]))

    plot_solutions(u)
