import numpy as np
import matplotlib.pyplot as plt


def euler_method(dt=0.02):
    # Grid parameters
    T = 100.0
    N = int(T / dt) + 1  # Number of time steps. n parts and n+1 points
    t = np.linspace(0.0, T, N)

    # Initial conditions
    z0 = 100.0  # Altitude
    b0 = 10.0  # Upward velocity
    zt = 100.0
    g = 9.81

    u = np.array([z0, b0])

    # Initialize an array to hold the changing elevation values
    z = np.empty_like(t)
    z[0] = z0

    # Time-loop using Euler's method
    for n in range(N):
        u = u + dt * np.array([u[1], g * (1 - u[0] / zt)])
        z[n] = u[0]  # This will be the final solution

    # Addig the exact solution (analytical)
    z_exact = b0 * (zt / g)**.5 * np.sin((g / zt)**.5 * t) +\
        (z0 - zt) * np.cos((g / zt)**.5 * t) + zt

    # Ploting the result
    plt.figure(figsize=(10, 4))
    plt.ylim(40, 160)
    plt.tick_params(axis='both', labelsize=14)
    plt.xlabel('t', fontsize=14)
    plt.ylabel('z', fontsize=14)
    plt.plot(t, z, 'k-')
    plt.plot(t, z_exact)
    plt.legend(['Numerical Solution', 'Analytical Solution'])

    plt.show()

    return z


def get_error(z, dt):
    N = len(z)
    T = 100.0
    z0 = 100.0  # Altitude
    b0 = 10.0  # Upward velocity
    zt = 100.0
    g = 9.81

    t = np.linspace(0.0, T, N)

    z_exact = b0 * (zt / g)**.5 * np.sin((g / zt)**.5 * t) +\
        (z0 - zt) * np.cos((g / zt)**.5 * t) + zt

    return dt * np.sum(np.abs(z - z_exact))


if __name__ == '__main__':
    dt_values = np.array([0.1, 0.05, 0.01, 0.005, 0.001, 0.0001])

    z_values = np.empty_like(dt_values, dtype=np.ndarray)

    for i, dt in enumerate(dt_values):
        z_values[i] = euler_method(dt)

    error_values = np.empty_like(dt_values)

    for i, dt in enumerate(dt_values):
        error_values[i] = get_error(z_values[i], dt)

    plt.figure(figsize=(10, 6))
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(True)
    plt.xlabel('$\Delta t$', fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.loglog(dt_values, error_values, 'ko-')
    plt.axis('equal')

    plt.show()
