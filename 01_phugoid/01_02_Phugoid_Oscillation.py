import numpy as np
import matplotlib.pyplot as plt


# Grid parameters
T = 100.0
dt = 0.02
N = int(T / dt) + 1  # Number of time steps. n parts and n+1 points
t = np.linspace(0.0, T, N)

# Initial conditions
z0 = 100.0  # Altitude
b0 = 10.0  # Upward velocity
zt = 100.0
g = 9.81

u = np.array([z0, b0])

# Initialize an array to hold the changing elevation values
z = np.zeros(N)
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
