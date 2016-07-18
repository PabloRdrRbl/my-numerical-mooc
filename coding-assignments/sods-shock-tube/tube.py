import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
t = 0.01  # (s)
L = 20  # (m), from (x = -10) to (x = 10)

nx = 81  # x divided in (nx - 1) 
dt = 0.0002  # (s), time steps


dx = L / (nx - 1)  # (m), interval's lenght
nt = int((t / dt) + 1)

gamma = 1.4  # ideal gas constant for air

x = np.linspace(-10, 10, nx)

def ini_con(rho, v, p, gamma):
    e = p / ((gamma - 1) * rho)

    u = np.array([rho,
                  rho * v,
                  rho * (e + 0.5 * v**2)])

    return u[np.newaxis].T

def computeP(u, gamma):
    """
    Computes the pressure in terms of vector u.

    u -- must be givien for a *given* time and space. 

         So: u[t0, 0, x0] = u_1(x0, t0) --> u[0]
             u[t0, 3, x0] = u_2(x0, t0) --> u[1]
             u[t0, 2, x0] = u_3(x0, t0) --> u[2]
    """
    return (gamma - 1) * (u[2] - 0.5 * u[1]**2 / u[0])


def computeF(u, gamma):
    """
    Computes the flux in terms of vector u.
    Flux returned is:

        f(x0, t0) = 

    u -- must be givien for a *given* time and space. 

         So: u[t0, 0, x0] = u_1(x0, t0) --> u[0]
             u[t0, 3, x0] = u_2(x0, t0) --> u[1]
             u[t0, 2, x0] = u_3(x0, t0) --> u[2]
    """
    p = computeP(u, gamma)
    
    f = np.array([u[1],
                  u[1]**2 / u[0] + p,
                  (u[2] + p) * u[1] / u[0]])
    
    return f


def track(u, f, n):
    fig1, (ax1, ax2, ax3) = plt.subplots(figsize=(10, 6), ncols=3)

    p = computeP(u, 1.4)
    rho = u[0]
    v = u[1] / u[0]
    
    ax1.plot(x, rho, color='r', lw=2)
    ax2.plot(x, v, color='b', lw=2)
    ax3.plot(x, p, color='g', lw=2)

    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(-0.2, 400)
    ax3.set_ylim(0, 1e5 + 100)
    
    plt.savefig('output/image-%03.d' % n)
    plt.close()

# 2 dimensinal array for vector u(x)
# u is 3x1 and depends on space (nx). So (3 x nx)
u = np.ones((3, nx))  # u(x) [=] (kg/m^3, m/s, N/m^2)

# Initial conditions (t = 0 s)
# (kg/m^3, m/s, N/m^2)
u[:, :int((nx - 1) / 2)] = ini_con(1, 0, 100e3, gamma)
u[:, int((nx - 1) / 2):] = ini_con(0.125, 0, 10e3, gamma)  

# 3 dimensinal array for vector u(x, t)
# u is 3x1 and depends on space (nx) and time (nt). So
# (3 x nt x nx)
un = np.ones((3, nt, nx))  # u(x, t) [=] (kg/m^3, m/s, kN/m^2)

us = np.ones_like(u) # It's going to be nx - 1 given tha it's in the middle

un[:, 0, :] = u.copy()

for n in range(1, nt):  # Looping in time
    # First step
    f = computeF(u, gamma)
    # It's going to be nx - 1 given tha it's in the middle ([:, 1])
    # Not possible to compute the -1
    us[:, :-1] = 0.5 * ((u[:, 1:] + u[:, :-1]) -
                     dt / dx * (f[:, 1:] - f[:, :-1]))

    # Second step
    fs = computeF(us, gamma)
    # I can't compute u[:, 0], neither u[:. -1]
    u[:, 1:-1] = u[:, 1:-1] - dt / dx * (fs[:, 1:-1] - fs[:, :-2])

    track(u, f, n)
    
    un[:, n, :] = u.copy()
    un[:, n, 1] = ini_con(1, 0, 100e3, gamma).T
    un[:, n, -1] = ini_con(0.125, 0, 10e3, gamma).T

