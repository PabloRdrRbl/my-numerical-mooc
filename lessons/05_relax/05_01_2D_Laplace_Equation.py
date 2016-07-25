import numpy as np
import matplotlib.pyplot as plt

# Matplotlib customizations
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# Working with 3D plots
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_3D(x, y, p):
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.view_init(30,45)


def p_analytical(x, y):
    X, Y = np.meshgrid(x, y)

    
    p_an = np.sinh(1.5*np.pi*Y / x[-1]) /\
           (np.sinh(1.5*np.pi*y[-1]/x[-1]))*np.sin(1.5*np.pi*X/x[-1])
    return p_an

nx = 41
ny = 41

x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)

p_an = p_analytical(x, y)

plot_3D(x, y, p_an)
plt.show()

def L2_error(p, pn):
    return np.sqrt(np.sum((p - pn)**2 / np.sum(pn**2)))

def laplace2d(p, l2_target):
    l2norm = 1
    pn = np.empty_like(p)
    iterations = 0

    while l2norm > l2_target:
        pn = p.copy()
        p[1: -1, 1: -1] = (0.25 * (pn[1: -1, 2:] + pn[1: -1, :-2] +
                           pn[2:, 1: -1] + pn[:-2, 1: -1]))

        p[1: -1, -1] = p[1: -1, -2]
        l2norm = L2_error(p, pn)

    return p

# Grid parameters
nx = 41
ny = 41

# Initial conditions
p = np.zeros((ny, nx))

# Ploting
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)

# Dirichlet BC
p[-1, :] = np.sin(1.5 * np.pi * x / x[-1])

p = laplace2d(p.copy(), 1e-8)

plot_3D(x, y, p)


def laplace_IG(nx):
    # Initial conditions
    p = np.zeros((nx, nx))

    x = np.linspace(0, 1, nx)
    y = x.copy()

    # Dirichlet BC
    p[:, 0] = 0
    p[0, :] = 0
    p[-1, :] = np.sin(1.5 * np.pi * x / x[-1])

    return p, x, y

nx_values = [11, 21, 41, 81]
l2_target = 1e-8

error = np.empty_like(nx_values, dtype=np.float)

for i, nx in enumerate(nx_values):
    p, x, y = laplace_IG(nx)

    p = laplace2d(p.copy(), l2_target)

    p_an = p_analytical(x, y)

    error[i] = L2_error(p, p_an)

    
plt.figure(figsize=(6,6))
plt.grid(True)
plt.xlabel(r'$n_x$', fontsize=18)
plt.ylabel(r'$L_2$-norm of the error', fontsize=18)

plt.loglog(nx_values, error, color='k', ls='--', lw=2, marker='o')
plt.axis('equal')

plt.show()

def laplace2d_neumann(p, l2_target):
    l2norm = 1
    pn = np.empty_like(p)
    iterations = 0

    while l2norm > l2_target:
        pn = p.copy()
        p[1:-1,1:-1] = (0.25 * (pn[1:-1,2:] + pn[1:-1, :-2] +
                        pn[2:, 1:-1] + pn[:-2, 1:-1]))

        ##2nd-order Neumann B.C. along x = L
        p[1:-1, -1] = 0.25 * (2*pn[1:-1,-2] + pn[2:, -1] + pn[:-2, -1])

        l2norm = L2_error(p, pn)

    return p


nx_values = [11, 21, 41, 81]
l2_target = 1e-8

error = np.empty_like(nx_values, dtype=np.float)


for i, nx in enumerate(nx_values):
    p, x, y = laplace_IG(nx)

    p = laplace2d_neumann(p.copy(), l2_target)

    p_an = p_analytical(x, y)

    error[i] = L2_error(p, p_an)

print("ola")
plt.figure(figsize=(6,6))
plt.grid(True)
plt.xlabel(r'$n_x$', fontsize=18)
plt.ylabel(r'$L_2$-norm of the error', fontsize=18)

plt.loglog(nx_values, error, color='k', ls='--', lw=2, marker='o')
plt.axis('equal')
plt.show()
