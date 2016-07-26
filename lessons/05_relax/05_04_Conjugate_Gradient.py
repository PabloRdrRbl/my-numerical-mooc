import numpy as np

from laplace_helper import plot_3D, L2_rel_error
from cg_helper import poisson_2d, p_analytical


# Grid patameters
nx = 101
ny = 101
xmin = 0
xmax = 1
ymin = -0.5
ymax = 0.5

l2_target = 1e-10

# Spacing
dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

# Mesh
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

# Source
L = xmax - xmin
b = -2 * (np.pi / L)**2 * np.sin(np.pi * X / L) * np.cos(np.pi * Y / L)

# Initialization
p_i = np.zeros((ny, nx))

# Analytical solution
pan = p_analytical(X, Y, L)

# Calculating alpha


def steepest_descent_2d(p, b, dx, dy, l2_target):
    ny, nx = p.shape
    r = np.zeros((ny, nx))  # residual
    Ar = np.zeros((ny, nx))  # to store result of matrix multiplication

    l2_norm = 1
    iterations = 0
    l2_conv = []

    # Iterations
    while l2_norm > l2_target:
        pd = p.copy()

        r[1:-1, 1:-1] = (b[1:-1, 1:-1] * dx**2 + 4 * pd[1:-1, 1:-1] -
                         pd[1:-1, 2:] - pd[1:-1, :-2] - pd[2:, 1:-1] -
                         pd[:-2, 1:-1])

        Ar[1:-1, 1:-1] = (-4 * r[1:-1, 1:-1] + r[1:-1, 2:] + r[1:-1, :-2] +
                          r[2:, 1:-1] + r[:-2, 1:-1])

        rho = np.sum(r * r)
        sigma = np.sum(r * Ar)
        alpha = rho / sigma

        p = pd + alpha * r

        # BCs automatically enforced

        l2_norm = L2_rel_error(pd, p)
        iterations += 1
        l2_conv.append(l2_norm)

    print('Number of SD iterations: {0:d}'.format(iterations))
    return p, l2_conv


p, l2_conv = steepest_descent_2d(p_i.copy(), b, dx, dy, l2_target)
L2_rel_error(p, pan)


def conjufate_gradient_2d(p, b, dx, dy, l2_target):
    ny, nx = p.shape
    r = np.zeros((ny, nx))  # Residual
    Ad = np.zeros((ny, nx))  # To store result of matrix multiplication

    l2_norm = 1
    iterations = 0
    l2_conv = []

    # Step-0 We compute the initial residual and
    # the first search direction is just this residual

    r[1:-1, 1:-1] = (b[1:-1, 1:-1] * dx**2 + 4 * p[1:-1, 1:-1] -
                     p[1:-1, 2:] - p[1:-1, :-2] - p[2:, 1:-1] - p[:-2, 1:-1])

    d = r.copy()
    rho = np.sum(r * r)

    Ad[1:-1, 1:-1] = (-4 * d[1:-1, 1:-1] + d[1:-1, 2:] + d[1:-1, :-2] +
                      d[2:, 1:-1] + d[:-2, 1:-1])

    sigma = np.sum(d * Ad)

    # Iterations
    while l2_norm > l2_target:
        pk = p.copy()
        rk = r.copy()
        dk = d.copy()

        alpha = rho / sigma

        p = pk + alpha * dk
        r = rk - alpha * Ad

        rhop1 = np.sum(r * r)
        beta = rhop1 / rho
        rho = rhop1

        d = r + beta * dk
        Ad[1:-1, 1:-1] = -4 * d[1:-1, 1:-1] + d[1:-1, 2:] + d[1:-1, :-2] + \
            d[2:, 1:-1] + d[:-2, 1:-1]
        sigma = np.sum(d * Ad)

        # BCs are automatically enforced

        l2_norm = L2_rel_error(pk, p)
        iterations += 1
        l2_conv.append(l2_norm)

    print('Number of CG iterations: {0:d}'.format(iterations))
    return p, l2_conv


p, l2_conv = conjugate_gradient_2d(p_i.copy(), b, dx, dy, l2_target)
L2_rel_error(p, pan)

# More difficult Poisson problems
b = (np.sin(pi * X / L) * np.cos(pi * Y / L) +
     np.sin(6 * pi * X / L) * np.sin(6 * pi * Y / L))

p, l2_conv = poisson_2d(p_i.copy(), b, dx, dy, l2_target)

p, l2_conv = steepest_descent_2d(p_i.copy(), b, dx, dy, l2_target)

p, l2_conv = conjugate_gradient_2d(p_i.copy(), b, dx, dy, l2_target)
