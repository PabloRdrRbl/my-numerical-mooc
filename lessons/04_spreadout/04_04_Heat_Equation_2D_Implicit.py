import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import solve

from matplotlib import rcParams, cm
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


def constructMatrix(nx, ny, sigma):
    A = np.zeros(((nx - 2) * (ny - 2), (nx - 2) * (ny - 2)))

    row_number = 0  # row counter

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):

            # Corners
            if (i == 1) and (j == 1):  # BL corner
                A[row_number, row_number] = 1 / sigma + 4  # Set diagonal
                A[row_number, row_number + 1] = -1      # fetch i+1
                A[row_number, row_number + nx - 2] = -1   # fetch j+1

            elif i == nx - 2 and j == 1:  # BR corner
                A[row_number, row_number] = 1 / sigma + 3  # Set diagonal
                A[row_number, row_number - 1] = -1      # Fetch i-1
                A[row_number, row_number + nx - 2] = -1   # fetch j+1

            elif i == 1 and j == ny - 2:  # TL corner
                A[row_number, row_number] = 1 / sigma + 3   # Set diagonal
                A[row_number, row_number + 1] = -1        # fetch i+1
                A[row_number, row_number - (nx - 2)] = -1   # fetch j-1

            elif i == nx - 2 and j == ny - 2:  # TR corner
                A[row_number, row_number] = 1 / sigma + 2   # Set diagonal
                A[row_number, row_number - 1] = -1        # Fetch i-1
                A[row_number, row_number - (nx - 2)] = -1   # fetch j-1

               # Sides
            elif i == 1:  # Left boundary (Dirichlet)
                A[row_number, row_number] = 1 / sigma + 4  # Set diagonal
                A[row_number, row_number + 1] = -1      # fetch i+1
                A[row_number, row_number + nx - 2] = -1   # fetch j+1
                A[row_number, row_number - (nx - 2)] = -1  # fetch j-1

            elif i == nx - 2:  # Right boundary (Neumann)
                A[row_number, row_number] = 1 / sigma + 3  # Set diagonal
                A[row_number, row_number - 1] = -1      # Fetch i-1
                A[row_number, row_number + nx - 2] = -1   # fetch j+1
                A[row_number, row_number - (nx - 2)] = -1  # fetch j-1

            elif j == 1:  # Bottom boundary (Dirichlet)
                A[row_number, row_number] = 1 / sigma + 4  # Set diagonal
                A[row_number, row_number + 1] = -1      # fetch i+1
                A[row_number, row_number - 1] = -1      # fetch i-1
                A[row_number, row_number + nx - 2] = -1   # fetch j+1

            elif j == ny - 2:  # Top boundary (Neumann)
                A[row_number, row_number] = 1 / sigma + 3  # Set diagonal
                A[row_number, row_number + 1] = -1      # fetch i+1
                A[row_number, row_number - 1] = -1      # fetch i-1
                A[row_number, row_number - (nx - 2)] = -1  # fetch j-1

            # Interior points
            else:
                A[row_number, row_number] = 1 / sigma + 4  # Set diagonal
                A[row_number, row_number + 1] = -1      # fetch i+1
                A[row_number, row_number - 1] = -1      # fetch i-1
                A[row_number, row_number + nx - 2] = -1   # fetch j+1
                A[row_number, row_number - (nx - 2)] = -1  # fetch j-1

            row_number += 1  # Jump to next row of the matrix!

    return A


def generateRHS(nx, ny, sigma, T, T_bc):
    RHS = np.zeros((nx - 2) * (ny - 2))

    row_number = 0

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):

            # Corners
            # Bottom left corner (Dirichlet down and left)
            if i == 1 and j == 1:
                RHS[row_number] = T[j, i] * 1 / sigma + 2 * T_bc

            # Bottom right corner (Dirichlet down, Neumann right)
            elif i == nx - 2 and j == 1:
                RHS[row_number] = T[j, i] * 1 / sigma + T_bc

            # Top left corner (Neumann up, Dirichlet left)
            elif i == 1 and j == ny - 2:
                RHS[row_number] = T[j, i] * 1 / sigma + T_bc

            # Top right corner (Neumann up and right)
            elif i == nx - 2 and j == ny - 2:
                RHS[row_number] = T[j, i] * 1 / sigma

            # Sides
            elif i == 1:  # Left boundary (Dirichlet)
                RHS[row_number] = T[j, i] * 1 / sigma + T_bc

            elif i == nx - 2:  # Right boundary (Neumann)
                RHS[row_number] = T[j, i] * 1 / sigma

            elif j == 1:  # Bottom boundary (Dirichlet)
                RHS[row_number] = T[j, i] * 1 / sigma + T_bc

            elif j == ny - 2:  # Top boundary (Neumann)
                RHS[row_number] = T[j, i] * 1 / sigma

            # Interior points
            else:
                RHS[row_number] = T[j, i] * 1 / sigma

            row_number += 1  # Jump to next row!

    return RHS


def map_1Dto2D(nx, ny, T_1D, T_bc):
    T = np.zeros((ny, nx))

    row_number = 0

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            T[j, i] = T_1D[row_number]
            row_number += 1

    # Dirichlet BC
    T[0, :] = T_bc
    T[:, 0] = T_bc

    # Neumann BC
    T[-1, :] = T[-2, :]
    T[:, -1] = T[:, -2]

    return T


def btcs_2D(T, A, nt, sigma, T_bc, nx, ny, dt):
    j_mid = int((np.shape(T)[0]) / 2)
    i_mid = int((np.shape(T)[1]) / 2)

    for t in range(nt):
        Tn = T.copy()
        b = generateRHS(nx, ny, sigma, Tn, T_bc)

        T_interior = solve(A, b)
        T = map_1Dto2D(nx, ny, T_interior, T_bc)

        # Check if we reached T=70C
        if T[j_mid, i_mid] >= 70:
            print("Center of plate reached 70C at time {0:.2f}s, in time step {1:d}.".format(
                dt * t, t))
            break

    if T[j_mid, i_mid] < 70:
        print("Center has not reached 70C yet, it is only {0:.2f}C.".format(
            T[j_mid, i_mid]))

    return T


# Model
alpha = 1e-4

L = 1.0e-2
H = 1.0e-2

nx = 21
ny = 21
nt = 300

dx = L / (nx - 1)
dy = H / (ny - 1)

x = np.linspace(0, L, nx)
y = np.linspace(0, H, ny)

T_bc = 100

Ti = np.ones((ny, nx)) * 20
Ti[0, :] = T_bc
Ti[:, 0] = T_bc

sigma = 0.25
A = constructMatrix(nx, ny, sigma)

dt = sigma * min(dx, dy)**2 / alpha
T = btcs_2D(Ti.copy(), A, nt, sigma, T_bc, nx, ny, dt)

plt.figure(figsize=(7, 7))
plt.contourf(x, y, T, 20, cmap=cm.viridis)

plt.show()
