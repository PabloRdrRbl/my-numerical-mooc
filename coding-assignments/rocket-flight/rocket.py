import numpy as np


# Physical parameters
# h --> the altitude of the rocket
ms = 50  # mass of the rocket shell (kg)
g = 9.81  # gravity acceleration (m/s^2)
rho = 1.091  # average density, cte (kg/m^3)
r = 0.5  #  radio of the rocket (m)
A = np.pi * r**2  # cross sectional area of the rocket (m^2)
ve = 325  # exhaust speed (m/s)
CD = 0.15  # drag coefficient (adim)
mpo = 100  # mass of the rocket propellant at time t=0 (kg)

# Grid parameters
dt = 0.1  # timestep (s)


def mp_rate(t):
    """
    Calculates the propellant burn rate (kg/s)
    """
    if (t < 5):
        me = 20 * t  # me: mass exhausted and 20 (kg/s) propellant burn rate
    else:
        me = 20 * 5  # before 5 (s) the rocket shuts down

    return me


def mp(t):
    """
    Calculates the remaining propellant in the rocket.
    """
    return mpo - mp_rate(t)


if __name__ == '__main__':
    print('-- Remaining fuel --')
    print('At a time t = 3.2 (s) the mass (kg) of the rocket '
          'propellant remaining in the roket is: %.4f (kg)' %
          (mp(3.2)))
