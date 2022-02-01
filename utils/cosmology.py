import numpy as np


def schechter(x, alpha=-1, mstar=-22):
    v = np.power(10, -0.4 * (x-mstar))
    return np.power(v, alpha + 1) * np.exp(-v)


def unnormed_schechter(x, alpha=-1, mstar=-22, n=1):
    return n * schechter(x, alpha, mstar)


def flat_schechter(x, mstar=-22, n=1):
    return n * schechter(x, -1, mstar)
