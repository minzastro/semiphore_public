#!/usr/bin/env python3
import numpy as np


def fun(x, a, b, n, h):
    return np.abs(n) * np.power(x / b, a) * np.exp(-np.power(x/b, a)) + \
        np.abs(h)


def fun2(x, a, b, n, h, s1):
    return fun(x, a, b, n, h) + np.abs(s1) / (1 + np.exp(100 * (x - b)))

