import numpy as np
import warnings

warnings.filterwarnings('ignore')


def linear_convolution(a, b):
    n = a.size
    m = b.size
    s = np.zeros(n+m-1, dtype=float)

    new_a = np.zeros(s.size)
    new_b = np.zeros(s.size)
    new_a[0:n] = a
    new_b[0:m] = b

    for n in range(s.size):
        for m in range(s.size):
            s[n] += new_a[m] * new_b[n - m]
    return s


def cyclic_convolution(a, b):
    s = np.zeros(a.size, dtype=float)

    for n in range(s.size):
        for m in range(s.size):
            s[n] += a[m] * b[n - m]
    return s
