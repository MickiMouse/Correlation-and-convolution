import numpy as np
import warnings

warnings.filterwarnings('ignore')


def convolution(a, b):
    n, m = a.size, b.size

    s = np.zeros(n+m-1, dtype=float)

    _a, _b = np.zeros(s.size), np.zeros(s.size)
    _a[0:n], _b[0:m] = a, b

    for n in range(s.size):
        for m in range(s.size):
            s[n] += _a[m] * _b[n - m]

    return s
