import numpy as np
import warnings

warnings.filterwarnings('ignore')


def correlation(f, g):
    n, m = f.size, g.size

    y = np.zeros(n+m-1, dtype=float)
    g = g[::-1]

    f, g = np.r_[f, np.zeros(y.size-n)], np.r_[g, np.zeros(y.size-m)]

    for i in range(y.size):
        for j in range(y.size):
            y[i] += f[j] * g[i-j]

    return y
