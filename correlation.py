import numpy as np
import warnings

warnings.filterwarnings('ignore')


def auto_corr(f):
    k = f.size
    y = np.zeros(k, dtype=float)

    for i in range(k):
        for j in range(k - i):
            y[i] += f[j] * f[j + i]
        y[i] /= f.size - i + 1

    return y


def cross_corr(f, g):
    if f.size == g.size:
        k = f.size
        y = np.zeros(k, dtype=float)

        for i in range(k):
            for j in range(k - i):
                y[i] += f[j] * g[j + i]
            y[i] /= k - i + 1

        return y
    else:
        raise ValueError('Signals must be one dimension')
