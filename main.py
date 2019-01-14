import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import warnings
from correlation import auto_corr, cross_corr
from convolution import linear_convolution, cyclic_convolution

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

if __name__ == "__main__":
    Fs = 200
    Ts = 1 / Fs
    t = np.arange(0, 1, Ts)

    noise = np.random.randint(-10, 10, 200)

    f = 20
    y = sp.sin(2*sp.pi*f*t) + noise

    plt.plot(t, y)
    plt.title('Signal with noise')
    plt.show()

    plt.plot(t, auto_corr(y))
    plt.title('Auto correlation function')
    plt.show()

    y1 = sp.cos(2*sp.pi*f*t) * sp.exp(-t)
    y2 = sp.sin(2*sp.pi*f*t)

    plt.plot(t, y1)
    plt.plot(t, y2)
    plt.title('Signals')
    plt.show()

    plt.plot(t, cross_corr(y1, y2))
    plt.title('Cross correlation function')
    plt.show()

    plt.plot(t, cyclic_convolution(y1, y2))
    plt.title('Cyclic convolution')
    plt.show()

    plt.plot(linear_convolution(y1, y2))
    plt.title('Linear convolution')
    plt.show()
