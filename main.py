import numpy as np
import matplotlib.pyplot as plt
import warnings
from correlation import correlation
from convolution import convolution
from scipy import signal

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

if __name__ == "__main__":
    y = np.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
    sig_noise = y + np.random.randn(len(y))
    corr = correlation(y, sig_noise)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(y)
    ax1.set_title('Signal y')
    ax2.plot(sig_noise)
    ax2.set_title('Signal with noise')
    ax3.plot(corr)
    ax3.set_title('My corr these signals')
    ax4.plot(signal.correlate(y, sig_noise))
    ax4.set_title('Scipy corr')
    plt.show()

    sig = np.repeat([0., 1., 0.], 100)
    win = signal.hanning(50)
    filtered = convolution(sig, win)

    fig2, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(sig)
    ax1.set_title('pulse signal')
    ax2.plot(win)
    ax2.set_title('Filter impulse')
    ax3.plot(filtered)
    ax3.set_title('My convolve these signals')
    ax4.plot(signal.convolve(sig, win))
    ax4.set_title('Scipy convolve')
    plt.show()

    y = np.random.randint(-5, 5, 256)
    corr = correlation(y, y)
    plt.plot(corr)
    plt.show()
