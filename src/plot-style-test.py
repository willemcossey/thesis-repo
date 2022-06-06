import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

mpl.style.use("grayscale_adjusted.mplstyle")


x = np.linspace(0, 1, 100)
n_plots = 20
y = [np.array([x + 0.1 * i]) for i in range(n_plots)]

plt.figure()
for p in range(n_plots):
    plt.plot(x, y[p].reshape(-1))
plt.show(block=True)
