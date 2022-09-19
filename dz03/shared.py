import numpy as np
import matplotlib.pyplot as plt

def init_plot(lim_from=-3, lim_to=3):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    ax.set_box_aspect(1)
    ax.set_xlim([lim_from, lim_to])
    ax.set_ylim([lim_from, lim_to])
    ax.set_xlabel("$x_I$")
    ax.set_ylabel("$y_I$")
    ax.set_axisbelow(True)
    plt.grid()

    return fig, ax


def R(theta):
    return np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=float)