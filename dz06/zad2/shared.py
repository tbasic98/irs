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


def R_array(thetas):
    t = np.zeros((thetas.shape[0], 3, 3), dtype=float)
    t[:, 0, 0] = np.cos(thetas)
    t[:, 0, 1] = np.sin(thetas)
    t[:, 1, 0] = -np.sin(thetas)
    t[:, 1, 1] = np.cos(thetas)
    t[:, 2, 2] = np.ones(thetas.shape[0])
    return t

# wlm: "without last mask"
wlm = np.array([1, 1, 0], dtype=float)


