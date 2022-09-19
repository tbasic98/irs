from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def init_plot(lim_from=-3, lim_to=3):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect((1,1,1))
    ax.view_init(azim=30, elev=20)
    ax.set_xlim3d([lim_from, lim_to])
    ax.set_ylim3d([lim_from, lim_to])
    ax.set_zlim3d([lim_from, lim_to])
    ax.set_xlabel("$x_j$")
    ax.set_ylabel("$y_j$")
    ax.set_zlabel("$z_j$")
    return fig, ax


def frame2quiver(frame, diff=1):
    x = np.repeat(frame[0, 0], 3)
    y = np.repeat(frame[0, 1], 3)
    z = np.repeat(frame[0, 2], 3)
    u = frame[1:, 0] - frame[0, 0] * diff
    v = frame[1:, 1] - frame[0, 1] * diff
    w = frame[1:, 2] - frame[0, 2] * diff
    return x, y, z, u, v, w


def fake_algorithm_transl_vel(i):
    j_p_i_dot = np.array([1, 1, 1], dtype=float)
    j_p_i_dot[2] = 2 * np.sin(4 * i / 120 * 2 * np.pi)
    return j_p_i_dot


def fake_algorithm_angular_vel(i):
    if i < 60:
        i_omega_i = np.array([np.pi, 0, 0], dtype=float)
    else:
        i_omega_i = np.array([0, 2 * np.pi, 0], dtype=float)
    return i_omega_i


def EA2R(EA):
    alpha = EA[0]
    beta = EA[1]
    gamma = EA[2]
    c = np.cos
    s = np.sin
    return np.array([
        [c(alpha)*c(beta), c(alpha)*s(beta)*s(gamma) - s(alpha)*c(gamma), c(alpha)*s(beta)*c(gamma) + s(alpha)*s(gamma)],
        [s(alpha)*c(beta), s(alpha)*s(beta)*s(gamma) + c(alpha)*c(gamma), s(alpha)*s(beta)*c(gamma) - c(alpha)*s(gamma)],
        [-s(beta), c(beta)*s(gamma), c(beta)*c(gamma)]
    ], dtype=float)


def R2EA(R):
    EA = np.zeros((3,))
    EA[1] = np.arctan2(-R[2, 0], np.sqrt(np.square(R[0, 0]) + np.square(R[1, 0])))
    EA[0] = np.arctan2(R[1, 0]/np.cos(EA[1]), R[0, 0]/np.cos(EA[1]))
    EA[2] = np.arctan2(R[2, 1]/np.cos(EA[1]), R[2, 2]/np.cos(EA[1]))
    return EA


def i_omega_i2EA_dot(i_omega_i, EA):
    beta = EA[1]
    gamma = EA[2]
    return (1/np.cos(beta)) * np.array([
        [0, np.sin(gamma), np.cos(gamma)],
        [0, np.cos(gamma)*np.cos(beta), -np.sin(gamma)*np.cos(beta)],
        [np.cos(beta), np.sin(gamma)*np.sin(beta), np.cos(gamma)*np.sin(beta)]
    ], dtype=float) @ i_omega_i


def animate(i, frames, quivers, configurations, dt):
    i_omega_i = fake_algorithm_angular_vel(i)
    T = configurations[1]
    EA = R2EA(T[0:3, 0:3])
    EA += i_omega_i2EA_dot(i_omega_i, EA) * dt
    T[0:3, 0:3] = EA2R(EA)
    T[0:3, 3] += fake_algorithm_transl_vel(i) * dt
    configurations[1] = T

    segs = np.array(frame2quiver(frames[1] @ T.T, diff=0)).reshape(6, -1)
    new_segs = [[[x,y,z],[u,v,w]] for x,y,z,u,v,w in zip(*segs.tolist())]
    quivers[1].set_segments(new_segs)


if __name__ == "__main__":
    fig, ax = init_plot()

    fps = 60
    dt = 1/60
    num_frames = 120

    frame_j = np.array([
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1]
    ], dtype=float)
    quiver_j = ax.quiver3D(*frame2quiver(frame_j), arrow_length_ratio=0, colors="#212529")
    j_T_j = np.eye(4, dtype=float)

    frame_i = np.copy(frame_j)
    quiver_i = ax.quiver3D(*frame2quiver(frame_i), arrow_length_ratio=0, colors="#00B6BC")
    j_T_i = np.eye(4, dtype=float)

    frames = [frame_j, frame_i]
    quivers = [quiver_j, quiver_i]
    configurations = [j_T_j, j_T_i]

    ani = FuncAnimation(fig, animate, frames = num_frames, fargs=(frames, quivers, configurations, dt), repeat=False, interval = dt * 1000, blit=False)
    #ani.save('zad_1.gif', writer='imagemagick')

    plt.show()
