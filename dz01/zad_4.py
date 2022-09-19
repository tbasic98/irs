from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy.linalg


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


def fake_algorithm_transl_vel(i, R, frame):
    if frame == "k":
        if i < 59:
            i_p_k_dot = np.array([0, -1, 0], dtype=float)
        else:
            i_p_k_dot = R @ np.array([2*np.pi, 0, 0], dtype=float)
        return i_p_k_dot
    else:
        return R @ np.array([0, 0, np.pi/4], dtype=float)


def fake_algorithm_angular_vel(i, frame):
    if frame == "k":
        if i < 29:
            k_omega_k = np.array([2*np.pi, 0, 0], dtype=float)
        elif i < 59:
            k_omega_k = np.array([0, 2*np.pi, 0], dtype=float)
        else:
            k_omega_k = np.array([0, 0, -2*np.pi], dtype=float)
        return k_omega_k
    else:
        return np.array([0, -np.pi/6, 0], dtype=float)


def twist2screw(twist):
    if (twist[0:3] == 0).all():
        normv = np.linalg.norm(twist[3:6])
        if normv == 0:
            return twist, normv
        return twist/normv, normv
    else:
        normomega = np.linalg.norm(twist[0:3])
        return twist/normomega, normomega


def screw2T(S, theta):
    omega = S[0:3]
    v = S[3:6]
    if np.linalg.norm(omega) - 1 <= 1e-8:
        skewomega = np.array([
            [0, -omega[2, 0], omega[1, 0]],
            [omega[2, 0], 0, -omega[0, 0]],
            [-omega[1, 0], omega[0, 0], 0]
        ], dtype = float)
        vdio = (np.eye(3)*theta + (1 - np.cos(theta))*skewomega + (theta - np.sin(theta))*skewomega @ skewomega) @ v
        return np.concatenate((np.concatenate((scipy.linalg.expm(skewomega*theta), vdio.reshape(3, 1)), axis=1, dtype=float), np.array([0, 0, 0, 1], dtype=float).reshape(1, 4)), dtype=float)
    else:
        return np.concatenate((np.concatenate((np.eye(3), (v*theta).reshape(3, 1)), axis=1, dtype=float), np.array([0, 0, 0, 1], dtype=float).reshape(1, 4)), dtype=float)


def T_update(T, j_p_i_dot, i_omega_i, dt):
    j_omega_i = T[0:3, 0:3] @ i_omega_i
    v_j = j_p_i_dot.reshape(3, 1) - np.cross(j_omega_i.reshape(1, 3), T[0:3, 3].reshape(1, 3)).T
    S, theta = twist2screw(np.concatenate((j_omega_i.reshape(3, 1), v_j), dtype=float).reshape(6, 1))
    return screw2T(S, theta*dt)


def animate(i, frames, quivers, configurations, dt):
    k_omega_k = fake_algorithm_angular_vel(i, "k")
    i_p_k_dot = fake_algorithm_transl_vel(i, configurations[2][0:3, 0:3], "k")

    i_omega_i = fake_algorithm_angular_vel(i, "i")
    j_p_i_dot = fake_algorithm_transl_vel(i, configurations[1][0:3, 0:3], "i")

    T_i = configurations[1]
    T_k = configurations[2]

    T_k = T_update(T_k, i_p_k_dot, k_omega_k, dt) @ T_k
    T_i = T_update(T_i, j_p_i_dot, i_omega_i, dt) @ T_i

    configurations[2] = T_k
    configurations[1] = T_i

    segsk = np.array(frame2quiver(frames[2] @ (T_i @ T_k).T, diff=0)).reshape(6, -1)
    new_segsk = [[[x,y,z],[u,v,w]] for x,y,z,u,v,w in zip(*segsk.tolist())]

    segsi = np.array(frame2quiver(frames[1] @ T_i.T, diff=0)).reshape(6, -1)
    new_segsi = [[[x,y,z],[u,v,w]] for x,y,z,u,v,w in zip(*segsi.tolist())]

    quivers[1].set_segments(new_segsi)
    quivers[2].set_segments(new_segsk)


if __name__ == "__main__":
    fig, ax = init_plot()

    fps = 60
    dt = 1/60
    num_frames = 179 # dva puta ide i = 0

    frame_j = np.array([
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1]
    ], dtype=float)
    quiver_j = ax.quiver3D(*frame2quiver(frame_j), arrow_length_ratio=0, colors="#212529")
    j_T_j = np.eye(4, dtype=float)

    frame_i = np.copy(frame_j)
    j_T_i = np.eye(4, dtype=float)
    j_T_i[0, 3] = 1.5
    quiver_i = ax.quiver3D(*frame2quiver(frame_i @ j_T_i.T), arrow_length_ratio=0, colors=["r", "g", "b"])

    frame_k = np.copy(frame_i)
    quiver_k = ax.quiver3D(*frame2quiver(frame_k @ j_T_i.T), arrow_length_ratio=0, colors=["y", "purple", "orange"])
    i_T_k = np.eye(4, dtype=float)

    frames = [frame_j, frame_i, frame_k]
    quivers = [quiver_j, quiver_i, quiver_k]
    configurations = [j_T_j, j_T_i, i_T_k]


    ani = FuncAnimation(fig, animate, frames = num_frames, fargs=(frames, quivers, configurations, dt), repeat=False, interval = dt * 1000, blit=False)
    #ani.save('zad_4.gif', writer='imagemagick')

    plt.show()
