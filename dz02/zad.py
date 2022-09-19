import numpy as np
import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


def DH(params):
    a = params[0]
    alpha = params[1]
    d = params[2]
    theta = params[3]
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ], dtype=float)


def update_plot():
    global axes_list
    global lines_list

    frame_last = None
    for i in range(manipulator_params.shape[0]):
        T_i = np.eye(4, dtype=float)
        for j in range(i + 1):
            T_i = T_i @ DH(manipulator_params[j])
        frame_i = frame.dot(T_i.transpose())

        if 0 < i < manipulator_params.shape[0] - 1:
            axes_list[i]._offsets3d = [frame_i[0, 0]], [frame_i[0, 1]], [frame_i[0, 2]]
        else:
            axes_list[i]._offsets3d = frame_i[:, 0], frame_i[:, 1], frame_i[:, 2]

        if i > 0:
            lines_list[i-1].set_data_3d([frame_last[0, 0], frame_i[0, 0]], [frame_last[0, 1], frame_i[0, 1]], [frame_last[0, 2], frame_i[0, 2]])
        frame_last = frame_i
    fig.canvas.draw_idle()
    plt.pause(dt)


if __name__ == "__main__":
    #okvir
    frame = np.array([
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1]
    ], dtype=float)
    # scale-at ću osi na 20 cm
    # da možemo pratiti što se događa
    frame[:, :3] = frame[:, :3] * 0.2

    manipulator_params = np.load("./stanford_params.npy")
    joint_infos = np.load("./stanford_joints.npy")
    colors = np.load("./stanford_colors.npy")
    joint_labels = np.load("./stanford_labels.npy")

    dt = 0.01

    axes_list = []
    lines_list = []

    frame_last = None

    # 1. OVO SU FRAMEOVI SVAKE POVEZNICE
    for i in range(manipulator_params.shape[0]):
        T_i = np.eye(4, dtype=float)
        for j in range(i + 1):
            T_i = T_i @ DH(manipulator_params[j])
        frame_i = frame.dot(T_i.transpose())
        if i == manipulator_params.shape[0] - 1:
            T_e = T_i

        if 0 < i < manipulator_params.shape[0] - 1:
            axes_list.append(ax.scatter(frame_i[0, 0], frame_i[0, 1], frame_i[0, 2], color=colors[i], marker="."))
        else:
            axes_list.append(ax.scatter(frame_i[:, 0], frame_i[:, 1], frame_i[:, 2], color=colors[i], marker="."))

        # 2. LINIJE KOJE POVEZUJU (i - 1). i i. poveznicu
        if i > 0:
            line = ax.plot([frame_last[0, 0], frame_i[0, 0]], [frame_last[0, 1], frame_i[0, 1]], [frame_last[0, 2], frame_i[0, 2]], color=colors[i], linewidth=4, solid_capstyle="round")[0]
            lines_list.append(line)
        frame_last = frame_i

    # 3. POZICIJA I ORIJENTACIJA CILJA

    T_g = np.load("./stanford_goal.npy")

    frame_g = frame.dot(T_g.transpose())
    goal = ax.scatter(frame_g[:, 0], frame_g[:, 1], frame_g[:, 2], color="#FF1744", marker="*")
    plt.draw()

    # 4. Inverzna kinematika Stanford manipulatora

    p_w = T_g[:3, 3] - manipulator_params[6, 2] * T_g[:3, 2]
    d_2 = manipulator_params[2, 2]

    theta_1 = 2*np.arctan2(-p_w[0] + np.sqrt(p_w[0]**2 + p_w[1]**2 - d_2**2), d_2 + p_w[1])
    theta_2 = np.arctan2(p_w[0]*np.cos(theta_1) + p_w[1]*np.sin(theta_1), p_w[2])
    d_3 = np.sqrt((p_w[0]*np.cos(theta_1) + p_w[1] * np.sin(theta_1))**2 + p_w[2]**2)

    manipulator_params_g = manipulator_params.copy()
    manipulator_params_g[1, 3] = theta_1
    manipulator_params_g[2, 3] = theta_2
    manipulator_params_g[3, 2] = d_3

    R_0_3 = np.eye(3, dtype=float)
    for i in [1, 2, 3]:
        R_0_3 = R_0_3 @ DH(manipulator_params_g[i])[:3, :3]

    R_3_6 = R_0_3.T @ T_g[:3, :3]

    theta_4 = np.arctan2(R_3_6[1, 2], R_3_6[0, 2])
    theta_5 = np.arctan2(np.sqrt(R_3_6[0, 2]**2 + R_3_6[1, 2]**2), R_3_6[2, 2])
    theta_6 = np.arctan2(R_3_6[2, 1], -R_3_6[2, 0])

    manipulator_params_g[4, 3] = theta_4
    manipulator_params_g[5, 3] = theta_5
    manipulator_params_g[6, 3] = theta_6
    #plt.pause(8)

    while True:
        T_e = np.eye(4, dtype=float)
        for params in manipulator_params:
            T_e = T_e @ DH(params)
        if np.linalg.norm(T_g - T_e) < 1e-2:
            break

        for k, j_info in enumerate(joint_infos):
            i = int(j_info[2])
            j = int(j_info[3])
            p_gain = 1
            q_dot = p_gain * (manipulator_params_g[i, j] - manipulator_params[i, j])/dt
            q_dot = np.sign(q_dot) * min(j_info[4], abs(q_dot))
            new_value = manipulator_params[i, j] + q_dot * dt
            if j_info[0] <= new_value <= j_info[1]:
                manipulator_params[i, j] = new_value
            else:
                # print("Unable to move further")
                0
        update_plot()

    while True:
        if plt.waitforbuttonpress():
            break