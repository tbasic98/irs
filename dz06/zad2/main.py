import numpy as np
from shared import init_plot, R, R_array
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge, Circle
from sampler import sampler
from motion import sample_motion_model_odometry
from robot import Robot

sign = -1
def fake_algorithm_for_velocity(i):
    v = 0.8
    omega = 3 / 4 * np.pi
    if i % 30 == 0:
        global sign
        sign *= -1
    return np.array([v, 0, sign * omega], dtype=float)


def update_wedge(wedge_patch, I_ksi):
    wedge_patch.center = I_ksi[:2]
    wedge_patch.theta1 = np.rad2deg(I_ksi[2]) + 10
    wedge_patch.theta2 = np.rad2deg(I_ksi[2]) - 10
    wedge_patch._recompute_path()


def animate(i, robot, estimated_robot, enc_robot, shapes, dt):
    R_ksi_dot = fake_algorithm_for_velocity(i)
    phis_dot = robot.R_inverse_kinematics(R_ksi_dot)
    I_ksi_dot = robot.forward_kinematics(phis_dot)
    robot.update_state(I_ksi_dot, dt)
    update_wedge(shapes[0], robot.I_ksi)

    deltas_c = robot.read_encoders(phis_dot, dt)
    phis_dot_enc = robot.enc_deltas_to_phis_dot(deltas_c, dt)
    I_ksi_dot_enc = robot.forward_kinematics(phis_dot_enc)

    # prethodno stanje dobiveno čitanjem enkodera
    I_ksi_prev = enc_robot.I_ksi.copy()
    enc_robot.update_state(I_ksi_dot_enc, dt)
    # trenutno stanje dobiveno čitanjem enkodera
    I_ksi = enc_robot.I_ksi.copy()

    # za algoritam
    u_current = np.vstack((I_ksi_prev, I_ksi))
    # x_prev: particles

    update_wedge(shapes[3], enc_robot.I_ksi)

    global particles
    """
    particles += np.tensordot(
        R_ksi_dot, R_array(particles[:, 2]), axes=(0, 1)
    ) * dt
    particles += np.random.normal(0, var_movement, (M, 3))
    """

    particles = sample_motion_model_odometry(u_current, particles)

    particles[:, 2] = np.arctan2(
        np.sin(particles[:, 2]), np.cos(particles[:, 2])
    )
    shapes[1].set_offsets(particles[:, :2])

    if i % 1 == 0:
        # reziduali za robota
        m_res = landmarks - robot.I_ksi[:2]
        phi_res = landmarks_phis - robot.I_ksi[2]
        m_res += np.random.normal(0, var_measurement, (landmarks.shape[0], 2))
        phi_res += np.random.normal(0, var_measurement, landmarks.shape[0])

        # reziduali za svaku česticu
        particles_reshaped = np.reshape(particles, (M, 1, 3))
        p_m_res = landmarks - particles_reshaped[:, :, :2]
        p_m_res += np.random.normal(0, var_measurement, (M, landmarks.shape[0], 2))
        p_phi_res = landmarks_phis - particles_reshaped[:, :, 2]
        p_phi_res += np.random.normal(0, var_measurement, (M, landmarks.shape[0]))

        diff_pos_r_p = np.linalg.norm(m_res - p_m_res, axis=(1, 2))
        diff_phi_r_p = np.linalg.norm(phi_res - p_phi_res, axis=1)
        diff = diff_pos_r_p + diff_phi_r_p

        x_0 = y_1 = np.min(diff)
        x_1 = y_0 = np.max(diff)
        sim = (y_1 - y_0) / (x_1 - x_0) * (diff - x_0) + y_0
        weights = sim_normalized = sim / np.sum(sim)
        indices = sampler(weights)
        particles = particles[indices]

        mean = np.mean(particles, axis=0)
        mean[2] = np.arctan2(np.sin(mean[2]), np.cos(mean[2]))
        estimated_robot.I_ksi = mean
        update_wedge(shapes[2], estimated_robot.I_ksi)





if __name__ == "__main__":
    fig, ax = init_plot(lim_from=-3.0, lim_to=3.0)

    num_frames = 150
    fps = 30
    dt = 1 / fps

    shapes = []

    robot = Robot(np.array([0.7, 1.5, 4 / 3 * np.pi], dtype=float))
    robot_shape = Wedge(
        robot.I_ksi[:2],
        0.2,
        np.rad2deg(robot.I_ksi[2]) + 10,
        np.rad2deg(robot.I_ksi[2]) - 10,
        facecolor="#039be588", zorder=2
    )
    shapes.append(ax.add_patch(robot_shape))

    landmarks = np.random.randn(7, 2)
    landmarks_phis = np.arctan2(landmarks[:, 1], landmarks[:, 0])
    for landmark in landmarks:
        ax.add_patch(
            Circle(xy=landmark, radius=0.1, facecolor="#f4511e88")
        )

    M = 1000
    particles = np.random.randn(M, 3)
    particles[:, 2] = np.arctan2(
        np.sin(particles[:, 2]),
        np.cos(particles[:, 2])
    )
    shapes.append(
        ax.scatter(
            particles[:, 0], particles[:, 1], s=0.5, c="#ff1744"
        )
    )

    var_movement = 0.01
    var_measurement = 0.1

    mean = np.mean(particles, axis=0)
    mean[2] = np.arctan2(np.sin(mean[2]), np.cos(mean[2]))

    estimated_robot = Robot(mean)
    estimated_robot_shape = Wedge(
        estimated_robot.I_ksi[:2],
        0.2,
        np.rad2deg(estimated_robot.I_ksi[2]) + 10,
        np.rad2deg(estimated_robot.I_ksi[2]) - 10,
        facecolor="#00c85388", zorder=2
    )
    shapes.append(ax.add_patch(estimated_robot_shape))

    enc_robot = Robot(robot.I_ksi.copy())
    enc_robot_shape = Wedge(
        enc_robot.I_ksi[:2],
        0.2,
        np.rad2deg(enc_robot.I_ksi[2]) + 10,
        np.rad2deg(enc_robot.I_ksi[2]) - 10,
        facecolor="#fdd83588", zorder=2
    )
    shapes.append(ax.add_patch(enc_robot_shape))


    ani = FuncAnimation(
        fig,
        animate,
        fargs=(robot, estimated_robot, enc_robot, shapes, dt),
        frames=num_frames,
        interval=dt * 1000,
        repeat=False,
        blit=False,
        init_func=lambda: None
    )
    # ani.save('2wd.gif', writer='imagemagick')

    plt.show()