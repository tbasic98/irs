import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shared import init_plot, R
from matplotlib.patches import Polygon


def fake_algorithm_for_velocity(i, theta):
    I_ksi_dot = np.zeros(3, dtype=float)
    if i < 60:
        v = 0.5
        omega = 2 / 3 * np.pi
        R_ksi_dot = np.array([v, 0, omega], dtype=float)
        I_ksi_dot = R(theta).T @ R_ksi_dot
    elif i < 120:
        v = 0.5
        omega = -2 / 3 * np.pi
        R_ksi_dot = np.array([v, 0, omega], dtype=float)
        I_ksi_dot = R(theta).T @ R_ksi_dot
    elif i < 180:
        v = 0.66
        omega = 0
        R_ksi_dot = np.array([v, 0, omega], dtype=float)
        I_ksi_dot = R(theta).T @ R_ksi_dot
    elif i < 240:
        I_ksi_dot = np.array([0.5, -0.3, np.pi], dtype=float)
    elif i < 300:
        I_ksi_dot = np.array([0.3, -0.5, 0], dtype=float)
    return I_ksi_dot


def animate(i, configurations, shapes, dt):
    I_ksi = configurations[0]
    theta = I_ksi[2]

    I_ksi_dot = fake_algorithm_for_velocity(i, theta)
    I_ksi += I_ksi_dot * dt

    for i in range(5):
        shapes[i].xy = ((chassis_points + np.array([l/2, 0, 0], dtype=float)) @ R(I_ksi[2] + (i*2*np.pi/5)) + I_ksi * wlm)[:, :2]
    for j, i in enumerate(range(5, 15, 2)):
        alpha = alphas[j]
        beta_j_dot = -1/d * np.array([np.cos(alpha + betas[j]), np.sin(alpha + betas[j]), d + l * np.sin(betas[j])], dtype=float) @ R(theta) @ I_ksi_dot
        betas[j] += beta_j_dot*dt

        wheel_points = init_office_chair_wheels(r, d, alpha, betas[j])
        d_thing_points = init_office_chair_d_thing(d, alpha, betas[j])
        shapes[i].xy = (wheel_points @ R(I_ksi[2]) + I_ksi * wlm)[:, :2]
        shapes[i + 1].xy = (d_thing_points @ R(I_ksi[2]) + I_ksi * wlm)[:, :2]

    configurations[0] = I_ksi


def init_office_chair_feet(l):
    # not an important kinematics parameter
    leg_width = 0.015
    chassis_points = np.array([
        [-l / 2, -leg_width, 0],
        [-l / 2, leg_width, 0],
        [l / 2, leg_width, 0],
        [l / 2, -leg_width,  0]
    ], dtype=float)
    return chassis_points


def init_office_chair_wheels(r, d, alpha, beta):
    # not an important wheel parameter
    wheel_width = 0.03
    wheel_points = np.array([
        [-wheel_width / 2, -r, 0],
        [wheel_width / 2, -r, 0],
        [wheel_width / 2, r, 0],
        [-wheel_width / 2, r, 0]
    ], dtype=float)

    return ((wheel_points + np.array([0, -d, 0], dtype=float)) @ R(beta) + np.array([l, 0, 0], dtype=float)) @ R(alpha)


def init_office_chair_d_thing(d, alpha, beta):
    width = 0.015
    points = np.array([
        [-width / 2, -d, 0],
        [width / 2, -d, 0],
        [width / 2, d, 0],
        [-width / 2, d, 0]
    ], dtype=float)

    return ((points + np.array([0, -d/2, 0], dtype=float)) @ R(beta) + np.array([l, 0, 0], dtype=float)) @ R(alpha)


if __name__ == "__main__":

    fig, ax = init_plot(lim_from=-1, lim_to=1)

    num_frames = 360
    fps = 60
    dt = 1 / 60

    configurations = []
    shapes = []

    # I_ksi
    I_ksi = np.array([0.5, -0.5, np.pi/2], dtype=float)
    configurations.append(I_ksi)

    # wlm: "without last mask"
    wlm = np.array([1, 1, 0], dtype=float)

    # CHASSIS
    l = 0.3
    chassis_points = init_office_chair_feet(l)
    for i in range(5):
        chassis_shape = ax.add_patch(
            Polygon(((chassis_points + np.array([l/2, 0, 0], dtype=float)) @ R(I_ksi[2] + (i * 2*np.pi / 5)) + I_ksi * wlm)[:, :2], color="grey", alpha=0.75)
        )
        shapes.append(chassis_shape)

    # WHEELS
    r = 0.025
    d = 0.03

    alphas = []
    betas = []
    for i in range(5):
        alpha = -i * 2*np.pi / 5
        beta = np.random.uniform(0, 2*np.pi)

        alphas.append(alpha)
        betas.append(beta)

        wheel_points = init_office_chair_wheels(r, d, alpha, beta)
        wheel_shape = ax.add_patch(
            Polygon(
                (wheel_points @ R(I_ksi[2]) + I_ksi * wlm)[:, :2],
                color="#3A3845"
            )
        )
        shapes.append(wheel_shape)

        d_thing_points = init_office_chair_d_thing(d, alpha, beta)
        d_thing_shape = ax.add_patch(
            Polygon(
                (d_thing_points @ R(I_ksi[2]) + I_ksi * wlm)[:, :2],
                color="#3A3845"
            )
        )

        shapes.append(d_thing_shape)

    ani = FuncAnimation(
        fig,
        animate,
        fargs=(configurations, shapes, dt),
        frames=num_frames,
        interval=dt * 1000,
        repeat=False,
        blit=False,
        init_func=lambda: None
    )
    #ani.save('stolica.gif', writer='imagemagick')

    plt.show()
