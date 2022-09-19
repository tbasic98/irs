import numpy as np


def sample_normal_distribution(b_square, M):
    b = np.sqrt(b_square)
    s = np.sum(np.random.uniform(-b, b, (M, 12)), axis=1)
    return s

alpha_1 = alpha_2 = alpha_3 = alpha_4 = alpha_5 = alpha_6 = 0.4

def sample_motion_model_odometry(u_current, x_prev):
    M = x_prev.shape[0]
    I_ksi_prev = u_current[0, :]
    I_ksi = u_current[1, :]

    d_rot1 = np.arctan2(I_ksi[1] - I_ksi_prev[1], I_ksi[0] - I_ksi_prev[0]) - I_ksi_prev[2]
    d_trans = np.sqrt((I_ksi_prev[0] - I_ksi[0])**2 + (I_ksi_prev[1] - I_ksi[1])**2)
    d_rot2 = I_ksi[2] - I_ksi_prev[2] - d_rot1
    d_rot1_hat = d_rot1 - sample_normal_distribution(alpha_1 * (d_rot1**2) + alpha_2 * (d_trans**2), M)
    d_trans_hat = d_trans - sample_normal_distribution(alpha_3 * (d_trans**2) + alpha_4 * (d_rot1**2) + alpha_4 * (d_rot2**2), M)
    d_rot2_hat = d_rot2 - sample_normal_distribution(alpha_1 * (d_rot2**2) + alpha_2 * (d_trans**2), M)

    x = x_prev[:, 0]
    y = x_prev[:, 1]
    theta = x_prev[:, 2]

    x_current = x + d_trans_hat * np.cos(theta + d_rot1_hat)

    y_current = y + d_trans_hat * np.sin(theta + d_rot1_hat)

    theta_current = theta + d_rot1_hat + d_rot2_hat

    return np.vstack((x_current, y_current, theta_current)).T

def sample_motion_model_velocity(u_current, x_prev, dt):
    M = x_prev.shape[0]
    v = u_current[0]
    omega = u_current[2]

    v_hat = v + sample_normal_distribution(
        alpha_1 * v ** 2 + alpha_2 * omega ** 2, M
    )

    omega_hat = omega + sample_normal_distribution(
        alpha_3 * v ** 2 + alpha_4 * omega ** 2, M
    )

    gamma_hat = sample_normal_distribution(
        alpha_5 * v ** 2 + alpha_6 * omega ** 2, M
    )

    x = x_prev[:, 0]
    y = x_prev[:, 1]
    theta = x_prev[:, 2]

    x_current = x - v_hat / omega_hat * np.sin(theta) \
        + v_hat / omega_hat * np.sin(theta + omega_hat * dt)

    y_current = y + v_hat / omega_hat * np.cos(theta) \
        - v_hat / omega_hat * np.cos(theta + omega_hat * dt)

    theta_current = theta + omega_hat * dt + gamma_hat * dt

    return np.vstack((x_current, y_current, theta_current)).T