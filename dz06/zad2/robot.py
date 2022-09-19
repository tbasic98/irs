import numpy as np
from shared import R


def M(l):
    return np.array([
        [1, 0, -l],
        [1, 0, l],
        [0, 1, 0]
    ], dtype=float)


def M_inv(l):
    return np.array([
        [1 / 2, 1 / 2, 0],
        [0, 0, 1],
        [-1 / 2 / l, 1 / 2 / l, 0]
    ], dtype=float)


class Robot:
    __slots__ = ["I_ksi", "l", "r", "cpr"]

    def __init__(self, I_ksi):
        self.I_ksi = I_ksi
        self.l = 0.1
        self.r = 0.05
        self.cpr = 300

    def update_state(self, I_ksi_dot, dt):
        self.I_ksi += I_ksi_dot * dt
        self.I_ksi[2] = np.arctan2(
            np.sin(self.I_ksi[2]), np.cos(self.I_ksi[2])
        )

    def update_state_R(self, R_ksi_dot, dt):
        I_ksi_dot = R(self.I_ksi[2]).T @ R_ksi_dot
        self.update_state(I_ksi_dot, dt)

    def R_inverse_kinematics(self, R_ksi_dot):
        return 1 / self.r * M(self.l) @ R_ksi_dot

    def forward_kinematics(self, phis_dot):
        return self.r * R(self.I_ksi[2]).T @ M_inv(self.l) @ phis_dot

    def read_encoders(self, phis_dot, dt):
        delta_c_left = int(self.cpr * phis_dot[0] * dt / 2 / np.pi)
        delta_c_right = int(self.cpr * phis_dot[1] * dt / 2 / np.pi)
        return np.array([delta_c_left, delta_c_right], dtype=float)

    def enc_deltas_to_phis_dot(self, deltas_c, dt):
        phi_dot_l_enc = deltas_c[0] * 2 * np.pi / self.cpr / dt
        phi_dot_r_enc = deltas_c[1] * 2 * np.pi / self.cpr / dt
        return np.array([phi_dot_l_enc, phi_dot_r_enc, 0], dtype=float)