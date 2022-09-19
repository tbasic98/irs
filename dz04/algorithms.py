import numpy as np
from shared import R

class PID:
    __slots__ = ["k_p", "k_i", "k_d", "e_prev", "e_acc"]

    def __init__(self, k_p=1.0, k_i=0.5, k_d=0.1):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.e_prev = self.e_acc = 0

    def __call__(self, e, dt):
        e_diff = (e - self.e_prev) / dt
        self.e_acc += e * dt
        self.e_prev = e
        return self.k_p * e + self.k_i * self.e_acc + self.k_d * e_diff


class StopAlgorithm:
    def __call__(self):
        return 0, 0

class AvoidObstacleAlgorithm:
    __slots__ = ["pid", "robot", "dt"]

    def __init__(self, robot, dt):
        self.pid = PID()
        self.robot = robot
        self.dt = dt

    def __call__(self):
        I_S, _, I_m_inv, _ = self.robot.strategy_3()
        I_ksi_star = np.array([0, 0, 0], dtype=float)
        for k, I_S_k in enumerate(I_S):
            I_ksi_star += (I_m_inv[k] - I_S_k)

        e = np.arctan2(I_ksi_star[1], I_ksi_star[0]) - self.robot.I_ksi[2]
        e = np.arctan2(np.sin(e), np.cos(e))

        R_omega = self.pid(e, self.dt)
        R_v_dot = self.robot.max_phi_dot * self.robot.r

        if R_v_dot == 0:
            R_omega_max = self.robot.max_phi_dot * self.robot.r / 2 / self.robot.l
            R_v_dot_feas = 0
            R_omega_feas = np.copysign(1, R_omega) * max(0, min(abs(R_omega), R_omega_max))
        else:
            phis_dot = self.robot.R_inverse_kinematics(np.array([R_v_dot, 0, abs(R_omega)], dtype=float))
            phi_dot_lrmax = max(phis_dot[0], phis_dot[1])
            phi_dot_lrmin = min(phis_dot[0], phis_dot[1])

            if phi_dot_lrmax > self.robot.max_phi_dot:
                phi_dot_ld = phis_dot[0] - (phi_dot_lrmax - self.robot.max_phi_dot)
                phi_dot_rd = phis_dot[1] - (phi_dot_lrmax - self.robot.max_phi_dot)
            elif phi_dot_lrmin < 0:
                phi_dot_ld = phis_dot[0] + (0 - phi_dot_lrmin)
                phi_dot_rd = phis_dot[1] + (0 - phi_dot_lrmin)
            else:
                phi_dot_ld = phis_dot[0]
                phi_dot_rd = phis_dot[1]

            phi_dot_ld = max(0, min(phi_dot_ld, self.robot.max_phi_dot))
            phi_dot_rd = max(0, min(phi_dot_rd, self.robot.max_phi_dot))

            R_ksi_dot = R(self.robot.I_ksi[2]) @ self.robot.forward_kinematics(
                np.array([phi_dot_ld, phi_dot_rd, 0], dtype=float)
            )

            R_v_dot_feas = R_ksi_dot[0]
            R_omega_feas = np.copysign(1, R_omega) * R_ksi_dot[2]


        return R_v_dot_feas, R_omega_feas


class GoToGoalAlgorithm:
    __slots__ = ["pid", "robot", "I_g", "dt"]

    def __init__(self, robot, I_g, dt):
        self.pid = PID()
        self.robot = robot
        self.I_g = I_g
        self.dt = dt

    def __call__(self):
        e = np.arctan2(
            self.I_g[1] - self.robot.I_ksi[1],
            self.I_g[0] - self.robot.I_ksi[0]
        ) - self.robot.I_ksi[2]
        e = np.arctan2(np.sin(e), np.cos(e))

        R_omega = self.pid(e, self.dt)
        R_v_dot = self.robot.max_phi_dot * self.robot.r

        if R_v_dot == 0:
            R_omega_max = self.robot.max_phi_dot * self.robot.r / 2 / self.robot.l
            R_v_dot_feas = 0
            R_omega_feas = np.copysign(1, R_omega) * max(0, min(abs(R_omega), R_omega_max))
        else:
            phis_dot = self.robot.R_inverse_kinematics(np.array([R_v_dot, 0, abs(R_omega)], dtype=float))
            phi_dot_lrmax = max(phis_dot[0], phis_dot[1])
            phi_dot_lrmin = min(phis_dot[0], phis_dot[1])

            if phi_dot_lrmax > self.robot.max_phi_dot:
                phi_dot_ld = phis_dot[0] - (phi_dot_lrmax - self.robot.max_phi_dot)
                phi_dot_rd = phis_dot[1] - (phi_dot_lrmax - self.robot.max_phi_dot)
            elif phi_dot_lrmin < 0:
                phi_dot_ld = phis_dot[0] + (0 - phi_dot_lrmin)
                phi_dot_rd = phis_dot[1] + (0 - phi_dot_lrmin)
            else:
                phi_dot_ld = phis_dot[0]
                phi_dot_rd = phis_dot[1]

            phi_dot_ld = max(0, min(phi_dot_ld, self.robot.max_phi_dot))
            phi_dot_rd = max(0, min(phi_dot_rd, self.robot.max_phi_dot))

            # imao sam bug jer sam umjesto forward_kinematics stavio inverse_kinematics
            R_ksi_dot = R(self.robot.I_ksi[2]) @ self.robot.forward_kinematics(
                np.array([phi_dot_ld, phi_dot_rd, 0], dtype=float)
            )

            R_v_dot_feas = R_ksi_dot[0]
            R_omega_feas = np.copysign(1, R_omega) * R_ksi_dot[2]


        return R_v_dot_feas, R_omega_feas