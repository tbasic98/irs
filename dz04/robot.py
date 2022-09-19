import numpy as np
from shared import R, wlm
#from polygon import Polygon
import polygon

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


# not an important kinematics parameter
chassis_length_front = 0.25
chassis_length_back = 0.05
# not an important wheel parameter
wheel_width = 0.03


def chassis_points(l):

    return np.array([
        [-chassis_length_back, -l, 0],
        [chassis_length_front, -l, 0],
        [chassis_length_front, l, 0],
        [-chassis_length_back, l, 0]
    ], dtype=float)


def wheels_points(r):
    return np.array([
        [-wheel_width / 2, -r, 0],
        [wheel_width / 2, -r, 0],
        [wheel_width / 2, r, 0],
        [-wheel_width / 2, r, 0]
    ], dtype=float)


class Sensor:
    def __init__(self, R_s=np.array([0, 0, 0], dtype=float)):
        self.d_max = 0.65
        self.d_min = 0.05
        self.d = self.d_max
        self.alpha = np.deg2rad(20)
        self.R_s = R_s


    @property
    def R_cone_full(self):
        S_cone_full = np.array([
            [0, 0, 0],
            [self.d_max * np.cos(self.alpha / 2), self.d_max * np.sin(self.alpha / 2), 0],
            [self.d_max, 0, 0],
            [self.d_max * np.cos(self.alpha / 2), -self.d_max * np.sin(self.alpha / 2), 0]
        ], dtype=float)
        return S_cone_full @ R(self.R_s[2]) + self.R_s * wlm


    @property
    def R_cone(self):
        S_cone = np.array([
            [self.d_min * np.cos(self.alpha / 2), self.d_min * np.sin(self.alpha / 2), 0],
            [self.d * np.cos(self.alpha / 2), self.d * np.sin(self.alpha / 2), 0],
            [self.d, 0, 0],
            [self.d * np.cos(self.alpha / 2), -self.d * np.sin(self.alpha / 2), 0],
            [self.d_min * np.cos(self.alpha / 2), -self.d_min * np.sin(self.alpha / 2), 0],
        ], dtype=float)
        return S_cone @ R(self.R_s[2]) + self.R_s * wlm


class Robot:
    __slots__ = ["I_ksi", "l", "r", "alpha_left", "beta_left", "alpha_right", "beta_right", "wheel_points", "chassis_points", "sensors_list", "max_phi_dot"]

    def __init__(self, I_ksi=np.array([0, 0, 0], dtype=float)):
        # CHASSIS & WHEELS
        self.l = 0.1
        self.r = 0.04
        self.alpha_left = np.pi / 2
        self.beta_left = np.pi
        self.alpha_right = -np.pi / 2
        self.beta_right = np.pi
        self.I_ksi = I_ksi

        self.max_phi_dot = 100 * 2 * np.pi / 60

        sensor_configurations = np.array([
            [chassis_length_front, self.l, np.pi / 2],
            [chassis_length_front, self.l * 2 / 3, np.pi / 2 * 2 / 3],
            [chassis_length_front, self.l * 1 / 3, np.pi / 2 * 1 / 3],
            [chassis_length_front, self.l * 0 / 3, np.pi / 2 * 0 / 3],
            [chassis_length_front, -self.l * 1 / 3, -np.pi / 2 * 1 / 3],
            [chassis_length_front, -self.l * 2 / 3, -np.pi / 2 * 2 / 3],
            [chassis_length_front, -self.l, -np.pi / 2],
        ], dtype=float)

        self.sensors_list = [Sensor(sc) for sc in sensor_configurations]


        self.wheel_points = wheels_points(self.r)
        self.chassis_points = chassis_points(self.l)

    @property
    def I_chassis_points(self):
        return self.chassis_points @ R(self.I_ksi[2]) + self.I_ksi * wlm

    @property
    def I_wheel_points_left(self):
        R_wheel_points_left = (self.wheel_points @ R(self.beta_left) + np.array([self.l, 0, 0], dtype=float)) @ R(self.alpha_left)
        return R_wheel_points_left @ R(self.I_ksi[2]) + self.I_ksi * wlm

    @property
    def I_wheel_points_right(self):
        R_wheel_points_right = (self.wheel_points @ R(self.beta_right) + np.array([self.l, 0, 0], dtype=float)) @ R(self.alpha_right)
        return R_wheel_points_right @ R(self.I_ksi[2]) + self.I_ksi * wlm

    @property
    def I_sensors(self):
        return np.array([sensor.R_cone for sensor in self.sensors_list], dtype=float)  @ R(self.I_ksi[2]) + self.I_ksi * wlm



    def R_inverse_kinematics(self, R_ksi_dot):
        return 1 / self.r * M(self.l) @ R_ksi_dot

    def inverse_kinematics(self, I_ksi_dot):
        return 1 / self.r * M(self.l) @ R(self.I_ksi[2]) @ I_ksi_dot

    def forward_kinematics(self, phis_dot):
        return self.r * R(self.I_ksi[2]).T @ M_inv(self.l) @ phis_dot

    def update_state(self, I_ksi_dot, dt):
        self.I_ksi += I_ksi_dot * dt

    def measure(self, qt):
        for i, sensor in enumerate(self.sensors_list):
            I_cone_full = sensor.R_cone_full @ R(self.I_ksi[2]) + self.I_ksi * wlm

            p = polygon.Polygon(I_cone_full[:, :2])
            I_s_x = I_cone_full[0, 0]
            I_s_y = I_cone_full[0, 1]

            min_dist = sensor.d_max
            for obstacle in qt.find_overlaps(I_cone_full):
                contact_points = p.intersection_points(obstacle.polygon)
                for I_o_x, I_o_y in contact_points:
                    dist = np.sqrt((I_s_x - I_o_x) ** 2 + (I_o_y - I_s_y) ** 2)
                    if dist < min_dist:
                        min_dist = dist
            sensor.d = min_dist

    def collision_detected(self, qt):
        all_points = np.concatenate([
            self.I_chassis_points,
            self.I_wheel_points_left,
            self.I_wheel_points_right
        ])

        p = polygon.Polygon(all_points[:, :2])
        for obstacle in qt.find_overlaps(all_points):
            collision = p.collidepoly(obstacle.polygon)
            if isinstance(collision, bool):
                if not collision:
                    continue
            return True
        return False

    def strategy_3(self):
        I_S = []
        I_m = []
        I_m_inv = []
        d = []
        for sensor in self.sensors_list:
            R_m_k = np.array([sensor.d, 0, 0], dtype=float) @ R(sensor.R_s[2]) + sensor.R_s * wlm
            I_m_k = R_m_k @ R(self.I_ksi[2]) + self.I_ksi * wlm
            I_S_k = sensor.R_s @ R(self.I_ksi[2]) + self.I_ksi * wlm

            S_m_k_inv = np.array([-(sensor.d_max - sensor.d), 0, 0], dtype=float)
            R_m_k_inv = S_m_k_inv @ R(sensor.R_s[2]) + sensor.R_s * wlm
            I_m_k_inv = R_m_k_inv @ R(self.I_ksi[2]) + self.I_ksi * wlm

            I_S.append(I_S_k)
            I_m.append(I_m_k)
            I_m_inv.append(I_m_k_inv)
            d.append(sensor.d)
        return I_S, I_m, I_m_inv, d