import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shared import init_plot, R
from matplotlib.patches import Polygon, Circle
from robot import Robot
from algorithms import StopAlgorithm, GoToGoalAlgorithm, AvoidObstacleAlgorithm
from quadtree import QuadTree
import polygon

obstacle_1_points = np.array([
    [-0.25, -0.25, 0],
    [0.25, -0.25, 0],
    [0.25, 0.25, 0],
    [-0.25, 0.25, 0],
], dtype=float)

I_R_o1 = R(np.pi / 2.2).T
I_t_o1 = np.array([0.75, 0.5, 0], dtype=float)

obstacle_2_points = np.array([
    [-0.25, -0.25, 0],
    [0.25, -0.25, 0],
    [0.25, 0.25, 0],
    [-0.25, 0.25, 0],
], dtype=float)

I_R_o2 = R(np.pi / 6).T
I_t_o2 = np.array([-0.1, 0.75, 0], dtype=float)

tao_1 = 0.5
tao_2 = 0.65

def init_strategy_3_lines(robot, shapes):
    I_S, I_m, I_m_inv, d = robot.strategy_3()
    zeleni = np.array([0, 0, 0], dtype=float)
    for k, I_S_k in enumerate(I_S):
        crveni = I_m[k] - I_S_k
        zeleni += (I_m_inv[k] - I_S_k)
        line = ax.plot([I_S_k[0], I_S_k[0] + crveni[0]], [I_S_k[1], I_S_k[1] + crveni[1]], color="r", linewidth=2, solid_capstyle="round")[0]
        shapes.append(line)
    shapes.append(ax.plot([I_S[3][0], I_S[3][0] + zeleni[0]], [I_S[3][1], I_S[3][1] + zeleni[1]], color="g", linewidth=2, solid_capstyle="round")[0])

def update_strategy_3_lines(robot, shapes):
    I_S, I_m, I_m_inv, d = robot.strategy_3()
    zeleni = np.array([0, 0, 0], dtype=float)
    for k, I_S_k in enumerate(I_S):
        crveni = I_m[k] - I_S_k
        zeleni += (I_m_inv[k] - I_S_k)
        shapes[10 + k].set_data([I_S_k[0], I_S_k[0] + crveni[0]], [I_S_k[1], I_S_k[1] + crveni[1]])
    shapes[17].set_data([I_S[3][0], I_S[3][0] + zeleni[0]], [I_S[3][1], I_S[3][1] + zeleni[1]])


class Obstacle:
    __slots__ = ["points", "R", "t", "polygon"]

    def __init__(self, points, R, t):
        self.points = points
        self.R = R
        self.t = t
        self.polygon = polygon.Polygon(self[:, :2])

    def __getitem__(self, item):
        return (self.points @ self.R.T + self.t)[item]

    def get_bounding_rectangle(self):
        x_min = np.min(self[:, 0])
        y_min = np.min(self[:, 1])
        x_max = np.max(self[:, 0])
        y_max = np.max(self[:, 1])
        return x_min, y_min, x_max - x_min, y_max - y_min


def animate(i, robot, shapes, dt):

    global algorithm

    if robot.collision_detected(qt):
        print("Collision 😢")
        ani.event_source.stop()
        algorithm = stop
    else:
        if np.linalg.norm(robot.I_ksi[:2] - I_g[:2]) < d1:
            print("At goal")
            ani.event_source.stop()
            algorithm = stop
        else:
            min_dist = robot.sensors_list[0].d_max
            for sensor in robot.sensors_list:
                if sensor.d < sensor.d_max and sensor.d < min_dist:
                    min_dist = sensor.d
            if algorithm == avoid_obstacle and min_dist > tao_1 * robot.sensors_list[0].d_max:
                algorithm = go_to_goal
            elif algorithm == go_to_goal and min_dist < tao_2 * robot.sensors_list[0].d_max:
                algorithm = avoid_obstacle

    v, omega = algorithm()

    robot.measure(qt)

    phis_dot = robot.R_inverse_kinematics(np.array([v, 0, omega], dtype=float))
    I_ksi_dot = robot.forward_kinematics(phis_dot)
    robot.update_state(I_ksi_dot, dt)

    shapes[0].xy = robot.I_chassis_points[:, :2]
    shapes[1].xy = robot.I_wheel_points_left[:, :2]
    shapes[2].xy = robot.I_wheel_points_right[:, :2]

    for k, I_sensor in enumerate(robot.I_sensors):
        shapes[k + 3].xy = I_sensor[:, :2]

    update_strategy_3_lines(robot, shapes)

if __name__ == "__main__":

    fig, ax = init_plot(lim_from=-2, lim_to=2)

    #num_frames = 360
    fps = 30
    dt = 1 / fps

    configurations = []
    shapes = []

    obstacles = [
        Obstacle(obstacle_1_points, I_R_o1, I_t_o1),
        Obstacle(obstacle_2_points, I_R_o2, I_t_o2),
    ]

    for obstacle in obstacles:
        ax.add_patch(Polygon(obstacle[:, :2], color="#37474F"))

    I_g = np.array([1.5, 0.5, 0], dtype=float)
    ax.add_patch(Circle(I_g[:2], radius=0.05, color="#e91e6388"))

    qt = QuadTree(obstacles)



    robot: Robot = Robot(np.array([-1, 0, 0], dtype=float))

    robot.measure(qt)

    shapes.append(ax.add_patch(Polygon(robot.I_chassis_points[:, :2], color="#39AEA988")))
    shapes.append(ax.add_patch(Polygon(robot.I_wheel_points_left[:, :2], color="#3A3845")))
    shapes.append(ax.add_patch(Polygon(robot.I_wheel_points_right[:, :2], color="#3A3845")))

    for I_sensor in robot.I_sensors:
        shapes.append(
            ax.add_patch(
                Polygon(I_sensor[:, :2], color="#1e88e555")
            )
        )

    init_strategy_3_lines(robot, shapes)

    go_to_goal = GoToGoalAlgorithm(robot, I_g, dt)
    stop = StopAlgorithm()
    avoid_obstacle = AvoidObstacleAlgorithm(robot, dt)

    algorithm = go_to_goal

    d1 = 0.1


    ani = FuncAnimation(
        fig,
        animate,
        fargs=(robot, shapes, dt),
        #frames=num_frames,
        interval=dt * 1000,
        repeat=False,
        blit=False,
        init_func=lambda: None
    )
    #ani.save('zad.gif', writer='imagemagick')

    plt.show()
