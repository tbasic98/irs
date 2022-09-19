import numpy as np
import matplotlib.pyplot as plt

class Sensor:
    def __init__(self, z_hit=0.75, z_short=0.25, sigma_hit=0.25, lambda_short=2.3):
        self.z_max = 5.0

        self.sigma_hit = sigma_hit
        self.lambda_short = lambda_short
        self.z_hit = z_hit
        self.z_short = z_short

    def measure(self, distances):
        n = distances.shape[0]

        case_1 = (
                distances + np.random.normal(0, self.sigma_hit, n)
        ).clip(min=0, max=self.z_max)

        case_2 = np.random.exponential(
            1 / self.lambda_short, n
        ).clip(max=distances)

        case_indices = np.random.choice(
            np.arange(2),
            (1, n),
            p=[self.z_hit, self.z_short]
        ).T.repeat(2, axis=1)

        case_range = np.arange(0, 2).reshape(1, 2).repeat(
            distances.shape[0], axis=0
        )

        # case probabilites
        cp = (case_indices == case_range) + 0
        return cp[:, 0] * case_1 + cp[:, 1] * case_2

def pmf(distances, X, sensor):
    sigma_hit = sensor.sigma_hit
    z_max = sensor.z_max
    lambda_short = sensor.lambda_short
    z_hit = sensor.z_hit
    z_short = sensor.z_short

    # CASE 1
    n = 1 / np.sqrt(2 * np.pi * sigma_hit ** 2) * np.exp(
        -0.5 * (distances - X) ** 2 / sigma_hit ** 2
    )
    # eta
    dz = 0.1
    z_arr = np.linspace(0, z_max, int(z_max / dz))

    z_tensor = np.tile(
        np.expand_dims(z_arr, axis=(1, 2)), (1, X.shape[0])
    )

    eta_int = 1 / np.sqrt(2 * np.pi * sigma_hit ** 2) * np.exp(
        -0.5 * (z_tensor - distances) ** 2 / sigma_hit ** 2
    ) * dz
    eta = np.sum(eta_int, axis=0) ** -1

    p_hit = n * eta
    p_hit[X.reshape(1, X.shape[0]) < 0] = 0
    p_hit[X.reshape(1, X.shape[0]) > z_max] = 0

    # CASE 2
    eta = 1 / (1 - np.exp(-lambda_short * distances))
    p_short = eta * lambda_short * np.exp(-lambda_short * X)
    p_short[X < 0] = 0
    p_short[X > distances] = 0

    return z_hit * p_hit + z_short * p_short

def learn_instrinsic_parameters(Z, sigma_hit=0.1, lam_short=1):
    Z_star = np.ones(Z.shape[0])*3
    z_max = 5
    theta = np.array([-1, -1, sigma_hit, lam_short], dtype=float)

    while True:
        # CASE 1
        n = 1 / np.sqrt(2 * np.pi * sigma_hit ** 2) * np.exp(
            -0.5 * (Z_star - Z) ** 2 / sigma_hit ** 2
        )
        # eta
        dz = 0.1
        z_arr = np.linspace(0, z_max, int(z_max / dz))

        z_tensor = np.tile(
            np.expand_dims(z_arr, axis=(1, 2)), (1, Z.shape[0])
        )

        eta_int = 1 / np.sqrt(2 * np.pi * sigma_hit ** 2) * np.exp(
            -0.5 * (z_tensor - Z_star) ** 2 / sigma_hit ** 2
        ) * dz
        eta = np.sum(eta_int, axis=0) ** -1

        p_hit = n * eta
        p_hit[Z.reshape(1, Z.shape[0]) < 0] = 0
        p_hit[Z.reshape(1, Z.shape[0]) > z_max] = 0

        # CASE 2
        eta = 1 / (1 - np.exp(-lam_short * Z_star))
        p_short = eta * lam_short * np.exp(-lam_short * Z)
        p_short[Z < 0] = 0
        p_short[Z > Z_star] = 0


        eta = (p_hit + p_short)** -1

        e_hit = eta * p_hit
        e_short = eta * p_short

        z_hit = np.sum(e_hit)/Z.shape[0]
        z_short = np.sum(e_short)/Z.shape[0]
        sigma_hit = np.sqrt((e_hit @ ((Z-Z_star)**2).reshape(Z.shape[0], 1))/np.sum(e_hit))
        lam_short = np.sum(e_short)/(np.sum(e_short*Z))

        theta_new = np.array([z_hit, z_short, sigma_hit[0, 0], lam_short], dtype=float)
        if np.linalg.norm(theta - theta_new) < 1e-10:
            theta = theta_new
            break
        theta = theta_new
    return theta


if __name__ == "__main__":
    sensor = Sensor()
    distances = np.ones(10000)*3

    za_histogram = sensor.measure(distances)

    X = np.linspace(-1, 5, 400)
    theta = learn_instrinsic_parameters(za_histogram)
    sensor_learned = Sensor(theta[0], theta[1], theta[2], theta[3])

    plt.plot(X, pmf(np.ones(400)*3, X, sensor).T, color="y")
    plt.plot(X, pmf(np.ones(400)*3, X, sensor_learned).T, color="b")
    plt.hist(za_histogram, 70, color="r", density=True)
    plt.xlim(-1.2, 5.2)
    plt.show()
