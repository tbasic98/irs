import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)

def sampler(weights):
    indices = np.zeros(weights.shape, dtype=int)
    M = weights.shape[0]
    index = int(np.random.random() * M)
    beta = 0
    weight_max = np.max(weights)
    for i in range(M):
        beta += np.random.random() * 2 * weight_max
        while weights[index] < beta:
            beta -= weights[index]
            index = (index + 1) % M
        indices[i] = index
    return indices


if __name__ == "__main__":
    dt = 0.1

    t_begin = 0
    t_end = 10

    t_arr = np.linspace(t_begin, t_end, int((t_end - t_begin) / dt))

    # početno stanje newtonova sustava
    x_prev = np.array([0.5, 0.2, 1], dtype=float)
    A = np.zeros((3, 3), dtype=float)
    A[0:2, 1:3] = np.eye(2, dtype=float)

    x_arr = np.zeros((t_arr.shape[0], x_prev.shape[0]), dtype=float)
    x_arr[0] = x_prev

    M = 100
    particles = np.random.uniform(-1, 7, (M, 3))
    plt.scatter(np.ones(M)*t_begin, particles[:, 1], color='#90ee90')
    plt.scatter(np.ones(M)*t_begin, particles[:, 2], color='#b19cd9')

    mean = np.mean(particles, axis=0)
    mean_arr = np.zeros((t_arr.shape[0], 3), dtype=float)
    mean_arr[0] = mean

    # procesni šum
    R = np.diag(np.array([0.01, 0.01, 0.01], dtype=float))

    #šum mjerenja
    Q = np.diag(np.array([0.003, 0.0001], dtype=float))
    Q_inv = np.linalg.inv(Q)

    C = np.array([[0, 1, 0],
                  [0, 0, 1]], dtype=float)

    Phi = np.eye(3, dtype=float) + dt * A + 0.5 * dt**2 * A @ A

    for t_i, t in enumerate(t_arr):
        if t > 0:
            x_current = Phi @ x_prev
            x_arr[t_i] = x_current
            x_prev = x_current

            x_with_noise = x_current + np.random.multivariate_normal(np.zeros(R.shape[0], dtype=float), R)
            measurement_noise = np.random.multivariate_normal(np.zeros(Q.shape[0], dtype=float), Q)
            z = C @ x_with_noise + measurement_noise

            particles = (Phi @ particles.T).T
            particles += np.random.normal(0, 0.05, (M, 3))
            weights = np.linalg.norm(particles[:, 1:] - z.reshape(1, 2), axis=1)

            x_0 = y_1 = np.min(weights)
            x_1 = y_0 = np.max(weights)
            sim = (y_1 - y_0) / (x_1 - x_0) * (weights - x_0) + y_0
            weights = sim / np.sum(sim)

            indices = sampler(weights)
            particles = particles[indices]

            mean = np.mean(particles, axis=0)
            mean_arr[t_i] = mean

            plt.scatter(np.ones(M)*t, particles[:, 1], color='#90ee90')
            plt.scatter(np.ones(M)*t, particles[:, 2], color='#b19cd9')

    plt.plot(t_arr, x_arr[:, 0], c="#1e90ff", linewidth=1.5, label=r'$r(t)$')
    plt.plot(t_arr, x_arr[:, 1], c="r", linewidth=1.5, label=r'$v(t)$')
    plt.plot(t_arr, x_arr[:, 2], c="#212121", linewidth=1.5, label=r'$a(t)$')
    plt.plot(t_arr, mean_arr[:, 1], c="g", linestyle="-.", linewidth=1.5, zorder=2, label=r'$\bar{X}[:, 1](t)$')
    plt.plot(t_arr, mean_arr[:, 2], c="purple", linestyle="-.", linewidth=1.5, zorder=2, label=r'$\bar{X}[:, 2](t)$')
    ax.legend()
    plt.ylim([-5, 15])

    plt.show()