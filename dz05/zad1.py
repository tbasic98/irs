import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)

if __name__ == "__main__":
    dt = 0.01

    t_begin = 0
    t_end = 15

    t_arr = np.linspace(t_begin, t_end, int((t_end - t_begin) / dt))

    # početno stanje newtonova sustava
    x_prev = np.array([0.5, 0.5, 1], dtype=float)
    A = np.zeros((3, 3), dtype=float)
    A[0:2, 1:3] = np.eye(2, dtype=float)

    x_arr = np.zeros((t_arr.shape[0], x_prev.shape[0]), dtype=float)
    x_arr[0] = x_prev

    mu_prev = np.array([-3, 2, 1], dtype=float)
    Sigma_prev = np.diag(np.array([10, 5, 2], dtype=float))
    omega_prev = np.linalg.inv(Sigma_prev)
    ksi_prev = omega_prev @ mu_prev
    #ksi_arr = np.zeros((t_arr.shape[0], 3), dtype=float)
    mu_arr = np.zeros((t_arr.shape[0], 3), dtype=float)
    #ksi_arr[0] = ksi_prev
    mu_arr[0] = mu_prev

    # procesni šum
    R = np.diag(np.array([15.25, 18.5, 15.1], dtype=float))

    #šum mjerenja
    Q = np.diag(np.array([10, 25], dtype=float))
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

            omega_prev_inv = np.linalg.inv(omega_prev)

            omega_pred = np.linalg.inv(Phi @ omega_prev_inv @ Phi.T + R)
            ksi_pred = omega_pred @ Phi @ omega_prev_inv @ ksi_prev
            omega_corr = C.T @ Q_inv @ C + omega_pred
            ksi_corr = C.T @ Q_inv @ z + ksi_pred

            ksi_prev = ksi_corr.copy()
            #ksi_arr[t_i] = ksi_corr
            omega_prev = omega_corr.copy()
            mu_arr[t_i] = np.linalg.inv(omega_corr) @ ksi_corr

    plt.plot(t_arr, x_arr[:, 0], c="#1e90ff", linewidth=1.5, label=r'$r(t)$')
    plt.plot(t_arr, x_arr[:, 1], c="#b0c4de", linewidth=1.5, label=r'$v(t)$')
    plt.plot(t_arr, x_arr[:, 2], c="#212121", linewidth=1.5, label=r'$a(t)$')
    plt.plot(t_arr, mu_arr[:, 0], c="pink", linewidth=1.3, zorder=2, label=r'$\mu_{r}(t)$')
    plt.plot(t_arr, mu_arr[:, 1], c="r", linestyle=":", linewidth=1.3, zorder=2, label=r'$\mu_{v}(t)$')
    plt.plot(t_arr, mu_arr[:, 2], c="b", linestyle=":", linewidth=1.3, zorder=2, label=r'$\mu_{a}(t)$')
    ax.legend()
    
    plt.show()