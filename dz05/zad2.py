import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

fig, ax = plt.subplots(1)

alpha = 2 / 3
beta = 1 / 3
gamma = 0.7
delta = 0.1


def g(x_prev, dt):
    return np.array([
        x_prev[0] + alpha * dt * x_prev[0] - beta * dt * x_prev[0] * x_prev[1],
        x_prev[1] - gamma * dt * x_prev[1] + delta * dt * x_prev[0] * x_prev[1]
        ], dtype=float)


if __name__ == "__main__":
    dt = 1 / 24
    t_end = 60
    t_arr = np.linspace(0, t_end, int(t_end / dt) + 1)

    x_0 = 10
    y_0 = 5

    ksi_curr = np.array([x_0, y_0], dtype=float)
    ksi_prev = ksi_curr.copy()
    ksi_arr = np.zeros((t_arr.shape[0], 2), dtype=float)
    ksi_arr[0] = ksi_curr

    mu_prev = np.array([5.0, 1.0], dtype=float)
    mu_curr = mu_prev.copy()
    Sigma_prev = np.diag(np.array([10, 10], dtype=float))
    mu_arr = np.zeros((t_arr.shape[0], 2), dtype=float)
    mu_arr[0] = mu_curr

    # procesni šum
    R = np.diag(np.array([0.001, 0.001], dtype=float))

    #šum mjerenja
    Q = np.diag(np.array([0.5], dtype=float))

    C = np.array([[0, 1]], dtype=float)

    alpha_UK = 2.5
    beta_UK = 2
    k_UK = 10
    n = Sigma_prev.shape[0]
    lam_UK = alpha_UK**2*(n + k_UK) - n
    gamma_UK = np.sqrt(n + lam_UK)

    Wm0 = lam_UK / (n + lam_UK)
    Wc0 = Wm0 + (1 - alpha_UK**2 + beta_UK)
    Wi = 1 / (2*(n + lam_UK))
    Wm = np.ones(2*n+1) * Wi
    Wm[0] = Wm0
    Wc = Wm.copy()
    Wc[0] = Wc0

    for t_i, t in enumerate(t_arr):
        if t > 0:
            ksi_curr = g(ksi_prev, dt)
            ksi_prev = ksi_curr.copy()
            ksi_arr[t_i] = ksi_curr

            # measurement
            ksi_with_noise = ksi_curr + np.random.multivariate_normal(np.zeros(R.shape[0], dtype=float), R)
            measurement_noise = np.random.multivariate_normal(np.zeros(Q.shape[0], dtype=float), Q)
            z = C @ ksi_with_noise + measurement_noise

            Sigma_prev_sqrt = sqrtm(Sigma_prev)
            X_prev = np.vstack((mu_prev, mu_prev.reshape(1, n) + gamma_UK * Sigma_prev_sqrt, mu_prev.reshape(1, n) - gamma_UK * Sigma_prev_sqrt))
            X_star = np.apply_along_axis(g, 1, X_prev, dt)

            mu_pred = Wm.reshape(1, 2*n + 1) @ X_star

            Sigma_pred = R.copy()
            for i, x_star in enumerate(X_star):
                Sigma_pred += (Wc[i] * (x_star - mu_pred).reshape(n, 1) @ (x_star - mu_pred).reshape(1, n))

            Sigma_pred_sqrt = sqrtm(Sigma_pred)

            X_pred = np.vstack((mu_pred, mu_pred.reshape(1, n) + gamma_UK * Sigma_pred_sqrt, mu_pred.reshape(1, n) - gamma_UK * Sigma_pred_sqrt))
            Z_pred = C @ X_pred.T
            z_kapa = Z_pred @ Wm

            S = Q.copy()
            Sigma_x_z = np.zeros((n, z_kapa.shape[0]), dtype=float)
            for i, Z_pred_i in enumerate(Z_pred.T):
                S += (Wc[i] * (Z_pred_i - z_kapa) @ (Z_pred_i - z_kapa).T)
                Sigma_x_z += (Wc[i] * (X_pred[i] - mu_pred).reshape(n, 1) @ (Z_pred_i - z_kapa).T).reshape(n, z_kapa.shape[0])

            K = Sigma_x_z @ np.linalg.inv(S)

            mu_corr = mu_pred + K @ (z - z_kapa)
            Sigma_corr = Sigma_pred - K @ S @ K.T
            mu_arr[t_i] = mu_corr
            mu_prev = mu_corr.copy()
            Sigma_prev = Sigma_corr.copy()


    plt.plot(
        t_arr, ksi_arr[:, 0], c="#00796b",
        linestyle="-", linewidth=1.5, zorder=1, label=r'$x(t)$'
    )
    plt.plot(
        t_arr, ksi_arr[:, 1], c="#e53935",
        linestyle="-", linewidth=1.5, zorder=1, label=r'$y(t)$'
    )
    plt.plot(
        t_arr, mu_arr[:, 0], c="#78909c",
        linewidth=1.3, zorder=2, label=r'$\mu_{x}(t)$'
    )
    plt.plot(
        t_arr, mu_arr[:, 1], c="#212121",
        linewidth=1.3, zorder=2, label=r'$\mu_{y}(t)$'
    )
    ax.legend()

    plt.show()