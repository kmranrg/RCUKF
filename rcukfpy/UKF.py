"""
UKF.py

Implements an Unscented Kalman Filter (UKF) for nonlinear state estimation.
Uses the unscented transform to propagate sigma points through a process model,
then corrects predictions using noisy measurements.

Dependencies:
    - numpy: For matrix operations and numerical computations.
"""

import numpy as np


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter (UKF) implementation for nonlinear systems.

    Uses a probabilistic approach with sigma points to avoid linearizing the process model.
    """

    def __init__(self, n_dim, process_noise, measurement_noise, alpha=1e-3, beta=2, kappa=0):
        """
        Initialize the filter.

        Args:
            n_dim (int): Dimensionality of the system state.
            process_noise (ndarray): Process noise covariance matrix Q (n_dim x n_dim).
            measurement_noise (ndarray): Measurement noise covariance matrix R.
            alpha (float): Sigma point spread scaling parameter.
            beta (float): Parameter incorporating prior knowledge (2 optimal for Gaussian).
            kappa (float): Secondary scaling parameter.
        """
        self.n_dim = n_dim
        self.Q = process_noise
        self.R = measurement_noise

        # UKF spread/scaling parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (n_dim + kappa) - n_dim

        # Sigma point weights
        denom = n_dim + self.lambda_
        self.W_m = np.full(2 * n_dim + 1, 1 / (2 * denom))
        self.W_c = np.full(2 * n_dim + 1, 1 / (2 * denom))
        self.W_m[0] = self.lambda_ / denom
        self.W_c[0] = self.lambda_ / denom + (1 - alpha**2 + beta)

        # Initial estimates
        self.x = np.zeros(n_dim)
        self.P = np.eye(n_dim)

    def generate_sigma_points(self):
        """
        Generate 2n+1 sigma points using current state and covariance.
        """
        sigma_points = np.zeros((2 * self.n_dim + 1, self.n_dim))
        sigma_points[0] = self.x
        sqrt_P = np.linalg.cholesky((self.n_dim + self.lambda_) * self.P)

        for i in range(self.n_dim):
            sigma_points[i + 1] = self.x + sqrt_P[i]
            sigma_points[self.n_dim + i + 1] = self.x - sqrt_P[i]

        return sigma_points

    def predict(self, process_model):
        """
        Prediction step of UKF.
        Propagates sigma points through process model and computes predicted mean and covariance.

        Args:
            process_model (function): Function f(x) returning next state.
        """
        sigma_points = self.generate_sigma_points()
        sigma_points_pred = np.array([process_model(sp) for sp in sigma_points])

        self.x = np.sum(self.W_m[:, None] * sigma_points_pred, axis=0)

        self.P = self.Q.copy()
        for i in range(2 * self.n_dim + 1):
            diff = sigma_points_pred[i] - self.x
            self.P += self.W_c[i] * np.outer(diff, diff)

        return sigma_points_pred

    def update(self, sigma_points_pred, measurement, measurement_model):
        """
        Update step of UKF.
        Incorporates measurement into prediction using Kalman gain.

        Args:
            sigma_points_pred (ndarray): Predicted sigma points from prediction step.
            measurement (ndarray): Current observation.
            measurement_model (function): Function h(x) returning measurement estimate.
        """
        sigma_points_meas = np.array([measurement_model(sp) for sp in sigma_points_pred])
        z_pred = np.sum(self.W_m[:, None] * sigma_points_meas, axis=0)

        P_zz = self.R.copy()
        P_xz = np.zeros((self.n_dim, len(measurement)))

        for i in range(2 * self.n_dim + 1):
            diff_z = sigma_points_meas[i] - z_pred
            diff_x = sigma_points_pred[i] - self.x
            P_zz += self.W_c[i] * np.outer(diff_z, diff_z)
            P_xz += self.W_c[i] * np.outer(diff_x, diff_z)

        K = np.dot(P_xz, np.linalg.inv(P_zz))
        self.x += np.dot(K, (measurement - z_pred))
        self.P -= np.dot(K, P_zz).dot(K.T)

    def get_state(self):
        """
        Get current state estimate.

        Returns:
            ndarray: Estimated state vector.
        """
        return self.x


if __name__ == "__main__":
    pass  # Optional test logic can be added here
