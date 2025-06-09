"""
RCplusUKF.py

This module defines RC_UKF: a hybrid model that combines Reservoir Computing (RC)
with the Unscented Kalman Filter (UKF) for nonlinear state estimation.

- RC learns the process model from data using an Echo State Network.
- UKF corrects RC predictions using noisy real-time measurements.

Dependencies:
    - numpy
    - RC: Contains ReservoirComputer class
    - UKF: Contains UnscentedKalmanFilter class
"""

import numpy as np
from rcukfpy.RC import ReservoirComputer
from rcukfpy.UKF import UnscentedKalmanFilter


class RC_UKF:
    """
    Combines a Reservoir Computer (RC) with an Unscented Kalman Filter (UKF)
    for data-driven state estimation in nonlinear systems.

    Attributes:
        rc (ReservoirComputer): The reservoir model for dynamics prediction.
        ukf (UnscentedKalmanFilter): UKF for Bayesian state estimation.
        reservoir_state (np.ndarray): Current state of reservoir (n_reservoir, 1).
        measurement_model_noise_std (float): Standard deviation of additive noise in measurements.
    """

    def __init__(
        self,
        n_inputs,
        n_reservoir,
        process_noise,
        measurement_noise,
        rc_params=None,
        ukf_params=None
    ):
        """
        Initialize the RC+UKF framework.

        Args:
            n_inputs (int): Dimensionality of the system (e.g., 3 for Lorenz).
            n_reservoir (int): Number of neurons in the reservoir.
            process_noise (np.ndarray): Process noise covariance matrix (Q).
            measurement_noise (np.ndarray): Measurement noise covariance matrix (R).
            rc_params (dict, optional): RC configuration parameters.
            ukf_params (dict, optional): UKF tuning parameters.
        """
        rc_params = rc_params or {}
        ukf_params = ukf_params or {}

        self.rc = ReservoirComputer(n_inputs=n_inputs, n_reservoir=n_reservoir, **rc_params)
        self.ukf = UnscentedKalmanFilter(n_dim=n_inputs, process_noise=process_noise,
                                         measurement_noise=measurement_noise, **ukf_params)

        self.reservoir_state = np.zeros((self.rc.n_reservoir, 1))  # Initial reservoir state
        self.measurement_model_noise_std = 1e-5  # Slight measurement noise for realism

    def train_reservoir(self, train_inputs, train_outputs):
        """
        Train RC on time-series input-output pairs.

        Args:
            train_inputs (ndarray): Input sequence, shape (T, n_inputs).
            train_outputs (ndarray): Target next states, shape (T, n_inputs).
        """
        states = self.rc.run_reservoir(train_inputs)                     # (T-washout, n_reservoir)
        valid_outputs = train_outputs[self.rc.washout:]                 # Align targets
        self.rc.train_readout(states, valid_outputs)

    def process_model(self, x):
        """
        RC-based prediction function used by UKF.

        Args:
            x (ndarray): Current state estimate, shape (n_inputs,).
        Returns:
            ndarray: Predicted next state, shape (n_inputs,).
        """
        x_col = x.reshape(-1, 1)
        self.reservoir_state = self.rc.update_reservoir_state(self.reservoir_state, x_col)

        # Readout using W_out
        x_next = self.rc.W_out[:, :self.rc.n_reservoir].dot(self.reservoir_state)
        if self.rc.use_bias:
            bias = self.rc.W_out[:, -1].reshape(-1, 1)
            x_next += bias
        return x_next.ravel()

    def reset_reservoir_state(self):
        """
        Reset internal reservoir state to zero. Useful before testing or validation.
        """
        self.reservoir_state = np.zeros((self.rc.n_reservoir, 1))

    def measurement_model(self, x):
        """
        Additive-noise measurement model.

        Args:
            x (ndarray): Current state.
        Returns:
            ndarray: Noisy observation of the state.
        """
        return x + np.random.normal(0, self.measurement_model_noise_std, x.shape)

    def filter_step(self, measurement):
        """
        Run a single prediction and update step of UKF.

        Args:
            measurement (ndarray): Current observed measurement, shape (n_inputs,).
        Returns:
            ndarray: Updated state estimate.
        """
        sigma_points_pred = self.ukf.predict(process_model=self.process_model)
        self.ukf.update(
            sigma_points_pred=sigma_points_pred,
            measurement=measurement,
            measurement_model=self.measurement_model
        )
        return self.ukf.get_state()

    def run_filter(self, measurements):
        """
        Apply UKF filtering to a full sequence of measurements.

        Args:
            measurements (ndarray): Measurement sequence, shape (T, n_inputs).
        Returns:
            ndarray: Filtered state estimates, shape (T, n_inputs).
        """
        estimates = []
        for z in measurements:
            x_est = self.filter_step(z)
            estimates.append(x_est)
        return np.array(estimates)


if __name__ == "__main__":
    # Optional test example here
    pass
