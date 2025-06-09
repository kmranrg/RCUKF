"""
RC.py

Implements a Reservoir Computer (RC), also known as an Echo State Network (ESN),
for modeling and predicting nonlinear dynamical systems.

Key features:
- Fixed, sparse reservoir with random weights
- Only the output weights are trained via ridge regression
- Suitable for chaotic systems like Lorenz, Rössler, Mackey-Glass

Dependencies:
    - numpy
    - matplotlib.pyplot (for optional plotting)
"""

import numpy as np
import matplotlib.pyplot as plt


class ReservoirComputer:
    """
    Reservoir Computer (Echo State Network) for time-series prediction.

    Attributes:
        n_inputs (int): Input dimensionality
        n_reservoir (int): Number of reservoir neurons
        spectral_radius (float): Controls stability/memory of reservoir
        sparsity (float): Fraction of non-zero weights in the reservoir
        reg (float): Regularization coefficient for ridge regression
        noise_std (float): Std. deviation of Gaussian input noise
        leak_rate (float): Leaking rate (1.0 means no leak)
        washout (int): Initial steps to discard
        use_bias (bool): Whether to add a bias term to readout
        input_scale (float): Scaling factor for input weights
        W_in (ndarray): Input-to-reservoir weight matrix
        W (ndarray): Reservoir recurrent weight matrix
        W_out (ndarray): Trained output weights
    """

    def __init__(
        self,
        n_inputs,
        n_reservoir,
        spectral_radius=0.9,
        sparsity=0.2,
        reg=1e-5,
        noise_std=0.1,
        random_seed=None,
        leak_rate=1.0,
        washout=100,
        use_bias=True,
        input_scale=0.1
    ):
        if random_seed is not None:
            np.random.seed(random_seed)

        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.reg = reg
        self.noise_std = noise_std
        self.leak_rate = leak_rate
        self.washout = washout
        self.use_bias = use_bias
        self.input_scale = input_scale

        self.W_in = self.initialize_input_weights()
        self.W = self.initialize_reservoir()
        self.W_out = None  # To be trained later

    def initialize_input_weights(self):
        """
        Initialize input-to-reservoir weights with uniform random values scaled by input_scale.
        Returns:
            ndarray: Weight matrix of shape (n_reservoir, n_inputs)
        """
        return self.input_scale * (2 * np.random.rand(self.n_reservoir, self.n_inputs) - 1)

    def initialize_reservoir(self):
        """
        Initialize sparse reservoir weights and rescale to have the desired spectral radius.
        Returns:
            ndarray: Weight matrix of shape (n_reservoir, n_reservoir)
        """
        W = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        mask = np.random.rand(*W.shape) > self.sparsity
        W[mask] = 0.0

        # Rescale to desired spectral radius
        eigs = np.linalg.eigvals(W)
        W *= self.spectral_radius / np.max(np.abs(eigs))
        return W

    def add_noise(self, inputs):
        """
        Add Gaussian noise to inputs if noise_std > 0.
        Args:
            inputs (ndarray): Input array of shape (T, n_inputs)
        Returns:
            ndarray: Noisy inputs
        """
        if self.noise_std > 0:
            return inputs + np.random.normal(0, self.noise_std, inputs.shape)
        return inputs

    def update_reservoir_state(self, x_prev, u):
        """
        Update reservoir state using leaky integration.
        Args:
            x_prev (ndarray): Previous reservoir state, shape (n_reservoir, 1)
            u (ndarray): Current input, shape (n_inputs, 1)
        Returns:
            ndarray: Updated reservoir state
        """
        pre_activation = self.W_in @ u + self.W @ x_prev
        return (1 - self.leak_rate) * x_prev + self.leak_rate * np.tanh(pre_activation)

    def run_reservoir(self, inputs):
        """
        Run the reservoir on a time-series input and return post-washout states.
        Args:
            inputs (ndarray): Input data of shape (T, n_inputs)
        Returns:
            ndarray: Reservoir states after washout, shape (T-washout, n_reservoir)
        """
        inputs = self.add_noise(inputs)
        n_steps = inputs.shape[0]
        x = np.zeros((self.n_reservoir, 1))
        states = np.zeros((n_steps, self.n_reservoir))

        for t in range(n_steps):
            u = inputs[t].reshape(-1, 1)
            x = self.update_reservoir_state(x, u)
            states[t] = x.ravel()

        if self.washout >= n_steps:
            raise ValueError("Washout is greater than or equal to number of input steps.")
        return states[self.washout:]

    def train_readout(self, states, outputs):
        """
        Train readout weights using ridge regression.
        Args:
            states (ndarray): Reservoir states, shape (N, n_reservoir)
            outputs (ndarray): Target outputs, shape (N, output_dim)
        """
        if self.use_bias:
            states = np.hstack([states, np.ones((states.shape[0], 1))])

        X = states
        Y = outputs
        ridge = self.reg * np.eye(X.shape[1])
        self.W_out = (Y.T @ X) @ np.linalg.inv(X.T @ X + ridge)

    def predict(self, states):
        """
        Predict output using trained readout weights.
        Args:
            states (ndarray): Input states, shape (N, n_reservoir)
        Returns:
            ndarray: Predicted outputs, shape (N, output_dim)
        """
        if self.W_out is None:
            raise ValueError("Readout weights not trained. Call train_readout first.")

        if self.use_bias:
            states = np.hstack([states, np.ones((states.shape[0], 1))])

        return states @ self.W_out.T

    def plot_results(self, true_values, predicted_values, title="Prediction Performance"):
        """
        Plot true vs predicted values for 1D output.
        Args:
            true_values (ndarray): Ground truth
            predicted_values (ndarray): Model predictions
            title (str): Title of the plot
        """
        plt.figure(figsize=(8, 4))
        plt.plot(true_values, label="True Output", color='blue')
        plt.plot(predicted_values, label="Predicted Output", linestyle="--", color='orange')
        plt.xlabel("Time Step")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Optional: Add a quick test here in future
    pass
