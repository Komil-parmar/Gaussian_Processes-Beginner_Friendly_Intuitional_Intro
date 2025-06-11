import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Generate time points (in hours)
X = np.linspace(0, 10, 8).reshape(-1, 1)


# True function (e.g., sensor temperature over time)
def true_function(x):
	return np.sin(x) + 0.5 * np.cos(2 * x)


# Add Gaussian noise
y = true_function(X) + np.random.normal(0, 0.2, X.shape)

# Plot
plt.scatter(X, y, label='Noisy observations')
plt.plot(X, true_function(X), label='True function', color='orange')
plt.xlabel("Time (hours)")
plt.ylabel("Temperature")
plt.legend()
plt.title("Sensor readings over time")
plt.show()


def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
	"""RBF (Squared Exponential) kernel."""
	sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + \
		 np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
	return sigma_f ** 2 * np.exp(-0.5 / length_scale ** 2 * sqdist)


# Test points for prediction
X_test = np.linspace(0, 10, 100).reshape(-1, 1)

# Kernel parameters
length_scale = 1.0
# length_scale = 2.0
sigma_f = 1.0
# sigma_n = 0.2  # noise level
sigma_n = 0.1  # noise level

# Covariance matrices
# K = rbf_kernel(X, X, length_scale, sigma_f)
K = rbf_kernel(X, X, length_scale, sigma_f) + sigma_n ** 2 * np.eye(len(X))
K_s = rbf_kernel(X, X_test, length_scale, sigma_f)
K_ss = rbf_kernel(X_test, X_test, length_scale, sigma_f)

# Invert K
K_inv = np.linalg.inv(K)

# Posterior mean and covariance
mu_s = K_s.T @ K_inv @ y
cov_s = K_ss - K_s.T @ K_inv @ K_s
std_s = np.sqrt(np.diag(cov_s))

plt.figure(figsize=(10, 6))
plt.plot(X, y, 'kx', label='Train data')
plt.plot(X_test, true_function(X_test), 'orange', label='True function')
plt.plot(X_test, mu_s, 'b', label='GP mean')
plt.fill_between(X_test.ravel(),
		 mu_s.ravel() - 2 * std_s,
		 mu_s.ravel() + 2 * std_s,
		 color='lightblue', alpha=0.5, label='95% confidence interval')
plt.legend()
plt.title("Gaussian Process Regression")
plt.xlabel("Time (hours)")
plt.ylabel("Temperature")
plt.show()
