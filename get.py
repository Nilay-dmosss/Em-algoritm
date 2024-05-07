import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data (two clusters)
np.random.seed(42)
data_cluster1 = np.random.normal(loc=2, scale=1, size=100)
data_cluster2 = np.random.normal(loc=6, scale=1, size=100)
data = np.concatenate([data_cluster1, data_cluster2])

# Initialize parameters
pi1 = 0.5  # Initial guess for cluster 1 probability
pi2 = 0.5  # Initial guess for cluster 2 probability
mu1 = 3.0  # Initial guess for cluster 1 mean
mu2 = 5.0  # Initial guess for cluster 2 mean
sigma1 = 1.0  # Initial guess for cluster 1 standard deviation
sigma2 = 1.0  # Initial guess for cluster 2 standard deviation

# EM algorithm
num_iterations = 100
for _ in range(num_iterations):
    # E-step: Compute responsibilities
    gamma1 = pi1 * np.exp(-0.5 * ((data - mu1) / sigma1) ** 2) / (np.sqrt(2 * np.pi) * sigma1)
    gamma2 = pi2 * np.exp(-0.5 * ((data - mu2) / sigma2) ** 2) / (np.sqrt(2 * np.pi) * sigma2)
    total_gamma = gamma1 + gamma2
    r1 = gamma1 / total_gamma
    r2 = gamma2 / total_gamma

    # M-step: Update parameters
    pi1 = np.mean(r1)
    pi2 = np.mean(r2)
    mu1 = np.sum(r1 * data) / np.sum(r1)
    mu2 = np.sum(r2 * data) / np.sum(r2)
    sigma1 = np.sqrt(np.sum(r1 * (data - mu1) ** 2) / np.sum(r1))
    sigma2 = np.sqrt(np.sum(r2 * (data - mu2) ** 2) / np.sum(r2))

# Plot the data and estimated Gaussian distributions
plt.hist(data, bins=30, density=True, alpha=0.6, color='blue')
plt.plot(np.linspace(0, 8, 1000), pi1 * np.exp(-0.5 * ((np.linspace(0, 8, 1000) - mu1) / sigma1) ** 2) / (np.sqrt(2 * np.pi) * sigma1), label='Cluster 1', color='red')
plt.plot(np.linspace(0, 8, 1000), pi2 * np.exp(-0.5 * ((np.linspace(0, 8, 1000) - mu2) / sigma2) ** 2) / (np.sqrt(2 * np.pi) * sigma2), label='Cluster 2', color='green')
plt.xlabel('Data')
plt.ylabel('Density')
plt.title('Expectation-Maximization (EM) for Gaussian Mixture Models')
plt.legend()
plt.show()
