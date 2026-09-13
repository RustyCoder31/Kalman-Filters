import random
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# 1. Generate a random walk for the true value and noisy measurements

y = np.zeros(201)  # Initialize the true value array
y[0] = 0  # Set the initial true value

for t in range(200):
    noise_t = np.random.normal(0, 1)  # Gaussian noise with mean 0 and std dev 1
    y[t+1] = y[t] + noise_t  # Update the true value with noise


# 2. Generate a stationary series for the true value and noisy measurements

x = np.zeros(201)  # Initialize the true value array
x[0] = 0  # Set the initial true value
mu = 5
phi = 0.8

for t in range(200):
    noise_i = np.random.normal(0, 1)  # Gaussian noise with mean 0 and std dev 1
    x[t+1] = mu + phi *x[t] + noise_i  # Update the true value with noise

# 3. Perform the Augmented Dickey-Fuller (ADF) test on both series to check for stationarity

adf_y = adfuller(y)
adf_x = adfuller(x)

print("ADF Test for Random Walk (y):")
print(f"ADF Statistic: {adf_y[0]}")
print(f"p-value: {adf_y[1]}")
print("ADF Test for AR(1) Process (x):")    
print(f"ADF Statistic: {adf_x[0]}")
print(f"p-value: {adf_x[1]}")

plt.figure(figsize=(12, 6))
plt.plot(y, label='True Value (Random Walk)', color='blue')
plt.plot(x, label='True Value (AR(1))', color='red')
plt.title('Comparison of Random Walk and AR(1) Processes')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()