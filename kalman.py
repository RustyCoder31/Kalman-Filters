import random
import matplotlib.pyplot as plt

import numpy as np

true_value = 10.0     # starting point
Q_true = 0.5           # how much the true value actually drifts each step (variance)
R_true = 4.0           # how noisy your "sensor" actually is (variance)

true_values = []
measurements = []

for t in range(50):
    true_value += np.random.normal(0, np.sqrt(Q_true))       # random walk step
    measurement = true_value + np.random.normal(0, np.sqrt(R_true))  # noisy observation
    true_values.append(true_value)
    measurements.append(measurement)


estimate = 10;
uncertainity_estimate = 1;
Q = 0.4;  # process variance
R = 3;  # estimate of measurement variance, change to see effect


estimates = []  # initial estimate

for i in range (50):
    random_measurement = measurements[i]
    predicted_estimate = estimate
    predicted_uncertainity_estimate = uncertainity_estimate + Q
    KG = predicted_uncertainity_estimate / (predicted_uncertainity_estimate + R)
    estimate = predicted_estimate + KG * (random_measurement - predicted_estimate)
    uncertainity_estimate = (1 - KG) * predicted_uncertainity_estimate
    print("Measurement: ", random_measurement, "Estimate: ", round(estimate,2), "Uncertainty: ", round(uncertainity_estimate,2))
    estimates.append(estimate)

plt.plot(true_values, label='True Value')
plt.plot(measurements, label='Measurements')
plt.plot(estimates, label='Estimates')
plt.legend()
plt.show()