import yfinance as yf
import matplotlib.pyplot as plt
import json
import os

# 1. Load the saved beta, R, and Q values from the JSON file

parameters_file = 'model_params.json'

with open(parameters_file, 'r') as json_file:
        loaded_params = json.load(json_file)

#2. Download the historical data for Visa and MasterCard from Yahoo Finance for the period after the initial analysis

data_visa = yf.download("V", start="2025-03-04", end="2026-09-01", group_by="column")
data_master = yf.download("MA", start="2025-03-04", end="2026-09-01", group_by="column")

# 3. Extract columns and force conversion to 1D squeeze arrays 
visa = data_visa['Close']['V'].squeeze()
master = data_master['Close']['MA'].squeeze()

# 4. Initialize the Kalman filter parameters using the loaded values

beta = loaded_params['beta']
P = 0.1
R = loaded_params['R']
Q = loaded_params['Q']

# 5. Initialize lists to store the Kalman filter outputs

betas = []
visa_predicted = []
residuals = []
residuals_zscore = []
system_vars = []
residuals_zscore = []

# 6. Apply the Kalman filter to the Visa and MasterCard data

for i in range(len(visa)):

    V = visa.iloc[i].item() #extract the numeric scaler from the Pandas index
    H = master.iloc[i].item()

    visa_predicted.append(beta * H)
    betas.append(beta)

    P = P + Q
    residual = V - beta * H
    system_var = P * H * H + R
    KG = (P * H) / (system_var)
    beta = beta + KG * residual
    P = (1 - KG * H) * P

    residuals.append(residual)
    system_vars.append(system_var)
    residuals_zscore.append(residual / (system_var ** 0.5))



# 7. Plotting the actual Visa prices, the Kalman filter predicted Visa prices, and the beta coefficient over time

plt.figure(figsize=(12, 6))
plt.plot(visa.index, visa.values, label='Actual Visa (V)', color='blue')
plt.plot(visa.index, visa_predicted, label='Kalman Predicted Visa', color='orange', linestyle='--')
plt.title('Visa Price vs Kalman Filter Rolling Prediction')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(visa.index, betas, label='beta', color='green')
plt.title('Kalman Filter Beta Coefficient')
plt.xlabel('Date')
plt.ylabel('Beta')
plt.legend()
plt.grid(True)
plt.show()