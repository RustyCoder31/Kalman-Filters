import yfinance as yf
import matplotlib.pyplot as plt
import json
import os

parameters_file = 'model_params.json'

with open(parameters_file, 'r') as json_file:
        loaded_params = json.load(json_file)

# 1. Add group_by="column" to flatten the DataFrame structure automatically
data_visa = yf.download("V", start="2025-03-04", end="2026-09-01", group_by="column")
data_master = yf.download("MA", start="2025-03-04", end="2026-09-01", group_by="column")

# 2. Extract columns and force conversion to 1D squeeze arrays or Series values
visa = data_visa['Close']['V'].squeeze()
master = data_master['Close']['MA'].squeeze()



beta = loaded_params['beta']
P = 0.1
R = loaded_params['R']
Q = loaded_params['Q']

betas = []
visa_predicted = []

for i in range(len(visa)):
    # .item() extracts the naked numeric scalar float from the Pandas index
    V = visa.iloc[i].item()
    H = master.iloc[i].item()

    # This will now append a pure number to your list
    visa_predicted.append(beta * H)
    P = P + Q

    betas.append(beta)
    residual = V - beta * H
    
    KG = (P * H) / (P * H * H + R)
    beta = beta + KG * residual
    P = (1 - KG * H) * P

# 3. Plotting cleanly
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
