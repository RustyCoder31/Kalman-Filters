import yfinance as yf
import matplotlib.pyplot as plt
import json

# 1. Load the calibrated static beta, R, and Q (same params pairs_data.py uses)

with open('model_params.json', 'r') as json_file:
    loaded_params = json.load(json_file)

static_beta = loaded_params['beta']
R = loaded_params['R']
Q = loaded_params['Q']

# 2. Download the same out-of-sample window used everywhere else

data_visa = yf.download("V", start="2025-03-04", end="2026-09-01", group_by="column")
data_master = yf.download("MA", start="2025-03-04", end="2026-09-01", group_by="column")

visa = data_visa['Close']['V'].squeeze()
master = data_master['Close']['MA'].squeeze()

# 3. Naive prediction: the calibrated beta never updates, held constant for the whole window

naive_predicted = static_beta * master

# 4. Kalman prediction: same loop as pairs_data.py, beta updates every step

beta = static_beta
P = 0.1
kalman_predicted = []

for i in range(len(visa)):
    V = visa.iloc[i].item()
    H = master.iloc[i].item()

    kalman_predicted.append(beta * H)

    P = P + Q
    residual = V - beta * H
    system_var = P * H * H + R
    KG = (P * H) / system_var
    beta = beta + KG * residual
    P = (1 - KG * H) * P

# 5. Plot actual vs naive (fixed beta) vs Kalman (dynamic beta)

plt.figure(figsize=(12, 6))
plt.plot(visa.index, visa.values, label='Actual Visa (V)', color='blue')
plt.plot(visa.index, naive_predicted.values, label='Naive Fixed-Beta Prediction', color='red', linestyle=':')
plt.plot(visa.index, kalman_predicted, label='Kalman Dynamic-Beta Prediction', color='orange', linestyle='--')
plt.title('Naive Fixed Beta vs Kalman Filter vs Actual Visa Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.savefig('Naive vs Kalman.png')
plt.show()
