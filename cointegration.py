from pairs_data import residuals_zscore
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.regression.rolling import RollingOLS


adf_result = adfuller(residuals_zscore)
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("ADF results:", adf_result)

# initialize the dataframes for Visa and MasterCard
data_visa = yf.download("V", start="2025-03-04", end="2026-09-01", group_by="column")
data_master = yf.download("MA", start="2025-03-04", end="2026-09-01", group_by="column")

# 2. Extract columns and force conversion to 1D squeeze arrays 
visa = data_visa['Close']['V'].squeeze()
master = data_master['Close']['MA'].squeeze()

# 3. Create the OLS model and fit it to find our beta (slope) and R (variance of the residuals)

model = sm.OLS(visa, master).fit()  # Create the OLS model
beta = model.params['MA']
visa_predicted_static = model.predict(master)
static_spread = visa - visa_predicted_static
R = np.var(static_spread)

print("Static OLS Regression Results:")
print(f"Beta (Slope): {beta}")
print(f"Static Spread (Visa - Predicted Visa): {static_spread}")

rolling_model = RollingOLS(visa, master, window=10)  # 10-day rolling window
rolling_results = rolling_model.fit()
rolling_params = rolling_results.params

Q_vals = rolling_params['MA'].values
Q_vals = Q_vals[~np.isnan(Q_vals)]  # Remove NaN values
Qvals_diff = np.diff(Q_vals)
Qvals_diff = Qvals_diff[~np.isnan(Qvals_diff)]  # Remove NaN values from the differences
Q = np.var(Qvals_diff)

print("Rolling OLS Regression Results:")
print(f"Rolling Beta (Slope): {Q}")

P = 0.1
visa_predicted = []
betas = []
residuals = []
system_vars = []
residuals_zscore_static = []

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
    residuals_zscore_static.append(residual / (system_var ** 0.5))

adf_result = adfuller(residuals_zscore_static)
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("ADF static results:", adf_result)


# 1. Convert your z-scores to a Pandas Series (assuming it's a list)
z_scores = pd.Series(residuals_zscore_static)

# 2. Calculate the exact variables the ADF test uses
yesterday_position = z_scores.shift(1)  # y_{t-1}
todays_move = z_scores.diff()           # Delta y_t

# 3. Clean the data (remove the first NaN row caused by shifting)
valid_data = pd.DataFrame({
    'x': yesterday_position,
    'y': todays_move
}).dropna()

# 4. Calculate the line of best fit (Y = mx + b)
m, b = np.polyfit(valid_data['x'], valid_data['y'], 1)

# 5. Plot the true ADF Visualization
plt.figure(figsize=(10, 6))
plt.scatter(valid_data['x'], valid_data['y'], alpha=0.5, color='blue', label='Daily Data Points')

# Draw the Gamma line
plt.plot(valid_data['x'], m * valid_data['x'] + b, color='red', linewidth=2, label=f'Line of Best Fit (Gamma = {m:.4f})')

# Add crosshairs at 0,0 for visual reference
plt.axhline(0, color='black', linestyle='--')
plt.axvline(0, color='black', linestyle='--')

plt.title('ADF Regression: The Rubber Band Effect')
plt.xlabel("Yesterday's Position (Z-Score)")
plt.ylabel("Today's Move (Change in Z-Score)")
plt.legend()
plt.grid(True)
plt.show()

