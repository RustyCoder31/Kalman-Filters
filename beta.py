import statsmodels.api as sm
import yfinance as yf
import matplotlib.pyplot as plt
import json
from statsmodels.regression.rolling import RollingOLS
import numpy as np

# initialize the dataframes for Visa and MasterCard
data_visa = yf.download("V", start="2025-01-01", end="2025-03-03", group_by="column")
data_master = yf.download("MA", start="2025-01-01", end="2025-03-03", group_by="column")

# 2. Extract columns and force conversion to 1D squeeze arrays 
visa = data_visa['Close']['V'].squeeze()
master = data_master['Close']['MA'].squeeze()

# 3. Create the OLS model and fit it to find our beta (slope) and R (variance of the residuals)

model = sm.OLS(visa, master).fit()  # Create the OLS model
beta = model.params['MA']
visa_predicted_static = model.predict(master)
static_spread = visa - visa_predicted_static
variance = np.var(static_spread)

print("Static OLS Regression Results:")
print(f"Beta (Slope): {beta}")
print(f"Static Spread (Visa - Predicted Visa): {static_spread}")
print(model.summary())

# 4. Create a rolling OLS model with a window of 10 days to capture the dynamic relationship between Visa and MasterCard

rolling_model = RollingOLS(visa, master, window=10)  # 10-day rolling window
rolling_results = rolling_model.fit()
rolling_params = rolling_results.params

print("--- Sample of the Rolling Output DataFrame ---")
print(rolling_params.tail(15))

Q_vals = rolling_params['MA'].values
Q_vals = Q_vals[~np.isnan(Q_vals)]  # Remove NaN values
Qvals_diff = np.diff(Q_vals)
Qvals_diff = Qvals_diff[~np.isnan(Qvals_diff)]  # Remove NaN values from the differences
Q = np.var(Qvals_diff)

#5. Save the beta, R, and Q values to a JSON file

alpha_beta_values = {
    'beta': beta,
    'R': variance,
    'Q': Q
}

with open('model_params.json', 'w') as json_file:
    json.dump(alpha_beta_values, json_file, indent=4)

print("Successfully saved OLS Alpha and Beta to 'model_params.json'!")
