import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

# Load the data
data = pd.read_csv('data/timeseries_data.csv')

plt.figure(figsize=(12, 8))

plt.plot(data['target_variable'], 'k-', label='Target', alpha=0.7)
plt.plot(data['metric1_var1'], 'b-', label='Metric1 (Var1)', alpha=0.7)
plt.title('Full Time Series')
plt.xlabel('Time point')
plt.ylabel('Normalized value')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Perform multiple linear regression
# Get all predictor columns
predictor_cols = [col for col in data.columns if col != 'target_variable']
X = np.array(data[predictor_cols])
y = np.array(data['target_variable'])

# Fit the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# Calculate R-squared and F-test statistics
r_squared = model.score(X, y)
n = len(y)
p = len(predictor_cols)
f_stat = (r_squared / p) / ((1 - r_squared) / (n - p - 1))
p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)

print("\nMultiple Linear Regression Results:")
print("-----------------------------------")
print(f"R-squared: {r_squared:.4f}")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4e}")
print("\nCoefficients:")
for predictor, coef in zip(predictor_cols, model.coef_):
    print(f"{predictor}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Create regression plot
plt.figure(figsize=(10, 6))
predictions = model.predict(X)
plt.scatter(predictions, y, alpha=0.1, color='blue', label='Data points')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect prediction')
plt.xlabel('Predicted Target Value')
plt.ylabel('Actual Target Value')
plt.title('Multiple Linear Regression: Predicted vs Actual Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nMultiple Linear Regression Results:\n")
print("-----------------------------------\n")
print(f"R-squared: {r_squared:.4f}\n")
print(f"F-statistic: {f_stat:.4f}\n")
print(f"P-value: {p_value:.4e}\n")
print("\nCoefficients:\n")
for predictor, coef in zip(predictor_cols, model.coef_):
    print(f"{predictor}: {coef:.4f}\n")
print(f"Intercept: {model.intercept_:.4f}\n")
print("\nNote: Due to autocorrelation in the time series,\n")
print("the effective sample size is much smaller than\n")
print("the actual number of samples, which means\n")
print("the p-value is likely substantially underestimated.")
