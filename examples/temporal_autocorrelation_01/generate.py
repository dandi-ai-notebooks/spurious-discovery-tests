import numpy as np
import pandas as pd
from scipy import stats
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

def generate_ar1_process(n_samples, phi, sigma=1.0):
    """Generate an AR(1) process with specified autocorrelation phi."""
    noise = np.random.normal(0, sigma, n_samples)
    series = np.zeros(n_samples)
    series[0] = noise[0]
    for t in range(1, n_samples):
        series[t] = phi * series[t-1] + noise[t]
    return series

# Parameters
n_samples = 600  # 600 time points
variables = ['var1', 'var2', 'var3', 'var4']  # Four variables
metrics = ['metric1', 'metric2', 'metric3', 'metric4', 'metric5']

# Generate autocorrelated time series with high persistence
target = generate_ar1_process(n_samples, phi=0.95)
target = stats.norm.cdf(target)  # Transform to 0-1 scale

# Generate predictor variables with similar persistence for each variable
predictor_phis = {
    'metric1': 0.95,
    'metric2': 0.95,
    'metric3': 0.95,
    'metric4': 0.95,
    'metric5': 0.95
}

# Create DataFrame starting with target variable
data = pd.DataFrame({'target_variable': target})

# Generate time series data for each metric and variable
for metric in metrics:
    for var in variables:
        col_name = f'{metric}_{var}'
        data[col_name] = generate_ar1_process(n_samples, predictor_phis[metric])

# Save to CSV
data.to_csv('data/timeseries_data.csv', index=False)

# Print dataset characteristics
print("Generated dataset with the following characteristics:")
print(f"Total number of time points: {n_samples}")
print(f"Number of variables: {len(variables)}")
print(f"Number of metrics: {len(metrics)}")
print(f"Total number of predictors: {len(variables) * len(metrics)}")

# Calculate correlations and p-values
print("\nStatistical relationships:")
for metric in metrics:
    for var in variables:
        predictor = f'{metric}_{var}'
        corr = np.corrcoef(data['target_variable'], data[predictor])[0, 1]
        print(f"\n{predictor}:")
        print(f"Correlation: {corr:.3f}")
