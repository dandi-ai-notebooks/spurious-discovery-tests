import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.signal import correlate

# Read the data
data = pd.read_csv('data/neural_firing_rates.csv')

# Basic statistical summaries
stats_a = data['region_a_firing_rate'].describe()
stats_b = data['region_b_firing_rate'].describe()

# Create output directory if it doesn't exist
import os
os.makedirs('figures', exist_ok=True)

# Data quality assessment
def detect_outliers(series, threshold=3):
    z_scores = np.abs(stats.zscore(series))
    return z_scores > threshold

outliers_a = detect_outliers(data['region_a_firing_rate'])
outliers_b = detect_outliers(data['region_b_firing_rate'])

# Plot time series with outliers highlighted
plt.figure(figsize=(15, 8))
plt.plot(data['region_a_firing_rate'], label='Region A', alpha=0.5)
plt.plot(data['region_b_firing_rate'], label='Region B', alpha=0.5)
plt.scatter(np.where(outliers_a)[0], data.loc[outliers_a, 'region_a_firing_rate'], 
           color='red', alpha=0.5, s=20, label='Region A Outliers')
plt.scatter(np.where(outliers_b)[0], data.loc[outliers_b, 'region_b_firing_rate'], 
           color='orange', alpha=0.5, s=20, label='Region B Outliers')
plt.xlabel('Time (seconds)')
plt.ylabel('Firing Rate (spikes/second)')
plt.title('Neural Firing Rates Over Time with Outliers')
plt.legend()
plt.savefig('figures/time_series.png')
plt.close()

# Distribution plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(data['region_a_firing_rate'], ax=ax1, kde=True)
ax1.set_title('Region A Firing Rate Distribution')
sns.histplot(data['region_b_firing_rate'], ax=ax2, kde=True)
ax2.set_title('Region B Firing Rate Distribution')
plt.tight_layout()
plt.savefig('figures/distributions.png')
plt.close()

# Correlation analysis
correlation = stats.pearsonr(data['region_a_firing_rate'], data['region_b_firing_rate'])

# Scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(data['region_a_firing_rate'], data['region_b_firing_rate'], alpha=0.1)
plt.xlabel('Region A Firing Rate')
plt.ylabel('Region B Firing Rate')
plt.title(f'Correlation between Region A and B\nr={correlation[0]:.3f}, p={correlation[1]:.3e}')
plt.savefig('figures/correlation.png')
plt.close()

# Cross-correlation analysis
max_lag = 100  # Look for relationships up to 100 seconds in either direction
region_a_centered = data['region_a_firing_rate'] - data['region_a_firing_rate'].mean()
region_b_centered = data['region_b_firing_rate'] - data['region_b_firing_rate'].mean()
cross_corr = correlate(region_a_centered, region_b_centered)
lags = np.arange(-max_lag, max_lag + 1)
cross_corr = cross_corr[len(cross_corr)//2 - max_lag:len(cross_corr)//2 + max_lag + 1]
cross_corr = cross_corr / len(region_a_centered)  # Normalize

plt.figure(figsize=(10, 6))
plt.plot(lags, cross_corr)
plt.xlabel('Lag (seconds)')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation between Region A and B')
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.savefig('figures/cross_correlation.png')
plt.close()

# Save numerical results
def convert_to_python_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    return obj

results = {
    'Region A Statistics': {k: float(v) for k, v in stats_a.to_dict().items()},
    'Region B Statistics': {k: float(v) for k, v in stats_b.to_dict().items()},
    'Correlation': {
        'coefficient': float(correlation[0]),
        'p_value': float(correlation[1])
    },
    'Data Quality': {
        'Region A Outliers': int(sum(outliers_a)),
        'Region B Outliers': int(sum(outliers_b))
    }
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=4)