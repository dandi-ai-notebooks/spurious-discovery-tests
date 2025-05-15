import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.multitest import multipletests

# Set style for plots
plt.style.use('default')

# Load the data
data = pd.read_csv('data/timeseries_data.csv')

# Create directory for plots if it doesn't exist
import os
os.makedirs('plots', exist_ok=True)

# Basic statistics
summary_stats = data.describe()
summary_stats.to_csv('summary_statistics.csv')

# Time series plot of target variable
plt.figure(figsize=(12, 6))
plt.plot(data['target_variable'])
plt.title('Target Variable Over Time')
plt.xlabel('Time Point')
plt.ylabel('Target Value')
plt.savefig('plots/target_timeseries.png')
plt.close()

# Correlation analysis
correlations = data.corr()['target_variable'].sort_values(ascending=False)
correlations.to_csv('correlations.csv')

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='RdBu', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')
plt.close()

# Statistical tests
results = []
p_values = []

# Perform statistical tests for each predictor
predictors = [col for col in data.columns if col != 'target_variable']
for predictor in predictors:
    # Pearson correlation
    corr, p_value = stats.pearsonr(data['target_variable'], data[predictor])
    results.append({
        'predictor': predictor,
        'correlation': corr,
        'p_value': p_value
    })
    p_values.append(p_value)

# Adjust p-values for multiple testing
adjusted_p = multipletests(p_values, method='fdr_bh')[1]
for idx, result in enumerate(results):
    result['adjusted_p_value'] = adjusted_p[idx]

# Save statistical test results
pd.DataFrame(results).to_csv('statistical_tests.csv', index=False)

# Create time series plots for most significant correlations
top_correlations = pd.DataFrame(results).nlargest(5, 'correlation')['predictor']
plt.figure(figsize=(15, 10))
for i, predictor in enumerate(top_correlations, 1):
    plt.subplot(5, 1, i)
    plt.plot(data[predictor], label=predictor)
    plt.title(f'{predictor} Time Series')
    plt.legend()
plt.tight_layout()
plt.savefig('plots/top_correlations_timeseries.png')
plt.close()

# Stationarity test for target variable
adf_result = adfuller(data['target_variable'])
with open('stationarity_test.txt', 'w') as f:
    f.write('Augmented Dickey-Fuller Test Results:\n')
    f.write(f'ADF Statistic: {adf_result[0]}\n')
    f.write(f'p-value: {adf_result[1]}\n')
    for key, value in adf_result[4].items():
        f.write(f'Critical Value ({key}): {value}\n')