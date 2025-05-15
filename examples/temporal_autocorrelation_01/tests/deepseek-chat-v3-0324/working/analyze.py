import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import pearsonr
import os

# Create plots directory
os.makedirs('plots', exist_ok=True)

# Load data
df = pd.read_csv('data/timeseries_data.csv')

# 1. Descriptive statistics
stats = df.describe().to_string()

# 2. Correlation analysis
correlations = {}
for col in df.columns:
    if col != 'target_variable':
        corr, pval = pearsonr(df['target_variable'], df[col])
        correlations[col] = {'correlation': corr, 'p-value': pval}

# 3. Time series decomposition
def plot_decomposition(series, metric_name):
    result = seasonal_decompose(series, model='additive', period=24)
    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.savefig(f'plots/{metric_name}_decomposition.png')
    plt.close()

plot_decomposition(df['target_variable'], 'target')

# 4. Granger causality tests (maxlag=5)
granger_results = {}
for col in df.columns:
    if col != 'target_variable':
        test_result = grangercausalitytests(df[[col, 'target_variable']], maxlag=5)
        granger_results[col] = test_result

# 5. Stationarity tests
stationarity = {}
for col in df.columns:
    test = adfuller(df[col])
    stationarity[col] = {
        'ADF Statistic': test[0],
        'p-value': test[1],
        'Critical Values': test[4]
    }

# 6. Visualizations
# Time series plot of top correlations
top_corrs = sorted([(k, v['correlation']) for k, v in correlations.items()], 
                   key=lambda x: abs(x[1]), reverse=True)[:4]

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for i, (col, corr) in enumerate(top_corrs):
    row, col_pos = divmod(i, 2)
    df.plot(x=col, y='target_variable', kind='scatter', ax=ax[row][col_pos], 
            title=f'{col} vs Target (r={corr:.2f})')
plt.tight_layout()
plt.savefig('plots/top_correlations.png')
plt.close()

# Save results for report
with open('analysis_results.txt', 'w') as f:
    f.write("=== Descriptive Statistics ===\n")
    f.write(stats + "\n\n")
    f.write("=== Correlations ===\n")
    for col, vals in correlations.items():
        f.write(f"{col}: {vals}\n")
    f.write("\n=== Stationarity Tests ===\n")
    for col, vals in stationarity.items():
        f.write(f"{col}: {vals}\n")