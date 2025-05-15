import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import os

# Create a directory for plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load the dataset
try:
    data = pd.read_csv('data/timeseries_data.csv')
except FileNotFoundError:
    print("Error: 'data/timeseries_data.csv' not found. Make sure the data file is in the 'data' directory.")
    exit()

print("Dataset loaded successfully.Shape:", data.shape)
print("\nFirst 5 rows:\n", data.head())
print("\nColumn names:\n", data.columns.tolist())

# --- Descriptive Statistics ---
print("\nDescriptive Statistics:\n", data.describe())

# --- Visualizations ---

# 1. Plot all time series
plt.figure(figsize=(15, 10))
for i, column in enumerate(data.columns):
    plt.subplot((len(data.columns) + 2) // 3, 3, i + 1) # Dynamic subplot layout
    plt.plot(data.index, data[column], label=column)
    plt.title(column, fontsize=10)
    plt.xlabel("Time Point")
    plt.ylabel("Value")
    plt.tight_layout()
plt.suptitle("Time Series Plots of All Variables", fontsize=16, y=1.02)
plt.savefig('plots/all_timeseries.png')
plt.close()
print("\nSaved plots/all_timeseries.png")

# 2. Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr(method='pearson') # Using Pearson correlation for continuous data
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})
plt.title('Correlation Matrix of Variables (Pearson)')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('plots/correlation_heatmap_pearson.png')
plt.close()
print("Saved plots/correlation_heatmap_pearson.png")

# For time series, Spearman rank correlation can also be useful if non-linear relationships are suspected
# or data is not normally distributed
plt.figure(figsize=(12, 10))
correlation_matrix_spearman = data.corr(method='spearman')
sns.heatmap(correlation_matrix_spearman, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})
plt.title('Correlation Matrix of Variables (Spearman Rank)')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('plots/correlation_heatmap_spearman.png')
plt.close()
print("Saved plots/correlation_heatmap_spearman.png")


# --- Statistical Tests ---
target_col = 'target_variable'
predictor_cols = [col for col in data.columns if col != target_col]

print(f"\n--- Correlation Analysis with '{target_col}' ---")
correlations = {}
for predictor in predictor_cols:
    # Pearson correlation
    pearson_corr, p_value_pearson = pearsonr(data[target_col], data[predictor])
    # Spearman correlation
    spearman_corr, p_value_spearman = spearmanr(data[target_col], data[predictor])
    correlations[predictor] = {
        'pearson_corr': pearson_corr,
        'p_value_pearson': p_value_pearson,
        'spearman_corr': spearman_corr,
        'p_value_spearman': p_value_spearman
    }
    print(f"\n{predictor}:")
    print(f"  Pearson correlation with {target_col}: {pearson_corr:.4f} (p-value: {p_value_pearson:.4g})")
    print(f"  Spearman correlation with {target_col}: {spearman_corr:.4f} (p-value: {p_value_spearman:.4g})")

# Store correlation results
correlation_df = pd.DataFrame(correlations).T
correlation_df.to_csv('correlation_results.csv')
print("\nSaved correlation_results.csv")

# --- Granger Causality (Example for a few potentially correlated variables) ---
# Granger causality requires stationary data. We should check for stationarity first (e.g., ADF test).
# For simplicity in this script, we'll demonstrate it on a couple of variables.
# Let's pick variables with high absolute Pearson correlation with the target.
# Sort by absolute Pearson correlation
sorted_predictors = sorted(predictor_cols, key=lambda p: abs(correlations[p]['pearson_corr']), reverse=True)

print(f"\n--- Granger Causality Tests (up to 5 lags) ---")
print("Note: Granger causality tests likelihood, not true causation. Assumes stationarity (not explicitly tested here for brevity).")

max_lag = 5
test_count = 0
for predictor in sorted_predictors:
    if test_count >= 3: # Test top 3 correlated predictors
        break
    print(f"\nTesting if '{predictor}' Granger-causes '{target_col}':")
    try:
        gc_results_target = grangercausalitytests(data[[target_col, predictor]], maxlag=max_lag, verbose=False)
        min_p_value_target = 1.0
        for lag in gc_results_target:
            p_val = gc_results_target[lag][0]['ssr_ftest'][1] # p-value from F-test
            min_p_value_target = min(min_p_value_target, p_val)
        print(f"  Minimum p-value for '{predictor}' -> '{target_col}': {min_p_value_target:.4g}")

        print(f"Testing if '{target_col}' Granger-causes '{predictor}':")
        gc_results_predictor = grangercausalitytests(data[[predictor, target_col]], maxlag=max_lag, verbose=False)
        min_p_value_predictor = 1.0
        for lag in gc_results_predictor:
            p_val = gc_results_predictor[lag][0]['ssr_ftest'][1]
            min_p_value_predictor = min(min_p_value_predictor, p_val)
        print(f"  Minimum p-value for '{target_col}' -> '{predictor}': {min_p_value_predictor:.4g}")
        test_count +=1
    except Exception as e:
        print(f"  Could not perform Granger causality test for {predictor} and {target_col}: {e}")


print("\n--- Script Finished ---")