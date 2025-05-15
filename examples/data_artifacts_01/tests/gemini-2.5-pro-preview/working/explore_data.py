import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np

# --- Configuration ---
DATA_FILE = 'data/neural_firing_rates.csv'
OUTPUT_DIR = 'plots' # Let's save plots in a dedicated directory
TIMESERIES_PLOT_FILE = f'{OUTPUT_DIR}/timeseries_plot.png'
SCATTER_PLOT_FILE = f'{OUTPUT_DIR}/scatter_plot.png'
CLEANED_TIMESERIES_PLOT_FILE = f'{OUTPUT_DIR}/cleaned_timeseries_plot.png'
CLEANED_SCATTER_PLOT_FILE = f'{OUTPUT_DIR}/cleaned_scatter_plot.png'

# Create output directory if it doesn't exist
import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Load Data ---
try:
    data = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: The file {DATA_FILE} was not found.")
    exit()

print("--- Dataset Info ---")
data.info()
print("\n--- First 5 Rows ---")
print(data.head())

# --- Descriptive Statistics (Initial) ---
print("\n--- Initial Descriptive Statistics ---")
print(data[['region_a_firing_rate', 'region_b_firing_rate']].describe())

# --- Visualizations (Initial) ---
print(f"\n--- Generating Initial Plots ---")
# Time series plot
plt.figure(figsize=(15, 6))
plt.plot(data['time_seconds'], data['region_a_firing_rate'], label='Region A Firing Rate', alpha=0.7)
plt.plot(data['time_seconds'], data['region_b_firing_rate'], label='Region B Firing Rate', alpha=0.7)
plt.xlabel('Time (seconds)')
plt.ylabel('Firing Rate (spikes/second)')
plt.title('Neural Firing Rates Over Time (Raw Data)')
plt.legend()
plt.grid(True)
plt.savefig(TIMESERIES_PLOT_FILE)
plt.close()
print(f"Saved initial time series plot to {TIMESERIES_PLOT_FILE}")

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='region_a_firing_rate', y='region_b_firing_rate', data=data, alpha=0.5)
plt.title('Scatter Plot of Firing Rates (Region A vs Region B - Raw Data)')
plt.xlabel('Region A Firing Rate (spikes/second)')
plt.ylabel('Region B Firing Rate (spikes/second)')
plt.grid(True)
plt.savefig(SCATTER_PLOT_FILE)
plt.close()
print(f"Saved initial scatter plot to {SCATTER_PLOT_FILE}")

# --- Handle Potential Data Corruption/Outliers ---
# The readme mentions potential data corruption.
# A simple approach: check for NaNs and extreme outliers.
# Physiological firing rates are typically non-negative.
# Let's assume any negative value or extremely high value (e.g., > 1000 spikes/sec, which is very high) is corruption.
# Also, check for NaN values.

print("\n--- Data Cleaning ---")
initial_rows = len(data)
print(f"Initial number of data points: {initial_rows}")

# Check for NaNs
nan_rows = data[data['region_a_firing_rate'].isna() | data['region_b_firing_rate'].isna()]
if not nan_rows.empty:
    print(f"Found {len(nan_rows)} rows with NaN values. Removing them.")
    data.dropna(subset=['region_a_firing_rate', 'region_b_firing_rate'], inplace=True)

# Filter out non-physiological values (e.g., negative or extremely high rates)
# For this example, let's set a plausible upper bound. Real analysis would require domain knowledge.
LOWER_BOUND = 0
UPPER_BOUND = 200 # Assuming spikes/sec typically don't exceed this by much for sustained periods.

data_cleaned = data[
    (data['region_a_firing_rate'] >= LOWER_BOUND) & (data['region_a_firing_rate'] <= UPPER_BOUND) &
    (data['region_b_firing_rate'] >= LOWER_BOUND) & (data['region_b_firing_rate'] <= UPPER_BOUND)
].copy() # Use .copy() to avoid SettingWithCopyWarning

cleaned_rows = len(data_cleaned)
rows_removed = initial_rows - data.shape[0] # After NaN removal
rows_removed_by_filtering = data.shape[0] - cleaned_rows # After value filtering

print(f"Rows removed due to NaN values: {len(nan_rows)}")
print(f"Rows removed due to out-of-bound values (0-200 spikes/sec): {rows_removed_by_filtering}")
print(f"Total rows effectively removed: {len(nan_rows) + rows_removed_by_filtering}")
print(f"Number of data points after cleaning: {cleaned_rows}")

if cleaned_rows < 0.5 * initial_rows:
    print("Warning: A significant portion of data was removed during cleaning. Results might be skewed.")
if cleaned_rows == 0:
    print("Error: All data was removed during cleaning. Cannot proceed with analysis.")
    exit()

# --- Descriptive Statistics (Cleaned) ---
print("\n--- Descriptive Statistics (Cleaned Data) ---")
print(data_cleaned[['region_a_firing_rate', 'region_b_firing_rate']].describe())

# --- Visualizations (Cleaned Data) ---
print(f"\n--- Generating Plots for Cleaned Data ---")
# Time series plot
plt.figure(figsize=(15, 6))
plt.plot(data_cleaned['time_seconds'], data_cleaned['region_a_firing_rate'], label='Region A Firing Rate (Cleaned)', alpha=0.7)
plt.plot(data_cleaned['time_seconds'], data_cleaned['region_b_firing_rate'], label='Region B Firing Rate (Cleaned)', alpha=0.7)
plt.xlabel('Time (seconds)')
plt.ylabel('Firing Rate (spikes/second)')
plt.title('Neural Firing Rates Over Time (Cleaned Data)')
plt.legend()
plt.grid(True)
plt.savefig(CLEANED_TIMESERIES_PLOT_FILE)
plt.close()
print(f"Saved cleaned time series plot to {CLEANED_TIMESERIES_PLOT_FILE}")

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='region_a_firing_rate', y='region_b_firing_rate', data=data_cleaned, alpha=0.5)
plt.title('Scatter Plot of Firing Rates (Region A vs Region B - Cleaned Data)')
plt.xlabel('Region A Firing Rate (spikes/second)')
plt.ylabel('Region B Firing Rate (spikes/second)')
plt.grid(True)
plt.savefig(CLEANED_SCATTER_PLOT_FILE)
plt.close()
print(f"Saved cleaned scatter plot to {CLEANED_SCATTER_PLOT_FILE}")


# --- Correlation Analysis (Cleaned Data) ---
print("\n--- Correlation Analysis (Cleaned Data) ---")
if cleaned_rows > 1: # Pearson R needs at least 2 data points
    corr, p_value = pearsonr(data_cleaned['region_a_firing_rate'], data_cleaned['region_b_firing_rate'])
    print(f"Pearson Correlation Coefficient between Region A and Region B: {corr:.4f}")
    print(f"P-value: {p_value:.4g}") # Using general format for p-value

    alpha = 0.05
    if p_value < alpha:
        print("The correlation is statistically significant (p < 0.05).")
    else:
        print("The correlation is not statistically significant (p >= 0.05).")
else:
    print("Not enough data points to calculate correlation after cleaning.")

print("\n--- Script Finished ---")