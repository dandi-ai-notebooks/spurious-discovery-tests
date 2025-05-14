import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Create output directory for plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load working-memory load trace
print("Loading memory load data...")
wm_df = pd.read_csv("data/memory_load.csv")

# Load theta power data
print("Loading theta power data...")
theta_df = pd.read_csv("data/theta_power.csv")

# Basic info about the datasets
print("\n--- Memory Load Data Info ---")
print(f"Shape: {wm_df.shape}")
print(wm_df.head())
print(wm_df.describe())

print("\n--- Theta Power Data Info ---")
print(f"Shape: {theta_df.shape}")
print(theta_df.head())
print(theta_df.describe())

# Get electrode names (excluding 'time' column)
electrodes = [col for col in theta_df.columns if col != 'time']
print(f"\nElectrodes: {electrodes}")

# Merge datasets on time for joint analysis
print("\nMerging datasets...")
merged_df = wm_df.merge(theta_df, on="time")
print(f"Merged shape: {merged_df.shape}")
print(merged_df.head())

# Plot working memory load over time
plt.figure(figsize=(15, 5))
plt.plot(wm_df['time'], wm_df['wm_load'])
plt.title('Working Memory Load Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('WM Load (0-1)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/wm_load_time_series.png')

# Plot mean theta power across all electrodes
theta_df['mean_theta'] = theta_df[electrodes].mean(axis=1)
plt.figure(figsize=(15, 5))
plt.plot(theta_df['time'], theta_df['mean_theta'])
plt.title('Mean Theta Power Over Time (All Electrodes)')
plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Theta Power (0-1)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/mean_theta_time_series.png')

# Plot theta power for each electrode (first 300 seconds for clarity)
time_limit = 300  # First 5 minutes
plt.figure(figsize=(15, 10))
for electrode in electrodes:
    plt.plot(theta_df.loc[theta_df['time'] <= time_limit, 'time'], 
             theta_df.loc[theta_df['time'] <= time_limit, electrode], 
             label=electrode)
plt.title('Theta Power by Electrode (First 5 Minutes)')
plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Theta Power (0-1)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/electrode_theta_time_series.png')

# Create correlation heatmap between theta power at different electrodes
plt.figure(figsize=(12, 10))
correlation_matrix = theta_df[electrodes].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Between Theta Power at Different Electrodes')
plt.tight_layout()
plt.savefig('plots/electrode_correlation_heatmap.png')

# Correlation between WM load and theta power
correlations = []
for electrode in electrodes:
    r, p = stats.pearsonr(merged_df['wm_load'], merged_df[electrode])
    correlations.append({
        'electrode': electrode,
        'correlation': r,
        'p_value': p
    })

corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values(by='correlation', ascending=False)
print("\n--- Correlations between WM Load and Theta Power ---")
print(corr_df)

# Plot correlation bar chart
plt.figure(figsize=(12, 6))
colors = ['green' if p < 0.05 else 'gray' for p in corr_df['p_value']]
plt.bar(corr_df['electrode'], corr_df['correlation'], color=colors)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title('Correlation Between WM Load and Theta Power by Electrode')
plt.xlabel('Electrode')
plt.ylabel('Pearson Correlation')
plt.xticks(rotation=45)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/wm_theta_correlation.png')

# Save correlation results to csv
corr_df.to_csv('plots/wm_theta_correlations.csv', index=False)

print("Exploratory analysis complete. Plots saved to 'plots' directory.")