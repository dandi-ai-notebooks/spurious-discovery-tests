import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np

# Load the datasets
try:
    attention_df = pd.read_csv("data/attention.csv")
    synchrony_df = pd.read_csv("data/neural_synchrony.csv")
except FileNotFoundError:
    print("Error: Data files not found. Make sure 'data/attention.csv' and 'data/neural_synchrony.csv' exist.")
    exit()

# Merge the dataframes on 'time'
merged_df = pd.merge(attention_df, synchrony_df, on="time")

# Identify synchrony columns
sync_columns = [col for col in merged_df.columns if col.startswith('sync_')]

if not sync_columns:
    print("Error: No synchrony columns (starting with 'sync_') found in the merged data.")
    exit()

# Calculate correlations
correlations = {}
for col in sync_columns:
    if merged_df[col].isnull().any() or merged_df['attention_score'].isnull().any():
        print(f"Warning: Skipping column {col} due to NaN values.")
        continue
    # Ensure there's enough variance, otherwise pearsonr can fail or give NaN
    if merged_df[col].nunique() < 2 or merged_df['attention_score'].nunique() < 2:
        print(f"Warning: Skipping column {col} due to insufficient variance.")
        continue
    
    corr, p_value = pearsonr(merged_df['attention_score'], merged_df[col])
    correlations[col] = {'correlation': corr, 'p_value': p_value}

if not correlations:
    print("Error: No valid correlations could be calculated. Check data integrity and variance.")
    exit()

# Sort by absolute correlation magnitude
sorted_correlations = sorted(correlations.items(), key=lambda item: abs(item[1]['correlation']), reverse=True)

print("Top 5 synchrony pairs correlated with attention_score:")
for i, (sync_pair, stats) in enumerate(sorted_correlations[:5]):
    print(f"{i+1}. {sync_pair}: Correlation = {stats['correlation']:.4f}, P-value = {stats['p_value']:.4g}")

if not sorted_correlations:
    print("No correlations found to plot.")
    exit()

# Get the most correlated pair
most_correlated_pair, most_correlated_stats = sorted_correlations[0]
print(f"\nMost correlated pair: {most_correlated_pair} (Correlation: {most_correlated_stats['correlation']:.4f}, P-value: {most_correlated_stats['p_value']:.4g})")

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid') # Using a style that's likely available

# 1. Time series plot of attention_score and the most correlated sync measure
plt.figure(figsize=(12, 6))
plt.plot(merged_df['time'], merged_df['attention_score'], label='Attention Score', color='blue', alpha=0.7)
plt.plot(merged_df['time'], merged_df[most_correlated_pair], label=f'{most_correlated_pair} (Most Correlated)', color='red', alpha=0.7)
plt.xlabel('Time (seconds)')
plt.ylabel('Value')
plt.title(f'Attention Score and Neural Synchrony ({most_correlated_pair}) Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('attention_sync_timeseries.png')
print("\nSaved 'attention_sync_timeseries.png'")

# 2. Scatter plot for the most correlated pair
plt.figure(figsize=(8, 6))
sns.regplot(x=merged_df[most_correlated_pair], y=merged_df['attention_score'], scatter_kws={'alpha':0.5})
plt.xlabel(f'{most_correlated_pair}')
plt.ylabel('Attention Score')
plt.title(f'Attention Score vs. {most_correlated_pair}\nCorrelation: {most_correlated_stats["correlation"]:.4f}, P-value: {most_correlated_stats["p_value"]:.2e}')
plt.tight_layout()
plt.savefig('attention_sync_scatter.png')
print("Saved 'attention_sync_scatter.png'")

print("\nScript finished.")