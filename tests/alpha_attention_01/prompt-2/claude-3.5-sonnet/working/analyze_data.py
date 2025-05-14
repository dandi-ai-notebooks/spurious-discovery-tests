import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

# Load data
attention_df = pd.read_csv("data/attention.csv")
synchrony_df = pd.read_csv("data/neural_synchrony.csv")

# Basic statistics for attention scores
att_stats = attention_df['attention_score'].describe()
print("\nAttention Score Statistics:")
print(att_stats)

# Plot attention time series
plt.figure(figsize=(12, 4))
plt.plot(attention_df['time'], attention_df['attention_score'])
plt.xlabel('Time (seconds)')
plt.ylabel('Attention Score')
plt.title('Attention Score Over Time')
plt.savefig('plots/attention_timeseries.png')
plt.close()

# Get all synchrony columns
sync_cols = [col for col in synchrony_df.columns if col.startswith('sync_')]

# Calculate mean synchrony across all pairs
synchrony_df['mean_sync'] = synchrony_df[sync_cols].mean(axis=1)

# Plot mean synchrony over time
plt.figure(figsize=(12, 4))
plt.plot(synchrony_df['time'], synchrony_df['mean_sync'])
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Synchrony')
plt.title('Mean Neural Synchrony Over Time')
plt.savefig('plots/mean_synchrony_timeseries.png')
plt.close()

# Merge attention and synchrony data
merged_df = pd.merge(attention_df, synchrony_df, on='time')

# Calculate correlations between attention and each synchrony pair
correlations = []
for col in sync_cols:
    corr, p = stats.pearsonr(merged_df['attention_score'], merged_df[col])
    correlations.append({
        'pair': col,
        'correlation': corr,
        'p_value': p
    })

corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('correlation', ascending=False)

# Bonferroni correction for multiple comparisons
n_tests = len(sync_cols)
alpha = 0.05
bonferroni_threshold = alpha / n_tests

# Add significance indicator
corr_df['significant'] = corr_df['p_value'] < bonferroni_threshold

# Plot top correlations
plt.figure(figsize=(12, 6))
plt.bar(range(len(corr_df)), corr_df['correlation'])
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xticks(range(len(corr_df)), corr_df['pair'], rotation=90)
plt.xlabel('Region Pair')
plt.ylabel('Correlation with Attention')
plt.title('Correlation between Regional Synchrony and Attention')
plt.tight_layout()
plt.savefig('plots/synchrony_attention_correlations.png')
plt.close()

# Save correlation results
corr_df.to_csv('correlation_results.csv', index=False)

# Calculate moving window correlations to examine temporal dynamics
window_size = 60  # 1-minute windows
correlation_over_time = []

for start_time in range(0, max(merged_df['time']) - window_size, window_size):
    window_data = merged_df[(merged_df['time'] >= start_time) & 
                          (merged_df['time'] < start_time + window_size)]
    
    window_corr = stats.pearsonr(window_data['attention_score'], 
                                window_data['mean_sync'])[0]
    
    correlation_over_time.append({
        'start_time': start_time,
        'correlation': window_corr
    })

corr_time_df = pd.DataFrame(correlation_over_time)

# Plot temporal evolution of correlation
plt.figure(figsize=(12, 4))
plt.plot(corr_time_df['start_time'], corr_time_df['correlation'])
plt.xlabel('Time (seconds)')
plt.ylabel('Correlation (1-min windows)')
plt.title('Temporal Evolution of Attention-Synchrony Correlation')
plt.savefig('plots/correlation_temporal_evolution.png')
plt.close()

# Print summary statistics for writing to report
print("\nSummary of findings:")
print(f"Number of significant correlations: {sum(corr_df['significant'])}")
print(f"Strongest correlation: {corr_df.iloc[0]['pair']} (r={corr_df.iloc[0]['correlation']:.3f}, p={corr_df.iloc[0]['p_value']:.6f})")
print(f"Mean correlation across all pairs: {corr_df['correlation'].mean():.3f}")