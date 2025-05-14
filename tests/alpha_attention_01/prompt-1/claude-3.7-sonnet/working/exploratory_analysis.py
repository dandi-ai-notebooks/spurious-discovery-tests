#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Create output directory for figures
if not os.path.exists('figures'):
    os.makedirs('figures')

# Load datasets
print("Loading data...")
attention_df = pd.read_csv("data/attention.csv")
synchrony_df = pd.read_csv("data/neural_synchrony.csv")

# Basic info about the datasets
print("\nAttention data shape:", attention_df.shape)
print("Attention data columns:", attention_df.columns.tolist())
print("\nSynchrony data shape:", synchrony_df.shape)
print("Number of synchrony columns:", len(synchrony_df.columns) - 1)  # -1 for time column

# Basic statistics for attention data
print("\nAttention score statistics:")
print(attention_df['attention_score'].describe())

# Plot attention score over time
plt.figure(figsize=(12, 5))
plt.plot(attention_df['time'], attention_df['attention_score'])
plt.title('Attention Score Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Attention Score')
plt.grid(True, alpha=0.3)
plt.savefig('figures/attention_over_time.png', dpi=300, bbox_inches='tight')
plt.close()

# Identify all synchrony columns (pairs)
sync_columns = [col for col in synchrony_df.columns if col.startswith('sync_')]
print(f"\nTotal number of synchrony pairs: {len(sync_columns)}")

# Calculate basic statistics for all synchrony pairs
sync_stats = synchrony_df[sync_columns].describe().T
sync_stats['range'] = sync_stats['max'] - sync_stats['min']
sync_stats = sync_stats.sort_values('mean', ascending=False)

# Plot distributions of top 5 and bottom 5 synchrony pairs by mean
top_pairs = sync_stats.head(5).index
bottom_pairs = sync_stats.tail(5).index

plt.figure(figsize=(12, 6))
for col in top_pairs:
    sns.kdeplot(synchrony_df[col], label=col)
plt.title('Distribution of Top 5 Synchrony Pairs (by mean)')
plt.xlabel('Synchrony Value')
plt.ylabel('Density')
plt.legend()
plt.savefig('figures/top5_synchrony_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
for col in bottom_pairs:
    sns.kdeplot(synchrony_df[col], label=col)
plt.title('Distribution of Bottom 5 Synchrony Pairs (by mean)')
plt.xlabel('Synchrony Value')
plt.ylabel('Density')
plt.legend()
plt.savefig('figures/bottom5_synchrony_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# Merge datasets for correlation analysis
print("\nMerging datasets...")
merged_df = pd.merge(attention_df, synchrony_df, on='time')

# Calculate correlations between all synchrony pairs and attention
print("\nCalculating correlations...")
correlations = []
for col in sync_columns:
    corr, p_value = stats.pearsonr(merged_df['attention_score'], merged_df[col])
    correlations.append({'pair': col, 'correlation': corr, 'p_value': p_value})

corr_df = pd.DataFrame(correlations)
corr_df['significant'] = corr_df['p_value'] < 0.05
corr_df = corr_df.sort_values('correlation', ascending=False)

# Save correlation results
corr_df.to_csv('correlation_results.csv', index=False)

# Plot top 10 most correlated sync pairs with attention
top_corr = corr_df.head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='correlation', y='pair', data=top_corr)
plt.title('Top 10 Synchrony Pairs Correlated with Attention')
plt.xlabel('Pearson Correlation')
plt.ylabel('Synchrony Pair')
plt.grid(True, alpha=0.3)
plt.savefig('figures/top10_correlations.png', dpi=300, bbox_inches='tight')
plt.close()

# Examine the time series of the top correlated sync pair with attention
top_pair = corr_df.iloc[0]['pair']
plt.figure(figsize=(12, 6))
plt.plot(merged_df['time'], merged_df['attention_score'], label='Attention Score')
plt.plot(merged_df['time'], merged_df[top_pair], label=f'{top_pair}')
plt.title(f'Attention Score vs Top Correlated Synchrony Pair ({top_pair})')
plt.xlabel('Time (seconds)')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('figures/attention_vs_top_sync.png', dpi=300, bbox_inches='tight')
plt.close()

# Count significant correlations
n_significant = corr_df['significant'].sum()
print(f"\nNumber of significantly correlated pairs: {n_significant} out of {len(sync_columns)}")

print("\nExploratory analysis complete!")