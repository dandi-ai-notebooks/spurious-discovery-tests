import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Create output directory for figures
Path('figures').mkdir(exist_ok=True)

# Load data
attention_df = pd.read_csv('attention.csv')
synchrony_df = pd.read_csv('neural_synchrony.csv')

# Get list of sync columns
sync_cols = [col for col in synchrony_df.columns if col.startswith('sync_')]

# Calculate correlation between each sync pair and attention
correlations = {}
p_values = {}
for col in sync_cols:
    r, p = stats.pearsonr(synchrony_df[col], attention_df['attention_score'])
    correlations[col] = r
    p_values[col] = p

# Create correlation matrix visualization
correlation_df = pd.DataFrame(columns=['region_1', 'region_2', 'correlation', 'p_value'])
for col in sync_cols:
    r1, r2 = map(int, col.split('_')[1:])
    correlation_df = pd.concat([correlation_df, pd.DataFrame({
        'region_1': [r1],
        'region_2': [r2],
        'correlation': [correlations[col]],
        'p_value': [p_values[col]]
    })])

# Create correlation matrix
n_regions = 16
corr_matrix = np.zeros((n_regions, n_regions))
for idx, row in correlation_df.iterrows():
    i, j = int(row['region_1']) - 1, int(row['region_2']) - 1
    corr_matrix[i, j] = row['correlation']
    corr_matrix[j, i] = row['correlation']

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, 
            xticklabels=range(1, n_regions + 1),
            yticklabels=range(1, n_regions + 1))
plt.title('Neural Synchrony-Attention Correlation Matrix')
plt.xlabel('Brain Region')
plt.ylabel('Brain Region')
plt.tight_layout()
plt.savefig('figures/correlation_matrix.png')
plt.close()

# Find top 5 most strongly correlated pairs
top_pairs = correlation_df.nlargest(5, 'correlation')

# Plot time series for top pair
top_pair = top_pairs.iloc[0]
top_sync_col = f"sync_{int(top_pair['region_1'])}_{int(top_pair['region_2'])}"

plt.figure(figsize=(12, 6))
plt.plot(synchrony_df['time'], synchrony_df[top_sync_col], label='Neural Synchrony')
plt.plot(attention_df['time'], attention_df['attention_score'], label='Attention Score')
plt.title(f'Neural Synchrony vs Attention Score\nRegions {int(top_pair["region_1"])}-{int(top_pair["region_2"])}')
plt.xlabel('Time (s)')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.savefig('figures/top_pair_timeseries.png')
plt.close()

# Create scatter plot for top pair
plt.figure(figsize=(8, 8))
plt.scatter(synchrony_df[top_sync_col], attention_df['attention_score'], alpha=0.5)
plt.xlabel(f'Neural Synchrony (Regions {int(top_pair["region_1"])}-{int(top_pair["region_2"])})')
plt.ylabel('Attention Score')
plt.title('Neural Synchrony vs Attention Correlation')
plt.tight_layout()
plt.savefig('figures/correlation_scatter.png')
plt.close()

# Save statistical results
results = {
    'top_correlations': top_pairs.to_dict('records'),
    'significant_pairs': len(correlation_df[correlation_df['p_value'] < 0.05]),
    'total_pairs': len(correlation_df),
    'max_correlation': correlation_df['correlation'].max(),
    'min_correlation': correlation_df['correlation'].min(),
    'mean_correlation': correlation_df['correlation'].mean(),
}

# Save results as numpy array for future reference
np.save('analysis_results.npy', results)