import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

# Load data
print("Loading data...")
attention_df = pd.read_csv("data/attention.csv")
synchrony_df = pd.read_csv("data/neural_synchrony.csv")

# Create output directory if it doesn't exist
import os
os.makedirs('figures', exist_ok=True)

# 1. Time series plot of attention
plt.figure(figsize=(12, 4))
plt.plot(attention_df['time'], attention_df['attention_score'])
plt.title('Attention Score over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Attention Score')
plt.savefig('figures/attention_timeseries.png')
plt.close()

# 2. Calculate correlation matrix for all synchrony pairs with attention
sync_columns = [col for col in synchrony_df.columns if col.startswith('sync')]
correlations = []
p_values = []

for col in sync_columns:
    corr, p = stats.pearsonr(synchrony_df[col], attention_df['attention_score'])
    correlations.append(corr)
    p_values.append(p)

# Create correlation results DataFrame
correlation_results = pd.DataFrame({
    'sync_pair': sync_columns,
    'correlation': correlations,
    'p_value': p_values
})

# Apply Bonferroni correction
correlation_results['significant'] = correlation_results['p_value'] < (0.05 / len(sync_columns))

# Sort by absolute correlation
correlation_results = correlation_results.iloc[np.abs(correlation_results['correlation']).argsort()[::-1]]

# Save top correlations
np.savetxt('figures/correlations.txt', 
           correlation_results.values,
           fmt='%s',
           header='\t'.join(correlation_results.columns),
           comments='')

# 3. Plot distribution of correlations
plt.figure(figsize=(10, 6))
plt.hist(correlations, bins=30)
plt.title('Distribution of Synchrony-Attention Correlations')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Count')
plt.savefig('figures/correlation_distribution.png')
plt.close()

# 4. PCA on synchrony data
print("Performing PCA...")
scaler = StandardScaler()
sync_scaled = scaler.fit_transform(synchrony_df[sync_columns])
pca = PCA()
pca_result = pca.fit_transform(sync_scaled)

# Plot variance explained
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.savefig('figures/pca_variance.png')
plt.close()

# 5. Analyze temporal structure
# Calculate autocorrelation of attention score
max_lag = 60  # 1 minute
autocorr = [1.] + [stats.pearsonr(attention_df['attention_score'][:-i], 
                                 attention_df['attention_score'][i:])[0]
                   for i in range(1, max_lag)]

plt.figure(figsize=(10, 6))
plt.plot(range(max_lag), autocorr)
plt.title('Attention Score Autocorrelation')
plt.xlabel('Lag (seconds)')
plt.ylabel('Autocorrelation')
plt.savefig('figures/attention_autocorrelation.png')
plt.close()

print("Analysis complete!")