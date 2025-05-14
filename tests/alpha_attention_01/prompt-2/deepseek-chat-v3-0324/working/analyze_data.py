import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Load data
attention = pd.read_csv('data/attention.csv')
synchrony = pd.read_csv('data/neural_synchrony.csv')

# Merge data on time column
data = pd.merge(attention, synchrony, on='time')

print("Data Overview:")
print(f"Time points: {len(data)}")
print(f"Attention range: {data['attention_score'].min():.4f} to {data['attention_score'].max():.4f}")

# Check if attention needs scaling
if data['attention_score'].max() < 0.01:
    print("\nWarning: Attention scores are much smaller than expected 0-1 range")

# Calculate correlations between attention and each sync pair
sync_cols = [col for col in data.columns if col.startswith('sync_')]
correlations = []

for col in sync_cols:
    r, p = stats.pearsonr(data['attention_score'], data[col])
    correlations.append({
        'region_pair': col,
        'correlation': r,
        'p_value': p
    })

corr_df = pd.DataFrame(correlations)

# Get top 5 most correlated pairs
top_corr = corr_df.sort_values('correlation', key=abs, ascending=False).head(5)

print("\nTop Correlated Region Pairs:")
print(top_corr[['region_pair', 'correlation', 'p_value']])

# Generate plots
plt.figure(figsize=(12, 6))

# Plot attention distribution
plt.subplot(1, 2, 1)
sns.histplot(data['attention_score'], bins=50)
plt.title('Attention Score Distribution')

# Plot example synchrony vs attention
example_pair = top_corr.iloc[0]['region_pair']
plt.subplot(1, 2, 2)
sns.scatterplot(x=data[example_pair], y=data['attention_score'])
plt.xlabel(f'Synchrony ({example_pair})')
plt.ylabel('Attention Score')
plt.title(f'Attention vs {example_pair}')

plt.tight_layout()
plt.savefig('attention_vs_sync.png')
plt.close()