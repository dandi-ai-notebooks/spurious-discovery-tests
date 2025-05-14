import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load data
attention_df = pd.read_csv("data/attention.csv")
synchrony_df = pd.read_csv("data/neural_synchrony.csv")

# Merge data
merged_df = pd.merge(attention_df, synchrony_df, on="time")

# Calculate correlations
synchrony_cols = [col for col in merged_df.columns if col.startswith('sync_')]
correlations = {}
p_values = {}

print("Correlations with attention_score:")
for sync_col in synchrony_cols:
    if merged_df[sync_col].isnull().any() or merged_df['attention_score'].isnull().any():
        print(f"Skipping {sync_col} due to NaN values.")
        continue
    # Ensure there's variance in the data for correlation calculation
    if merged_df[sync_col].nunique() < 2 or merged_df['attention_score'].nunique() < 2:
        print(f"Skipping {sync_col} due to insufficient variance.")
        continue
    
    corr, p_val = pearsonr(merged_df['attention_score'], merged_df[sync_col])
    correlations[sync_col] = corr
    p_values[sync_col] = p_val
    print(f"{sync_col}: r = {corr:.4f}, p = {p_val:.4f}")

# Find the sync column with the highest absolute correlation
if correlations:
    max_corr_col = max(correlations, key=lambda k: abs(correlations[k]))
    max_corr_val = correlations[max_corr_col]
    max_corr_pval = p_values[max_corr_col]
    print(f"\nHighest absolute correlation with attention_score:")
    print(f"{max_corr_col}: r = {max_corr_val:.4f}, p = {max_corr_pval:.4f}")

    # Create scatter plot for the most correlated sync column
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=merged_df[max_corr_col], y=merged_df['attention_score'])
    plt.title(f'Attention Score vs. {max_corr_col}\n(r={max_corr_val:.2f}, p={max_corr_pval:.3f})')
    plt.xlabel(f'{max_corr_col} (Neural Synchrony)')
    plt.ylabel('Attention Score')
    plt.grid(True)
    plt.savefig('attention_vs_max_sync.png')
    print("\nScatter plot saved to attention_vs_max_sync.png")
else:
    print("\nNo valid correlations found to plot.")

print("\nScript finished.")