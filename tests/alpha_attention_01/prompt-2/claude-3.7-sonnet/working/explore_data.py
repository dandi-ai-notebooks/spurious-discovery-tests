import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up output directory for figures
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# Load the datasets
print("Loading attention data...")
attention_df = pd.read_csv("data/attention.csv")
print("Loading neural synchrony data...")
synchrony_df = pd.read_csv("data/neural_synchrony.csv")

# Basic information about the datasets
print("\nAttention dataset info:")
print(f"Shape: {attention_df.shape}")
print(f"First few rows:\n{attention_df.head()}")
print(f"Missing values:\n{attention_df.isnull().sum()}")
print(f"Summary statistics:\n{attention_df.describe()}")

print("\nNeural synchrony dataset info:")
print(f"Shape: {synchrony_df.shape}")
print(f"First few rows:\n{synchrony_df.head()}")
print(f"Columns: {synchrony_df.columns.tolist()}")
print(f"Missing values (total): {synchrony_df.isnull().sum().sum()}")

# Plot the distribution of attention scores
plt.figure(figsize=(10, 6))
sns.histplot(attention_df['attention_score'], kde=True)
plt.title("Distribution of Attention Scores")
plt.xlabel("Attention Score")
plt.ylabel("Frequency")
plt.savefig("figures/attention_score_distribution.png")

# Plot attention over time
plt.figure(figsize=(12, 6))
plt.plot(attention_df['time'], attention_df['attention_score'])
plt.title("Attention Score Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Attention Score")
plt.savefig("figures/attention_over_time.png")

# Get the synchrony column names
synchrony_columns = [col for col in synchrony_df.columns if col.startswith('sync_')]
print(f"\nNumber of synchrony pairs: {len(synchrony_columns)}")

# Plot distributions of a few random synchrony pairs
plt.figure(figsize=(12, 8))
sample_cols = np.random.choice(synchrony_columns, 6, replace=False)
for i, col in enumerate(sample_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(synchrony_df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel("Synchrony Value")
plt.tight_layout()
plt.savefig("figures/sample_synchrony_distributions.png")

# Compute the average synchrony across all pairs over time
synchrony_df['mean_synchrony'] = synchrony_df[synchrony_columns].mean(axis=1)

# Plot mean synchrony over time
plt.figure(figsize=(12, 6))
plt.plot(synchrony_df['time'], synchrony_df['mean_synchrony'])
plt.title("Mean Synchrony Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Mean Synchrony")
plt.savefig("figures/mean_synchrony_over_time.png")

# Merge the datasets on time column
merged_df = pd.merge(attention_df, synchrony_df, on='time')
print(f"\nMerged dataset shape: {merged_df.shape}")

# Calculate the correlation between attention score and each synchrony pair
correlations = []
for col in synchrony_columns:
    corr = merged_df['attention_score'].corr(merged_df[col])
    correlations.append((col, corr))

# Sort by absolute correlation values
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nTop 10 synchrony pairs with strongest correlation with attention:")
for pair, corr in correlations[:10]:
    print(f"{pair}: {corr:.4f}")

# Plot the correlation between attention and top 5 synchrony pairs
top_pairs = [pair for pair, _ in correlations[:5]]
plt.figure(figsize=(12, 8))
for i, pair in enumerate(top_pairs, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=merged_df[pair], y=merged_df['attention_score'], alpha=0.3)
    plt.title(f"Attention vs {pair}\nCorr: {merged_df['attention_score'].corr(merged_df[pair]):.4f}")
    plt.xlabel(pair)
    plt.ylabel("Attention Score")
plt.tight_layout()
plt.savefig("figures/top_correlations_scatter.png")

print("\nExploration completed. Figures saved in the 'figures' directory.")