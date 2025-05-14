import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load data
attention = pd.read_csv("data/attention.csv")
synchrony = pd.read_csv("data/neural_synchrony.csv")
data = pd.merge(attention, synchrony, on="time")

# 1. Plot time series of attention and summary statistics
plt.figure(figsize=(12, 3))
plt.plot(data["time"], data["attention_score"])
plt.xlabel("Time (seconds)")
plt.ylabel("Attention Score")
plt.title("Time Series of Attention Score")
plt.tight_layout()
plt.savefig("attention_timeseries.png")
plt.close()

# 2. Compute correlations between each synchrony feature and attention
sync_cols = [col for col in data.columns if col.startswith("sync_")]
correlations = {}
pvals = {}
for col in sync_cols:
    r, p = pearsonr(data[col], data["attention_score"])
    correlations[col] = r
    pvals[col] = p

# 3. Plot heatmap of correlations (16x16 symmetric matrix)
corr_matrix = np.zeros((16, 16)) * np.nan
pval_matrix = np.zeros((16, 16)) * np.nan

for col in sync_cols:
    i, j = col.split("_")[1:]
    i, j = int(i)-1, int(j)-1  # zero-indexed
    corr_matrix[i, j] = correlations[col]
    pval_matrix[i, j] = pvals[col]

# Attention correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, vmin=-1, vmax=1,
            xticklabels=[str(i+1) for i in range(16)],
            yticklabels=[str(i+1) for i in range(16)])
plt.title("Pearson Correlation: Region-Pair Synchrony vs. Attention")
plt.xlabel("Region")
plt.ylabel("Region")
plt.tight_layout()
plt.savefig("sync_attention_corr_heatmap.png")
plt.close()

# 4. List the top positive and negative correlations
sorted_corrs = sorted([(col, correlations[col], pvals[col]) for col in sync_cols], key=lambda x: abs(x[1]), reverse=True)
with open("top_sync_correlations.txt", "w") as f:
    f.write("Region Pair\tCorrelation\tp-value\n")
    for col, r, p in sorted_corrs[:10]:
        f.write(f"{col}\t{r:.3f}\t{p:.2e}\n")
    f.write("\n")
    for col, r, p in sorted_corrs[-10:]:
        f.write(f"{col}\t{r:.3f}\t{p:.2e}\n")

# 5. Save list of significantly correlated region pairs after Bonferroni correction
alpha = 0.05 / len(sync_cols)
significant = [(col, correlations[col], pvals[col]) for col in sync_cols if pvals[col] < alpha]
with open("significant_sync_pairs.txt", "w") as f:
    f.write(f"Bonferroni threshold for significance: {alpha:.2e}\n")
    f.write("Region Pair\tCorrelation\tp-value\n")
    for col, r, p in significant:
        f.write(f"{col}\t{r:.3f}\t{p:.2e}\n")

# 6. Plot histogram of attention scores
plt.figure(figsize=(7, 4))
plt.hist(data["attention_score"], bins=30, color="dodgerblue", alpha=0.7)
plt.xlabel("Attention Score")
plt.ylabel("Count")
plt.title("Distribution of Attention Scores")
plt.tight_layout()
plt.savefig("attention_histogram.png")
plt.close()