import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Load data
attention = pd.read_csv("data/attention.csv")
synchrony = pd.read_csv("data/neural_synchrony.csv")

# Merge on time column
data = pd.merge(attention, synchrony, on="time")

# Basic statistics
print("Attention score statistics:")
print(data["attention_score"].describe())

# Correlation analysis
correlations = []
sync_columns = [col for col in data.columns if col.startswith("sync_")]

for sync_col in sync_columns:
    r, p = stats.pearsonr(data["attention_score"], data[sync_col])
    correlations.append({
        "pair": sync_col,
        "correlation": r,
        "p_value": p
    })

corr_df = pd.DataFrame(correlations)

# Get top 10 strongest correlations (absolute value)
top_corrs = corr_df.iloc[np.argsort(-abs(corr_df["correlation"]))].head(10)

# Plot attention vs top correlated sync pairs
plt.figure(figsize=(12, 8))
for i, row in top_corrs.iterrows():
    sns.regplot(x=row["pair"], y="attention_score", data=data, 
                scatter_kws={"alpha": 0.1}, label=f"{row['pair']} (r={row['correlation']:.2f})")
plt.title("Attention Score vs Top Neural Synchrony Correlations")
plt.ylabel("Attention Score")
plt.xlabel("Neural Synchrony")
plt.legend()
plt.tight_layout()
plt.savefig("top_correlations.png")

# Plot distribution of all correlations
plt.figure(figsize=(8, 5))
sns.histplot(corr_df["correlation"], bins=30)
plt.title("Distribution of Correlation Coefficients")
plt.xlabel("Pearson r")
plt.savefig("correlation_distribution.png")

# Save correlation results
corr_df.to_csv("correlation_results.csv", index=False)