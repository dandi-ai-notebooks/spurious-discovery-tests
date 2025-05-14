import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

# Load data
attention = pd.read_csv("data/attention.csv")
synchrony = pd.read_csv("data/neural_synchrony.csv")

# Merge on time
df = pd.merge(attention, synchrony, on="time")

# Compute correlation between attention_score and all sync features
sync_cols = [col for col in df.columns if col.startswith("sync_")]
correlations = []
p_values = []

for col in sync_cols:
    r, p = pearsonr(df["attention_score"], df[col])
    correlations.append((col, r))
    p_values.append(p)

# Adjust for multiple comparisons
reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

# Compile results
results = pd.DataFrame({
    "region_pair": [x[0] for x in correlations],
    "correlation_r": [x[1] for x in correlations],
    "p_value": p_values,
    "p_value_corrected": pvals_corrected,
    "significant": reject
})

# Save results
results.to_csv("correlation_results.csv", index=False)

# Plot top significant correlations (max 5)
top_results = results[results["significant"]].nlargest(5, "correlation_r")
for _, row in top_results.iterrows():
    region = row["region_pair"]
    plt.figure(figsize=(8, 6))
    sns.regplot(x=df[region], y=df["attention_score"], scatter_kws={'s': 10}, line_kws={"color": "red"})
    plt.title(f"{region} vs Attention Score\nr={row['correlation_r']:.2f}, p={row['p_value_corrected']:.2e}")
    plt.xlabel(region)
    plt.ylabel("Attention Score")
    plt.tight_layout()
    plt.savefig(f"{region}_vs_attention.png")
    plt.close()

print("Analysis complete. Results saved.")