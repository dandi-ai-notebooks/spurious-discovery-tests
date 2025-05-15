import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load datasets
attention_df = pd.read_csv("data/attention.csv")
synchrony_df = pd.read_csv("data/neural_synchrony.csv")

# Merge on time
df = pd.merge(attention_df, synchrony_df, on="time")

# Compute correlations and p-values
results = []
for col in synchrony_df.columns:
    if col != "time":
        r, p = pearsonr(df["attention_score"], df[col])
        results.append({"region_pair": col, "correlation": r, "p_value": p})

results_df = pd.DataFrame(results)
results_df.to_csv("correlation_results.csv", index=False)

# Apply Bonferroni correction for multiple comparisons
alpha = 0.05
corrected_alpha = alpha / len(results_df)
significant = results_df[results_df["p_value"] < corrected_alpha]

# Plot top significant correlations
top_sig = significant.sort_values(by="correlation", key=abs, ascending=False).head(5)
for _, row in top_sig.iterrows():
    plt.figure(figsize=(10, 4))
    sns.scatterplot(x=df[row["region_pair"]], y=df["attention_score"], alpha=0.5)
    sns.regplot(x=df[row["region_pair"]], y=df["attention_score"], scatter=False, color='red')
    plt.xlabel(row["region_pair"])
    plt.ylabel("Attention Score")
    plt.title(f"{row['region_pair']} vs Attention (r={row['correlation']:.2f}, p={row['p_value']:.1e})")
    plt.tight_layout()
    plt.savefig(f"{row['region_pair']}_vs_attention.png")
    plt.close()