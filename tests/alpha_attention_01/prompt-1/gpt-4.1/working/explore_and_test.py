import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, zscore
from statsmodels.api import OLS, add_constant

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Load data
attention = pd.read_csv("data/attention.csv")
sync = pd.read_csv("data/neural_synchrony.csv")
df = pd.merge(attention, sync, on="time")

# --- 1. Descriptive statistics and distribution plots ---

# Plot distribution of attention scores
plt.figure(figsize=(6,4))
sns.histplot(df["attention_score"], bins=50, kde=True, color="skyblue")
plt.title("Distribution of Attention Scores")
plt.xlabel("Attention Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("attention_distribution.png")
plt.close()

# Plot mean and std for synchrony columns
sync_cols = [col for col in df.columns if col.startswith("sync_")]
sync_means = df[sync_cols].mean()
sync_stds = df[sync_cols].std()

plt.figure(figsize=(12,5))
plt.plot(sync_means.values, label='Mean synchrony')
plt.fill_between(range(len(sync_means)), sync_means - sync_stds, sync_means + sync_stds, color='gray', alpha=0.2, label='±1 std')
plt.title("Mean and Std of Synchrony Across Region Pairs")
plt.xlabel("Region Pair Index")
plt.ylabel("Synchrony (coherence)")
plt.legend()
plt.tight_layout()
plt.savefig("synchrony_means_std.png")
plt.close()

# --- 2. Correlation between synchrony and attention ---

pair_corrs = {}
for col in sync_cols:
    r, p = pearsonr(df[col], df["attention_score"])
    pair_corrs[col] = (r, p)

# Create correlation dataframe and heatmap
corr_vals = np.zeros((16,16))
pval_mask = np.ones((16,16), dtype=bool)

for col, (r, p) in pair_corrs.items():
    _, i, j = col.split('_')
    i = int(i)
    j = int(j)
    corr_vals[i-1, j-1] = r
    corr_vals[j-1, i-1] = r
    if p < 0.05:
        pval_mask[i-1, j-1] = False
        pval_mask[j-1, i-1] = False

plt.figure(figsize=(10,8))
sns.heatmap(corr_vals, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, mask=np.eye(16))
plt.title("Pearson Correlation between Synchrony (pairs) and Attention Score")
plt.xlabel("Region")
plt.ylabel("Region")
plt.tight_layout()
plt.savefig("pairwise_attention_correlations.png")
plt.close()

# --- 3. Regression: Predicting attention from all synchrony ---

# Z-score predictors and target
X = df[sync_cols].apply(zscore)
y = zscore(df["attention_score"])
X = add_constant(X)
reg = OLS(y, X).fit()

# Save top predictors
summary_df = pd.DataFrame({
    'pair': sync_cols,
    'beta': reg.params[1:],  # exclude intercept
    'pval': reg.pvalues[1:]
})
summary_df = summary_df.reindex(summary_df['beta'].abs().sort_values(ascending=False).index)
summary_df.to_csv("regression_coefficients.csv", index=False)

# Plot top 10 absolute beta values
top10 = summary_df.iloc[:10]
plt.figure(figsize=(8,5))
sns.barplot(x="beta", y="pair", data=top10, palette="Blues_d")
plt.title("Top 10 Region-Pair Synchrony Predictors of Attention (Regression Beta)")
plt.xlabel("Standardized Beta")
plt.ylabel("Synchrony Pair")
plt.tight_layout()
plt.savefig("top10_predictors.png")
plt.close()

# --- 4. Permutation test for strongest pair ---

# Identify pair with highest absolute correlation
best_pair = summary_df.iloc[0]['pair']
observed_corr = np.abs(pearsonr(df[best_pair], df["attention_score"])[0])
n_perms = 1000
permuted_corrs = []
for _ in range(n_perms):
    y_shuffled = np.random.permutation(df["attention_score"])
    r_perm = np.abs(pearsonr(df[best_pair], y_shuffled)[0])
    permuted_corrs.append(r_perm)

p_perm = np.mean(np.array(permuted_corrs) >= observed_corr)

# Plot permutation distribution
plt.figure(figsize=(6,4))
sns.histplot(permuted_corrs, bins=40, color="orange", label="Permuted")
plt.axvline(observed_corr, color='red', linestyle='--', label="Observed |r|")
plt.title(f"Permutation Test for {best_pair} - Attention Correlation\np={p_perm:.4f}")
plt.xlabel("Absolute Correlation (|r|)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("permutation_best_pair.png")
plt.close()

# --- 5. Save findings as summary text ---
with open("findings.txt", "w") as f:
    f.write("Top correlated pair: {}\n".format(best_pair))
    f.write("Observed correlation: {:.3f}\n".format(observed_corr))
    f.write("Permutation p-value: {:.4g}\n\n".format(p_perm))
    f.write(reg.summary().as_text())
    f.write("\n\nSee regression_coefficients.csv for all beta values.")

print("Analysis complete. Figures, tables, and summary saved.")