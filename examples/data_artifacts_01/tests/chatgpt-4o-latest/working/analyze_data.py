import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Load data
df = pd.read_csv("data/neural_firing_rates.csv")

# Drop rows with NaNs or clearly corrupted data
clean_df = df.dropna()

# Descriptive statistics
desc_stats = clean_df[['region_a_firing_rate', 'region_b_firing_rate']].describe()
desc_stats.to_csv("descriptive_statistics.csv")

# Correlation full trace
pearson_corr, pearson_p = pearsonr(clean_df['region_a_firing_rate'], clean_df['region_b_firing_rate'])
spearman_corr, spearman_p = spearmanr(clean_df['region_a_firing_rate'], clean_df['region_b_firing_rate'])

with open("correlation_results.txt", "w") as f:
    f.write(f"Pearson correlation: {pearson_corr:.4f}, p-value: {pearson_p:.4e}\\n")
    f.write(f"Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_p:.4e}\\n")

# Plot time series (first 1000 seconds for clarity)
plt.figure(figsize=(12, 5))
plt.plot(clean_df["time_seconds"][:1000], clean_df["region_a_firing_rate"][:1000], label="Region A")
plt.plot(clean_df["time_seconds"][:1000], clean_df["region_b_firing_rate"][:1000], label="Region B")
plt.title("Firing Rates over Time (First 1000 seconds)")
plt.xlabel("Time (seconds)")
plt.ylabel("Firing Rate (spikes/sec)")
plt.legend()
plt.tight_layout()
plt.savefig("firing_rates_timeseries.png")
plt.close()

# Scatter plot
sns.jointplot(data=clean_df, x="region_a_firing_rate", y="region_b_firing_rate", kind="hex", color="purple")
plt.savefig("firing_rate_scatter.png")