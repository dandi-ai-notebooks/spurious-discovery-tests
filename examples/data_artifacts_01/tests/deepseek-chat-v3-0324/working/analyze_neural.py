import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Load data
print("Loading data...")
df = pd.read_csv('data/neural_firing_rates.csv')

# Basic statistics
print("\nBasic Statistics:")
stats_a = df['region_a_firing_rate'].describe()
stats_b = df['region_b_firing_rate'].describe()
print("\nRegion A:")
print(stats_a)
print("\nRegion B:")
print(stats_b)

# Data quality check
print("\nData Quality:")
na_a = df['region_a_firing_rate'].isna().sum()
na_b = df['region_b_firing_rate'].isna().sum()
print(f"Missing values in Region A: {na_a}")
print(f"Missing values in Region B: {na_b}")

# Correlation analysis
print("\nCorrelation Analysis:")
corr, p_value = stats.pearsonr(
    df['region_a_firing_rate'].dropna(), 
    df['region_b_firing_rate'].dropna()
)
print(f"Pearson correlation: {corr:.3f} (p={p_value:.4g})")

# Time series visualization
plt.figure(figsize=(12, 6))
plt.plot(df['time_seconds'], df['region_a_firing_rate'], label='Region A', alpha=0.7)
plt.plot(df['time_seconds'], df['region_b_firing_rate'], label='Region B', alpha=0.7)
plt.xlabel('Time (seconds)')
plt.ylabel('Firing Rate (spikes/second)')
plt.title('Neural Firing Rates Over Time')
plt.legend()
plt.savefig('firing_rates_time_series.png')
plt.close()

# Scatter plot with regression
plt.figure(figsize=(10, 6))
sns.regplot(
    x='region_a_firing_rate',
    y='region_b_firing_rate',
    data=df,
    scatter_kws={'alpha': 0.1}
)
plt.xlabel('Region A Firing Rate')
plt.ylabel('Region B Firing Rate')
plt.title('Relationship Between Brain Regions')
plt.savefig('firing_rates_scatter.png')
plt.close()

print("\nAnalysis complete. Visualizations saved.")