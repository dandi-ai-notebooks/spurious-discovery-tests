import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Load data
data = pd.read_csv('data/tomato_health_data.csv')

# Basic exploration
print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

# Summary statistics
print("\nSummary statistics:")
print(data.describe())

# Correlation analysis
correlations = data.corr()['daily_tomato_consumption'].sort_values(ascending=False)
top_correlations = correlations[1:11]  # Top 10 positive correlations
bottom_correlations = correlations[-10:]  # Top 10 negative correlations

print("\nTop positive correlations with tomato consumption:")
print(top_correlations)
print("\nTop negative correlations with tomato consumption:")
print(bottom_correlations)

# Visualizations
os.makedirs('plots', exist_ok=True)

# Plot top correlations
plt.figure(figsize=(10, 6))
sns.barplot(x=top_correlations.values, y=top_correlations.index)
plt.title('Top Positive Correlations with Tomato Consumption')
plt.tight_layout()
plt.savefig('plots/top_correlations.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x=bottom_correlations.values, y=bottom_correlations.index)
plt.title('Top Negative Correlations with Tomato Consumption')
plt.tight_layout()
plt.savefig('plots/bottom_correlations.png')
plt.close()

# Scatter plots for top 3 correlations
for var in top_correlations.index[:3]:
    plt.figure(figsize=(8, 6))
    sns.regplot(x='daily_tomato_consumption', y=var, data=data)
    plt.title(f'Tomato Consumption vs {var}')
    plt.tight_layout()
    plt.savefig(f'plots/{var}_scatter.png')
    plt.close()

# Statistical tests for top correlations
results = []
for var in top_correlations.index[:5]:
    r, p = stats.pearsonr(data['daily_tomato_consumption'], data[var])
    results.append({
        'variable': var,
        'correlation': r,
        'p_value': p,
        'significant': p < 0.05
    })

results_df = pd.DataFrame(results)
print("\nStatistical test results for top correlations:")
print(results_df)

# Save results for report
with open('analysis_results.txt', 'w') as f:
    f.write("Top Positive Correlations:\n")
    f.write(top_correlations.to_string())
    f.write("\n\nTop Negative Correlations:\n")
    f.write(bottom_correlations.to_string())
    f.write("\n\nStatistical Test Results:\n")
    f.write(results_df.to_string())