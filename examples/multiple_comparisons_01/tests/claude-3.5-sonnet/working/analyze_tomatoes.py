import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

# Set random seed for reproducibility
np.random.seed(42)

# Read the data
df = pd.read_csv('data/tomato_health_data.csv')

# Create output directory for plots
import os
os.makedirs('plots', exist_ok=True)

# Basic statistics of tomato consumption
tomato_stats = df['daily_tomato_consumption'].describe()
print("\nTomato Consumption Statistics:")
print(tomato_stats)

# Calculate correlations with tomato consumption
correlations = df.corr()['daily_tomato_consumption'].sort_values(ascending=False)

# Perform statistical tests for each variable
p_values = []
variable_names = []

for column in df.columns:
    if column != 'daily_tomato_consumption':
        # Pearson correlation test
        correlation, p_value = stats.pearsonr(df['daily_tomato_consumption'], df[column])
        p_values.append(p_value)
        variable_names.append(column)

# Correct for multiple comparisons using Benjamini-Hochberg
rejected, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')

# Create DataFrame with results
results = pd.DataFrame({
    'Variable': variable_names,
    'Correlation': [correlations[var] for var in variable_names],
    'P-value': p_values,
    'Corrected P-value': corrected_p_values,
    'Significant': rejected
})

# Sort by absolute correlation
results['Abs_Correlation'] = abs(results['Correlation'])
results = results.sort_values('Abs_Correlation', ascending=False)
results = results.drop('Abs_Correlation', axis=1)

# Save top 20 strongest correlations
print("\nTop 20 Strongest Correlations:")
print(results.head(20))

# Plot distribution of tomato consumption
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='daily_tomato_consumption', bins=30)
plt.title('Distribution of Daily Tomato Consumption')
plt.xlabel('Number of Tomatoes per Day')
plt.ylabel('Count')
plt.savefig('plots/tomato_distribution.png')
plt.close()

# Plot top 5 positive and negative correlations
top_pos = results[results['Correlation'] > 0].head(5)
top_neg = results[results['Correlation'] < 0].head(5)
top_correlations = pd.concat([top_pos, top_neg])

plt.figure(figsize=(12, 6))
sns.barplot(data=top_correlations, x='Correlation', y='Variable')
plt.title('Top 5 Positive and Negative Correlations with Tomato Consumption')
plt.tight_layout()
plt.savefig('plots/top_correlations.png')
plt.close()

# Create scatter plots for top 3 positive correlations
for _, row in top_pos.head(3).iterrows():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='daily_tomato_consumption', y=row['Variable'])
    plt.title(f'Tomato Consumption vs {row["Variable"]}')
    plt.xlabel('Daily Tomato Consumption')
    plt.savefig(f'plots/scatter_{row["Variable"].replace("/", "_")}.png')
    plt.close()

# Save results to CSV
results.to_csv('correlation_results.csv', index=False)