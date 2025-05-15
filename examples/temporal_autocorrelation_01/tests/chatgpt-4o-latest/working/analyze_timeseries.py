import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

# Load data
data = pd.read_csv('data/timeseries_data.csv')

# Summary statistics
summary = data.describe()
summary.to_csv('summary_statistics.csv')

# Correlation matrix
corr = data.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Pearson correlation between each predictor and target_variable
correlations = []
for col in data.columns:
    if col != 'target_variable':
        r, p = pearsonr(data['target_variable'], data[col])
        correlations.append({'variable': col, 'pearson_r': r, 'p_value': p})

# Save correlations to CSV
cor_df = pd.DataFrame(correlations)
cor_df.to_csv('target_correlations.csv', index=False)

# Plot top 5 variables most correlated with the target
top_vars = cor_df.sort_values('p_value').head(5)['variable']
for var in top_vars:
    plt.figure()
    sns.scatterplot(x=data[var], y=data['target_variable'])
    plt.title(f'target_variable vs {var}')
    plt.xlabel(var)
    plt.ylabel('target_variable')
    plt.tight_layout()
    plt.savefig(f'scatter_{var}.png')
    plt.close()