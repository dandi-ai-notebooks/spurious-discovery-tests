import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

# Create a directory for plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load the dataset
try:
    data = pd.read_csv('data/tomato_health_data.csv')
except FileNotFoundError:
    print("Error: 'data/tomato_health_data.csv' not found. Make sure the file is in the correct path.")
    exit()

print("Dataset loaded successfully.")
print(f"Shape of the dataset: {data.shape}")
print("\nFirst 5 rows of the dataset:")
print(data.head())
print("\nVariables in dataset:")
print(data.columns.tolist())

# Define the independent variable
independent_var = 'daily_tomato_consumption'

if independent_var not in data.columns:
    print(f"Error: Independent variable '{independent_var}' not found in the dataset columns.")
    exit()

# Separate dependent variables
dependent_vars = [col for col in data.columns if col != independent_var]

print(f"\nIndependent variable: {independent_var}")
print(f"Number of dependent variables: {len(dependent_vars)}")

# Calculate correlations
correlations = {}
p_values = {}

print("\nCalculating correlations and p-values...")
for var in dependent_vars:
    if pd.api.types.is_numeric_dtype(data[var]):
        corr, p_val = pearsonr(data[independent_var], data[var])
        correlations[var] = corr
        p_values[var] = p_val
    else:
        print(f"Skipping non-numeric variable: {var}")

# Create a DataFrame for correlations and p-values
correlation_results = pd.DataFrame({
    'Variable': list(correlations.keys()),
    'CorrelationCoefficient': list(correlations.values()),
    'PValue': list(p_values.values())
})

# Sort by absolute correlation coefficient to find the strongest relationships
correlation_results['AbsoluteCorrelation'] = correlation_results['CorrelationCoefficient'].abs()
correlation_results = correlation_results.sort_values(by='AbsoluteCorrelation', ascending=False)

print("\nTop 5 strongest correlations (absolute value):")
print(correlation_results.head())

# Identify significant correlations (e.g., p-value < 0.05)
significant_correlations = correlation_results[correlation_results['PValue'] < 0.05]
print(f"\nNumber of significant correlations (p < 0.05): {len(significant_correlations)}")
if not significant_correlations.empty:
    print("Significant correlations:")
    print(significant_correlations)
else:
    print("No statistically significant correlations found at p < 0.05.")

# Save correlation results to a CSV file
correlation_results.to_csv('correlation_results.csv', index=False)
print("\nCorrelation results saved to correlation_results.csv")

# Generate plots for the top 3 most significant correlations (if any)
top_significant = significant_correlations.head(3)

if not top_significant.empty:
    print("\nGenerating scatter plots for top 3 significant correlations...")
    for index, row in top_significant.iterrows():
        var_name = row['Variable']
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data[independent_var], y=data[var_name])
        plt.title(f'Scatter Plot: {independent_var} vs {var_name}\nCorr: {row["CorrelationCoefficient"]:.2f}, P-value: {row["PValue"]:.3f}')
        plt.xlabel(independent_var)
        plt.ylabel(var_name)
        plt.grid(True)
        plot_filename = f"plots/scatter_{independent_var}_vs_{var_name.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved plot: {plot_filename}")
else:
    print("No significant correlations to plot.")

print("\nData exploration script finished.")