import pandas as pd
from scipy.stats import pearsonr
import sys

# Load attention scores
try:
    attention_df = pd.read_csv("data/attention.csv")
    print("Loaded data/attention.csv", file=sys.stderr) # Print to stderr to keep stdout clean for results
except FileNotFoundError:
    print("Error: data/attention.csv not found.", file=sys.stderr)
    exit()

# Load synchrony values
try:
    synchrony_df = pd.read_csv("data/neural_synchrony.csv")
    print("Loaded data/neural_synchrony.csv", file=sys.stderr) # Print to stderr
except FileNotFoundError:
    print("Error: data/neural_synchrony.csv not found.", file=sys.stderr)
    exit()

# Merge dataframes
merged_df = pd.merge(attention_df, synchrony_df, on='time')
print(f"Merged dataframes. Shape: {merged_df.shape}", file=sys.stderr) # Print to stderr

# Perform correlation analysis
attention_scores = merged_df['attention_score']
sync_cols = [col for col in merged_df.columns if col.startswith('sync_')]

correlation_results = []
for col in sync_cols:
    # Ensure no NaN or infinite values that could cause issues with pearsonr
    valid_indices = merged_df[[col, 'attention_score']].dropna().index
    if len(valid_indices) > 0:
        correlation, p_value = pearsonr(merged_df.loc[valid_indices, col], merged_df.loc[valid_indices, 'attention_score'])
        correlation_results.append({'sync_pair': col, 'correlation': correlation, 'p_value': p_value})
    else:
        correlation_results.append({'sync_pair': col, 'correlation': None, 'p_value': None})


# Convert results to a DataFrame and print
results_df = pd.DataFrame(correlation_results)

# Sort by absolute correlation for easier interpretation of strongest relationships
results_df['abs_correlation'] = results_df['correlation'].abs()
results_df = results_df.sort_values(by='abs_correlation', ascending=False).drop(columns=['abs_correlation'])

# Print results (only print relevant columns to stdout)
# Setting a significance level (e.g., alpha = 0.05)
alpha = 0.05
print("\nCorrelation Results (sorted by absolute correlation):")
print(results_df[['sync_pair', 'correlation', 'p_value']].to_string(index=False))

# Optional: Identify significant correlations
# print(f"\nSignificant Correlations (p < {alpha}):")
# significant_results = results_df[results_df['p_value'] < alpha]
# print(significant_results[['sync_pair', 'correlation', 'p_value']].to_string(index=False))

print("\nStatistical analysis complete.", file=sys.stderr) # Print to stderr