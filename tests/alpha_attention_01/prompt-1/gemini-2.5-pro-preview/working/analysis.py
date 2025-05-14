import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import re

def analyze_data():
    # Load data
    try:
        attention_df = pd.read_csv("data/attention.csv")
        synchrony_df = pd.read_csv("data/neural_synchrony.csv")
    except FileNotFoundError:
        print("Error: Ensure 'attention.csv' and 'neural_synchrony.csv' are in the working directory.")
        return

    # Merge data
    merged_df = pd.merge(attention_df, synchrony_df, on="time")

    # Identify synchrony columns
    sync_columns = [col for col in merged_df.columns if col.startswith('sync_')]

    if not sync_columns:
        print("Error: No synchrony columns (e.g., 'sync_1_2') found in 'neural_synchrony.csv'.")
        return

    # --- Hypothesis 1: Overall neural synchrony correlates with attention score ---
    print("\n--- Hypothesis 1: Overall Synchrony vs. Attention ---")
    merged_df['mean_synchrony'] = merged_df[sync_columns].mean(axis=1)

    # Correlation
    corr_overall, p_overall = pearsonr(merged_df['mean_synchrony'], merged_df['attention_score'])
    print(f"Correlation between mean synchrony and attention score: {corr_overall:.4f}")
    print(f"P-value: {p_overall:.4g}")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged_df, x='mean_synchrony', y='attention_score', alpha=0.5)
    sns.regplot(data=merged_df, x='mean_synchrony', y='attention_score', scatter=False, color='red')
    plt.title('Mean Neural Synchrony vs. Attention Score')
    plt.xlabel('Mean Synchrony')
    plt.ylabel('Attention Score')
    plt.grid(True)
    plt.savefig('avg_sync_vs_attention.png')
    plt.close()
    print("Saved plot to avg_sync_vs_attention.png")

    # --- Hypothesis 2: Specific region-pair connections are more informative ---
    print("\n--- Hypothesis 2: Specific Pair Synchrony vs. Attention ---")
    pair_correlations = {}
    for sync_col in sync_columns:
        corr, p_value = pearsonr(merged_df[sync_col], merged_df['attention_score'])
        if not np.isnan(corr): # Check for NaN correlations (e.g., if a sync column has no variance)
             pair_correlations[sync_col] = {'correlation': corr, 'p_value': p_value}

    if not pair_correlations:
        print("No valid correlations found for individual pairs.")
        return

    sorted_correlations = sorted(pair_correlations.items(), key=lambda item: item[1]['correlation'])

    print("\nTop 5 positively correlated pairs:")
    for pair, data in reversed(sorted_correlations[-5:]):
        print(f"{pair}: Correlation = {data['correlation']:.4f}, P-value = {data['p_value']:.4g}")

    print("\nTop 5 negatively correlated pairs:")
    for pair, data in sorted_correlations[:5]:
        print(f"{pair}: Correlation = {data['correlation']:.4f}, P-value = {data['p_value']:.4g}")

    # Plot all pair correlations
    correlations_df = pd.DataFrame.from_dict(pair_correlations, orient='index').sort_values(by='correlation', ascending=False)

    # For better visualization, if there are too many pairs, we might plot only a subset
    # or use a different plot type. For now, a bar plot for all.
    num_pairs = len(correlations_df)
    if num_pairs > 50: # Arbitrary threshold for readability
        print(f"Plotting all {num_pairs} pair correlations. This might be dense.")

    plt.figure(figsize=(max(15, num_pairs * 0.2), 8)) # Adjust width based on number of pairs
    sns.barplot(x=correlations_df.index, y=correlations_df['correlation'], palette="vlag")
    plt.xticks(rotation=90, fontsize=8 if num_pairs > 30 else 10)
    plt.xlabel('Region Pair')
    plt.ylabel('Correlation with Attention Score')
    plt.title('Correlation of Individual Region Pair Synchrony with Attention Score')
    plt.tight_layout()
    plt.savefig('pair_correlations.png')
    plt.close()
    print("Saved plot to pair_correlations.png")

if __name__ == "__main__":
    analyze_data()