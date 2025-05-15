import pandas as pd
import numpy as np

# Load attention scores
try:
    attention_df = pd.read_csv("data/attention.csv")
    print("Loaded data/attention.csv")
except FileNotFoundError:
    print("Error: data/attention.csv not found.")
    exit()

# Load synchrony values
try:
    synchrony_df = pd.read_csv("data/neural_synchrony.csv")
    print("Loaded data/neural_synchrony.csv")
except FileNotFoundError:
    print("Error: data/neural_synchrony.csv not found.")
    exit()

# Merge dataframes
merged_df = pd.merge(attention_df, synchrony_df, on='time')
print(f"Merged dataframes. Shape: {merged_df.shape}")

# Descriptive statistics for attention
print("\nAttention Score Descriptive Statistics:")
print(merged_df['attention_score'].describe())

# Descriptive statistics for synchrony columns (first few as an example)
print("\nSample Neural Synchrony Descriptive Statistics (first 5 columns):")
sync_cols = [col for col in merged_df.columns if col.startswith('sync_')]
if sync_cols:
    print(merged_df[sync_cols[:5]].describe())
else:
    print("No synchrony columns found.")

# Potential visualization steps (add visualization libraries like matplotlib or seaborn)
# print("\nPotential plots:")
# print("- Histogram of attention scores")
# print("- Time series plot of attention score")
# print("- Time series plots of selected synchrony pairs")
# print("- Correlation matrix or heat map of synchrony pairs")
# print("- Scatter plots of attention score vs. key synchrony pairs")

print("\nData exploration complete.")