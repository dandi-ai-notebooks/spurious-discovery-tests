import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
try:
    attention_df = pd.read_csv("data/attention.csv")
    synchrony_df = pd.read_csv("data/neural_synchrony.csv")
except FileNotFoundError as e:
    print(f"Error loading data file: {e}")
    exit()

# Merge dataframes
merged_df = pd.merge(attention_df, synchrony_df, on="time")

# --- Research Question 1: Can synchrony predict attention? ---
# Calculate correlations between attention score and all synchrony measures
sync_columns = [col for col in merged_df.columns if col.startswith('sync_')]
correlations = merged_df[sync_columns].corrwith(merged_df['attention_score'])

print("--- Correlation Analysis ---")
print("Correlation coefficients between attention_score and synchrony measures:")
print(correlations.sort_values(ascending=False).head()) # Print top 5 positive correlations
print(correlations.sort_values(ascending=True).head())  # Print top 5 negative correlations
print(f"\nNumber of synchrony measures: {len(sync_columns)}")
print(f"Mean absolute correlation: {correlations.abs().mean():.4f}")
print(f"Maximum absolute correlation: {correlations.abs().max():.4f}")

# Perform a multiple linear regression to see overall predictive power
X = merged_df[sync_columns]
y = merged_df['attention_score']

# Add a constant for the intercept
X = sm.add_constant(X)

print("\n--- Multiple Linear Regression Analysis ---")
try:
    model = sm.OLS(y, X, n_jobs=-1).fit()
    print(model.summary())
except Exception as e:
    print(f"Error during regression analysis: {e}")


# --- Research Question 2: Are specific region-pair connections more informative? ---
# The correlations already give us insight into this. We can also look at p-values from regression if significant.
# For simplicity, we'll focus on the correlation magnitudes.
print("\n--- Most Informative Region Pairs (based on absolute correlation) ---")
abs_correlations = correlations.abs().sort_values(ascending=False)
print(abs_correlations.head(10)) # Print top 10 region pairs by absolute correlation

# Note: For a more rigorous analysis of 'informativeness', one would typically use feature selection
# or look at significant coefficients in the regression model after addressing multicollinearity.
# Given the large number of synchrony measures, multicollinearity is likely present.