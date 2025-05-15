import pandas as pd
from scipy.stats import pearsonr
import sys

try:
    # Read the CSV file. Using low_memory=False to handle large files.
    df = pd.read_csv('data/neural_firing_rates.csv', low_memory=False)

    # Calculate Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(df['region_a_firing_rate'], df['region_b_firing_rate'])

    # Output the results
    print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
    print(f"P-value: {p_value}")

except FileNotFoundError:
    print("Error: data/neural_firing_rates.csv not found.")
    sys.exit(1)
except KeyError:
    print("Error: Ensure 'region_a_firing_rate' and 'region_b_firing_rate' columns exist in the CSV.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)