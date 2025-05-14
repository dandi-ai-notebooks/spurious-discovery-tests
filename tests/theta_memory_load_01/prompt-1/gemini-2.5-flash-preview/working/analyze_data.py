import pandas as pd
from scipy.stats import pearsonr
import numpy as np

# Load working‑memory load trace
try:
    gl_df = pd.read_csv("data/memory_load.csv")
except FileNotFoundError:
    print("Error: data/memory_load.csv not found.")
    exit()

# Load theta power data
try:
    th_df = pd.read_csv("data/theta_power.csv")
except FileNotFoundError:
    print("Error: data/theta_power.csv not found.")
    exit()

# Merge on shared time column for joint analyses
merged = gl_df.merge(th_df, on="time")

# Define frontal-midline electrodes
frontal_midline_electrodes = ["theta_Fpz", "theta_Fz", "theta_FCz", "theta_Cz", "theta_CPz", "theta_Pz"]

# Calculate and report correlations and p-values
correlation_results = {}
print("Analyzing correlations between WM load and frontal-midline theta power:")
for electrode in frontal_midline_electrodes:
    if electrode in merged.columns:
        correlation, p_value = pearsonr(merged["wm_load"], merged[electrode])
        correlation_results[electrode] = {"correlation": correlation, "p_value": p_value}
        print(f"  {electrode}: Correlation = {correlation:.4f}, P-value = {p_value:.4f}")
    else:
        print(f"  Warning: Electrode {electrode} not found in data.")

# You can save these results to a file if needed
# import json
# with open("correlation_results.json", "w") as f:
#     json.dump(correlation_results, f, indent=4)