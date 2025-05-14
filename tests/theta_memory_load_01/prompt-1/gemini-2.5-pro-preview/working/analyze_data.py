import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np

# --- Configuration ---
OUTPUT_DIR = "plots"
FRONTAL_MIDLINE_ELECTRODES = ['Fz', 'FCz', 'Cz'] # As per typical definitions

# --- Helper Functions ---
def ensure_dir(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_plot(fig, filename, directory=OUTPUT_DIR):
    ensure_dir(directory)
    path = f"{directory}/{filename}"
    fig.savefig(path)
    print(f"Saved plot: {path}")
    plt.close(fig)

# --- Load Data ---
print("Loading data...")
wm_load_df = pd.read_csv("data/memory_load.csv")
theta_power_df = pd.read_csv("data/theta_power.csv")

# Merge dataframes
merged_df = pd.merge(wm_load_df, theta_power_df, on="time")
print("Data loaded and merged.")
print("Merged DataFrame head:")
print(merged_df.head())

# --- Analysis for Research Question 1: Theta power and WM load ---
print("\n--- Analyzing Research Question 1 ---")
print("Does frontal-midline theta power track moment-to-moment fluctuations in WM load?")

# Select frontal-midline electrodes
theta_columns = [f"theta_{elec}" for elec in FRONTAL_MIDLINE_ELECTRODES]

# 1. Time series plot of WM load and average frontal-midline theta
merged_df['theta_frontal_midline_avg'] = merged_df[theta_columns].mean(axis=1)

fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(merged_df['time'], merged_df['wm_load'], label='WM Load', color='blue')
ax1.plot(merged_df['time'], merged_df['theta_frontal_midline_avg'], label='Avg. Frontal-Midline Theta', color='red', alpha=0.7)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Normalized Value')
ax1.set_title('WM Load and Average Frontal-Midline Theta Power Over Time')
ax1.legend()
ax1.grid(True)
save_plot(fig1, "wm_load_vs_avg_frontal_theta_timeseries.png")

# 2. Scatter plots and correlations for individual frontal-midline electrodes
correlations_rq1 = {}
for elec_col in theta_columns:
    electrode_name = elec_col.replace('theta_', '')

    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=merged_df['wm_load'], y=merged_df[elec_col], ax=ax_scatter, alpha=0.5)
    # Add a regression line
    sns.regplot(x=merged_df['wm_load'], y=merged_df[elec_col], ax=ax_scatter, scatter=False, color='red')
    ax_scatter.set_xlabel('WM Load')
    ax_scatter.set_ylabel(f'Theta Power ({electrode_name})')
    ax_scatter.set_title(f'WM Load vs. Theta Power at {electrode_name}')
    ax_scatter.grid(True)
    save_plot(fig_scatter, f"wm_load_vs_{elec_col}_scatter.png")

    # Calculate Pearson correlation
    corr, p_value = pearsonr(merged_df['wm_load'], merged_df[elec_col])
    correlations_rq1[electrode_name] = {'correlation': corr, 'p_value': p_value}
    print(f"Correlation between WM Load and {electrode_name} Theta: r = {corr:.3f}, p = {p_value:.3e}")

# Also for the average frontal-midline theta
corr_avg, p_value_avg = pearsonr(merged_df['wm_load'], merged_df['theta_frontal_midline_avg'])
correlations_rq1['Frontal_Midline_Avg'] = {'correlation': corr_avg, 'p_value': p_value_avg}
print(f"Correlation between WM Load and Avg Frontal-Midline Theta: r = {corr_avg:.3f}, p = {p_value_avg:.3e}")


# --- Analysis for Research Question 2: Multivariate theta patterns ---
print("\n--- Analyzing Research Question 2 ---")
print("Do multivariate theta patterns improve prediction of WM load compared with single-channel measures?")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X = merged_df.drop(columns=['time', 'wm_load', 'theta_frontal_midline_avg']) # All theta columns
y = merged_df['wm_load']

# Standardize features for regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Define a KFold for cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 1. Single-channel model (using the "best" frontal-midline electrode from RQ1)
best_single_electrode_name = max(
    (e for e in FRONTAL_MIDLINE_ELECTRODES),
    key=lambda elec: abs(correlations_rq1[elec]['correlation'])
)
best_single_electrode_col = f"theta_{best_single_electrode_name}"
print(f"Best single frontal-midline electrode for prediction: {best_single_electrode_name} (based on correlation with WM Load)")

X_single_best = X_scaled_df[[best_single_electrode_col]]
model_single_best = LinearRegression()
scores_single_best = cross_val_score(model_single_best, X_single_best, y, cv=cv, scoring='r2')
r2_single_best_mean = np.mean(scores_single_best)
print(f"Cross-validated R^2 for best single frontal-midline electrode ({best_single_electrode_name}): {r2_single_best_mean:.3f}")

# 2. Multivariate model (all frontal-midline electrodes initially)
X_frontal_midline_multi = X_scaled_df[theta_columns]
model_frontal_midline_multi = LinearRegression()
scores_frontal_midline_multi = cross_val_score(model_frontal_midline_multi, X_frontal_midline_multi, y, cv=cv, scoring='r2')
r2_frontal_midline_multi_mean = np.mean(scores_frontal_midline_multi)
print(f"Cross-validated R^2 for multivariate frontal-midline theta: {r2_frontal_midline_multi_mean:.3f}")

# 3. Multivariate model (all available electrodes)
X_all_multi = X_scaled_df # All theta columns (already scaled)
model_all_multi = LinearRegression()
scores_all_multi = cross_val_score(model_all_multi, X_all_multi, y, cv=cv, scoring='r2')
r2_all_multi_mean = np.mean(scores_all_multi)
print(f"Cross-validated R^2 for multivariate all-electrode theta: {r2_all_multi_mean:.3f}")

results_rq2 = {
    'single_best_electrode': best_single_electrode_name,
    'r2_single_best': r2_single_best_mean,
    'r2_frontal_midline_multi': r2_frontal_midline_multi_mean,
    'r2_all_multi': r2_all_multi_mean
}

# Save results to a file for report generation
import json
with open('analysis_results.json', 'w') as f:
    json.dump({'rq1_correlations': correlations_rq1, 'rq2_r2_scores': results_rq2}, f, indent=4)

print("\nAnalysis complete. Results and plots saved.")
print("Run 'python analyze_data.py' to generate plots and results.")