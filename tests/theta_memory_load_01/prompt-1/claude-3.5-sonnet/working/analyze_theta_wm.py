import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import sys

try:
    print("Loading data files...")
    # Load data
    memory_df = pd.read_csv("data/memory_load.csv")
    print(f"Loaded memory load data: {len(memory_df)} samples")
    
    theta_df = pd.read_csv("data/theta_power.csv")
    print(f"Loaded theta power data: {len(theta_df)} samples")

    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)

    # Merge datasets
    print("Merging datasets...")
    merged_df = memory_df.merge(theta_df, on="time")
    print(f"Merged dataset shape: {merged_df.shape}")

    # 1. Single Channel Analysis
    print("\nPerforming single channel analysis...")
    frontal_electrodes = ['Fpz', 'Fz', 'FCz']
    correlations = {}
    plt.figure(figsize=(12, 8))

    for electrode in frontal_electrodes:
        col_name = f'theta_{electrode}'
        r, p = stats.pearsonr(merged_df['wm_load'], merged_df[col_name])
        correlations[electrode] = {'r': r, 'p': p}
        print(f"{electrode}: r={r:.3f}, p={p:.6f}")
        
        plt.scatter(merged_df['wm_load'], merged_df[col_name], alpha=0.1, label=f'{electrode} (r={r:.3f})')

    plt.xlabel('Working Memory Load')
    plt.ylabel('Theta Power')
    plt.title('Working Memory Load vs Theta Power by Electrode')
    plt.legend()
    plt.savefig('plots/single_channel_correlation.png')
    print("Saved single channel correlation plot")
    plt.close()

    # 2. Multivariate Analysis
    print("\nPerforming multivariate analysis...")
    electrode_cols = [col for col in theta_df.columns if col.startswith('theta_')]
    X = merged_df[electrode_cols]
    y = merged_df['wm_load']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # Train multivariate model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)
    multivariate_r2 = r2_score(y_test, y_pred)
    print(f"Multivariate model R² score: {multivariate_r2:.3f}")

    # Compare to best single channel
    single_channel_r2s = {}
    for col in electrode_cols:
        reg = LinearRegression()
        reg.fit(X_train[[col]], y_train)
        y_pred_single = reg.predict(X_test[[col]])
        single_channel_r2s[col] = r2_score(y_test, y_pred_single)

    best_single_channel = max(single_channel_r2s.items(), key=lambda x: x[1])
    print(f"Best single channel ({best_single_channel[0]}): R²={best_single_channel[1]:.3f}")

    # Plot electrode coefficients
    plt.figure(figsize=(12, 8))
    plt.bar(electrode_cols, model.coef_)
    plt.xticks(rotation=45)
    plt.xlabel('Electrode')
    plt.ylabel('Coefficient Weight')
    plt.title('Contribution of Each Electrode to WM Load Prediction')
    plt.tight_layout()
    plt.savefig('plots/electrode_weights.png')
    print("Saved electrode weights plot")
    plt.close()

    # Calculate additional statistics for interpretation
    print("\nCalculating additional statistics...")
    # Test if multivariate R² is significantly better than best single channel
    from scipy import stats
    n = len(y_test)
    z = 0.5 * np.log((1 + multivariate_r2)/(1 - multivariate_r2)) - \
        0.5 * np.log((1 + best_single_channel[1])/(1 - best_single_channel[1]))
    se = np.sqrt(1/(n-3))
    z_score = z/se
    p_value = 2*(1 - stats.norm.cdf(abs(z_score)))

    # Save comprehensive results
    print("\nSaving analysis results...")
    with open('analysis_results.txt', 'w') as f:
        f.write("Single Channel Analysis:\n")
        f.write("----------------------\n")
        for electrode, stats_dict in correlations.items():
            f.write(f"{electrode}: r={stats_dict['r']:.3f}, p={stats_dict['p']:.6f}\n")
        
        f.write("\nMultivariate Analysis:\n")
        f.write("---------------------\n")
        f.write(f"Multivariate R² score: {multivariate_r2:.3f}\n")
        f.write(f"Best single channel ({best_single_channel[0]}) R² score: {best_single_channel[1]:.3f}\n")
        f.write(f"\nStatistical Comparison:\n")
        f.write(f"Z-score for R² difference: {z_score:.3f}\n")
        f.write(f"P-value for R² difference: {p_value:.6f}\n")

    print("Analysis complete! Check plots/ directory and analysis_results.txt for results.")

except Exception as e:
    print(f"Error during analysis: {str(e)}", file=sys.stderr)
    sys.exit(1)