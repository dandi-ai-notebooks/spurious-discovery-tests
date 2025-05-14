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
from pathlib import Path

def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"Ensured directory exists: {directory}")

def main():
    # Create plots directory
    ensure_dir('plots')
    
    try:
        print("Loading data files...")
        # Load data
        memory_df = pd.read_csv("data/memory_load.csv")
        print(f"Loaded memory load data: {len(memory_df)} samples")
        
        theta_df = pd.read_csv("data/theta_power.csv")
        print(f"Loaded theta power data: {len(theta_df)} samples")

        # Merge datasets
        print("Merging datasets...")
        merged_df = memory_df.merge(theta_df, on="time")
        print(f"Merged dataset shape: {merged_df.shape}")

        # Set style for all plots
        plt.style.use('seaborn')
        
        # 1. Basic Analysis
        print("\nPerforming basic analysis...")
        frontal_electrodes = ['Fpz', 'Fz', 'FCz']
        correlations = {}
        
        # Create correlation plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        # Theta-WM correlation plot
        for electrode in frontal_electrodes:
            col_name = f'theta_{electrode}'
            r, p = stats.pearsonr(merged_df['wm_load'], merged_df[col_name])
            correlations[electrode] = {'r': r, 'p': p}
            print(f"{electrode}: r={r:.3f}, p={p:.6f}")
            
            sns.regplot(data=merged_df, x='wm_load', y=col_name, 
                       scatter_kws={'alpha':0.1}, line_kws={'color': 'red'},
                       ax=ax1, label=f'{electrode} (r={r:.3f})')

        ax1.set_xlabel('Working Memory Load')
        ax1.set_ylabel('Theta Power')
        ax1.set_title('Working Memory Load vs Theta Power by Electrode')
        ax1.legend()
        
        # Time series plot
        time_window = merged_df['time'] <= 200  # First 200 seconds
        ax2.plot(merged_df[time_window]['time'], merged_df[time_window]['wm_load'],
                label='WM Load', color='black', linewidth=2)
        ax2.plot(merged_df[time_window]['time'], merged_df[time_window]['theta_Fz'],
                label='Fz Theta Power', color='red', alpha=0.7)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Normalized Value')
        ax2.set_title('Working Memory Load and Frontal Theta Power Over Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = 'plots/basic_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved basic analysis plot to {plot_path}")
        plt.close()

        # 2. Multivariate Analysis
        print("\nPerforming multivariate analysis...")
        electrode_cols = [col for col in theta_df.columns if col.startswith('theta_')]
        X = merged_df[electrode_cols]
        y = merged_df['wm_load']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        multivariate_r2 = r2_score(y_test, y_pred)
        print(f"Multivariate model R² score: {multivariate_r2:.3f}")

        # Electrode weights visualization
        plt.figure(figsize=(12, 8))
        coef_df = pd.DataFrame({
            'electrode': electrode_cols,
            'weight': model.coef_
        })
        sns.barplot(data=coef_df, x='electrode', y='weight')
        plt.xticks(rotation=45)
        plt.xlabel('Electrode')
        plt.ylabel('Weight')
        plt.title('Electrode Contributions to Working Memory Load Prediction')
        
        plot_path = 'plots/electrode_weights.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved electrode weights plot to {plot_path}")
        plt.close()

        # 3. State Transition Analysis
        print("\nAnalyzing state transitions...")
        merged_df['wm_state'] = (merged_df['wm_load'] > merged_df['wm_load'].median()).astype(int)
        transitions = np.where(np.diff(merged_df['wm_state']))[0]
        
        window = 10  # 5 seconds before and after
        transition_matrices = []

        for trans in transitions[10:-10]:
            window_data = merged_df.iloc[trans-window:trans+window+1]
            if len(window_data) == 2*window+1:
                transition_matrices.append(window_data['theta_Fz'].values)

        transition_avg = np.mean(transition_matrices, axis=0)
        transition_sem = stats.sem(transition_matrices, axis=0)

        plt.figure(figsize=(12, 8))
        time_points = np.arange(-window, window+1) * 0.5
        plt.plot(time_points, transition_avg, 'b-', label='Mean Theta Power')
        plt.fill_between(time_points, 
                        transition_avg - transition_sem,
                        transition_avg + transition_sem,
                        alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='--', label='State Transition')
        plt.xlabel('Time relative to transition (seconds)')
        plt.ylabel('Theta Power')
        plt.title('Theta Power Dynamics Around WM Load Transitions')
        plt.legend()
        plt.grid(True)
        
        plot_path = 'plots/state_transitions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved state transitions plot to {plot_path}")
        plt.close()

        # Save numerical results
        print("\nSaving detailed results...")
        results_path = 'analysis_results.md'
        with open(results_path, 'w') as f:
            f.write("# Detailed Analysis Results\n\n")
            
            f.write("## Single Channel Analysis\n")
            for electrode, stats_dict in correlations.items():
                f.write(f"- {electrode}: r={stats_dict['r']:.3f}, p={stats_dict['p']:.6f}\n")
            
            f.write("\n## Multivariate Analysis\n")
            f.write(f"- Multivariate R² score: {multivariate_r2:.3f}\n")
            f.write(f"- Number of features (electrodes): {len(electrode_cols)}\n")
            
            f.write("\n## State Transition Analysis\n")
            f.write(f"- Number of state transitions: {len(transitions)}\n")
            f.write(f"- Average time between transitions: {np.mean(np.diff(transitions)) * 0.5:.2f} seconds\n")

        print(f"Analysis results saved to {results_path}")
        print("\nAnalysis complete! Please check the plots/ directory and analysis_results.md")

    except Exception as e:
        print(f"Error during analysis: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()