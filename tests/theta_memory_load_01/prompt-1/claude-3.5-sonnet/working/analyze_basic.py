import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Create plots directory
    ensure_dir('plots')
    
    print("Loading data files...")
    # Load data
    memory_df = pd.read_csv("data/memory_load.csv")
    theta_df = pd.read_csv("data/theta_power.csv")
    
    # Merge datasets
    merged_df = memory_df.merge(theta_df, on="time")
    
    # Analysis 1: Basic correlations
    frontal_electrodes = ['Fpz', 'Fz', 'FCz']
    correlations = {}
    
    # Create correlation plot
    plt.figure(figsize=(10, 6))
    
    for electrode in frontal_electrodes:
        col_name = f'theta_{electrode}'
        r, p = stats.pearsonr(merged_df['wm_load'], merged_df[col_name])
        correlations[electrode] = {'r': r, 'p': p}
        print(f"{electrode}: r={r:.3f}, p={p:.6f}")
        
        plt.scatter(merged_df['wm_load'], merged_df[col_name], 
                   alpha=0.1, label=f'{electrode} (r={r:.3f})')
    
    plt.xlabel('Working Memory Load')
    plt.ylabel('Theta Power')
    plt.title('Working Memory Load vs Theta Power')
    plt.legend()
    plt.savefig('plots/correlations.png')
    plt.close()
    
    # Analysis 2: Time series visualization
    plt.figure(figsize=(12, 6))
    # Plot first 200 seconds
    time_window = merged_df['time'] <= 200
    plt.plot(merged_df[time_window]['time'], 
            merged_df[time_window]['wm_load'], 
            label='WM Load', linewidth=2)
    plt.plot(merged_df[time_window]['time'], 
            merged_df[time_window]['theta_Fz'], 
            label='Fz Theta', alpha=0.7)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Value')
    plt.title('Working Memory Load and Theta Power Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/timeseries.png')
    plt.close()
    
    # Analysis 3: State transitions
    wm_median = merged_df['wm_load'].median()
    merged_df['wm_state'] = (merged_df['wm_load'] > wm_median).astype(int)
    transitions = np.where(np.diff(merged_df['wm_state']))[0]
    
    window = 10  # 5 seconds before and after
    transition_matrices = []
    
    for trans in transitions[10:-10]:
        window_data = merged_df.iloc[trans-window:trans+window+1]
        if len(window_data) == 2*window+1:
            transition_matrices.append(window_data['theta_Fz'].values)
    
    transition_avg = np.mean(transition_matrices, axis=0)
    transition_sem = stats.sem(transition_matrices, axis=0)
    
    plt.figure(figsize=(10, 6))
    time_points = np.arange(-window, window+1) * 0.5
    plt.plot(time_points, transition_avg, label='Mean Theta Power')
    plt.fill_between(time_points, 
                    transition_avg - transition_sem,
                    transition_avg + transition_sem,
                    alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='--', label='State Transition')
    plt.xlabel('Time relative to transition (seconds)')
    plt.ylabel('Theta Power')
    plt.title('Theta Power Around WM State Transitions')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/transitions.png')
    plt.close()
    
    # Save numerical results
    with open('analysis_results.md', 'w') as f:
        f.write("# Analysis Results\n\n")
        
        f.write("## Correlation Analysis\n")
        for electrode, stats_dict in correlations.items():
            f.write(f"- {electrode}: r={stats_dict['r']:.3f}, p={stats_dict['p']:.6f}\n")
        
        f.write("\n## State Transition Analysis\n")
        f.write(f"- Number of transitions analyzed: {len(transitions)}\n")
        f.write(f"- Average time between transitions: {np.mean(np.diff(transitions)) * 0.5:.2f} seconds\n")
        
        # Calculate effect size for state transitions
        pre_trans = transition_avg[:window]
        post_trans = transition_avg[window+1:]
        t_stat, p_val = stats.ttest_ind(pre_trans, post_trans)
        cohen_d = (np.mean(post_trans) - np.mean(pre_trans)) / np.std(np.concatenate([pre_trans, post_trans]))
        
        f.write(f"\n## Statistical Tests\n")
        f.write(f"- Pre vs Post Transition t-test: t={t_stat:.3f}, p={p_val:.6f}\n")
        f.write(f"- Effect size (Cohen's d): {cohen_d:.3f}\n")

if __name__ == "__main__":
    main()