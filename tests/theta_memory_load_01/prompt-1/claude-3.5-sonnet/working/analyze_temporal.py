import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Loading data files...")
# Load data
memory_df = pd.read_csv("data/memory_load.csv")
theta_df = pd.read_csv("data/theta_power.csv")

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

# Merge datasets
merged_df = memory_df.merge(theta_df, on="time")

# 1. Time series plot of WM load and theta power
print("Generating time series plot...")
plt.figure(figsize=(15, 10))

# Plot first 200 seconds for better visibility
time_window = merged_df['time'] <= 200
plt.plot(merged_df[time_window]['time'], merged_df[time_window]['wm_load'], 
         label='WM Load', color='black', linewidth=2)
plt.plot(merged_df[time_window]['time'], merged_df[time_window]['theta_Fz'], 
         label='Fz Theta Power', color='red', alpha=0.7)

plt.xlabel('Time (seconds)')
plt.ylabel('Normalized Value')
plt.title('Working Memory Load and Frontal Theta Power Over Time')
plt.legend()
plt.grid(True)
plt.savefig('plots/temporal_dynamics.png')
plt.close()

# 2. Cross-correlation analysis
print("Performing cross-correlation analysis...")
frontal_electrodes = ['Fpz', 'Fz', 'FCz']
max_lag = 10  # 5 seconds (given 0.5s sampling)
plt.figure(figsize=(12, 8))

for electrode in frontal_electrodes:
    col_name = f'theta_{electrode}'
    xcorr = np.correlate(merged_df['wm_load'] - merged_df['wm_load'].mean(),
                        merged_df[col_name] - merged_df[col_name].mean(),
                        mode='full') / len(merged_df)
    lags = np.arange(-max_lag, max_lag + 1)
    center = len(xcorr) // 2
    xcorr_subset = xcorr[center-max_lag:center+max_lag+1]
    
    plt.plot(lags * 0.5, xcorr_subset, label=electrode)  # multiply by 0.5 to convert to seconds

plt.xlabel('Lag (seconds)')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation between WM Load and Theta Power')
plt.legend()
plt.grid(True)
plt.savefig('plots/cross_correlation.png')
plt.close()

# 3. State transition analysis
print("Analyzing state transitions...")
# Define high/low WM states using median split
wm_median = merged_df['wm_load'].median()
merged_df['wm_state'] = (merged_df['wm_load'] > wm_median).astype(int)
# Find state transitions
transitions = np.where(np.diff(merged_df['wm_state']))[0]

# Analyze theta power around transitions
window = 10  # 5 seconds before and after
transition_matrices = []

for trans in transitions[10:-10]:  # Skip transitions too close to edges
    window_data = merged_df.iloc[trans-window:trans+window+1]
    if len(window_data) == 2*window+1:  # Ensure complete window
        transition_matrices.append(window_data['theta_Fz'].values)

transition_avg = np.mean(transition_matrices, axis=0)
transition_sem = stats.sem(transition_matrices, axis=0)

plt.figure(figsize=(12, 8))
time_points = np.arange(-window, window+1) * 0.5  # Convert to seconds
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
plt.savefig('plots/state_transitions.png')
plt.close()

# Save numerical results
with open('temporal_analysis.txt', 'w') as f:
    f.write("Temporal Analysis Results\n")
    f.write("========================\n\n")
    
    # Calculate phase lag between WM load and theta power
    from scipy.signal import hilbert
    analytic_wm = hilbert(merged_df['wm_load'] - merged_df['wm_load'].mean())
    analytic_theta = hilbert(merged_df['theta_Fz'] - merged_df['theta_Fz'].mean())
    phase_diff = np.angle(analytic_wm * np.conj(analytic_theta))
    mean_phase_diff = np.mean(phase_diff)
    
    f.write(f"Mean phase difference: {mean_phase_diff:.3f} radians\n")
    
    # Calculate transition statistics
    f.write(f"\nNumber of state transitions: {len(transitions)}\n")
    f.write(f"Average time between transitions: {np.mean(np.diff(transitions)) * 0.5:.2f} seconds\n")

print("Analysis complete! Check plots/ directory and temporal_analysis.txt for results.")