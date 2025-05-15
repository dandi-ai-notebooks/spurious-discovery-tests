import os
import numpy as np
import pandas as pd

# Set a seed for reproducibility
np.random.seed(42)

if not os.path.exists('data'):
    os.makedirs('data')

# Simulation parameters
n_seconds = 1800  # 30 minutes of data at 1 Hz sampling rate
n_regions = 16     # Number of brain regions
sampling_rate = 1 # Hz

# Time vector
time = np.arange(n_seconds)

# ---------------------------------------
# Generate the attention score timeseries
# ---------------------------------------

# Create a Gaussian random walk (Brownian motion) for attention
attention_noise = np.random.normal(loc=0, scale=0.05, size=n_seconds)
attention_score = np.cumsum(attention_noise)

# Smooth the attention signal to simulate plausible slow drift (e.g., using convolution)
window = np.hanning(50)
window /= window.sum()
attention_score_smooth = np.convolve(attention_score, window, mode='same')

# Normalize to [0, 1]
attention_score_norm = (attention_score_smooth - np.min(attention_score_smooth)) / \
                       (np.max(attention_score_smooth) - np.min(attention_score_smooth))

# ---------------------------------------
# Generate synthetic synchrony predictors
# ---------------------------------------

# Generate AR(1) time series for each region's signal
def generate_ar1_series(length, alpha=0.95, noise_std=0.05):
    series = np.zeros(length)
    for t in range(1, length):
        series[t] = alpha * series[t - 1] + np.random.normal(0, noise_std)
    return series

# Compute coherence-like values between region pairs
# In this synthetic case, just use correlations or combined AR(1) signals
region_signals = [generate_ar1_series(n_seconds) for _ in range(n_regions)]

# Normalize region signals to [0, 1] for realism
region_signals = [(sig - np.min(sig)) / (np.max(sig) - np.min(sig)) for sig in region_signals]

# Compute pseudo-coherence values (again, entirely synthetic)
synchrony_data = {}
for i in range(n_regions):
    for j in range(i + 1, n_regions):
        # Combine the two signals and add noise to simulate coherence
        sync_signal = 0.5 * (region_signals[i] + region_signals[j]) \
                      + np.random.normal(0, 0.01, n_seconds)
        # Normalize to [0, 1] to mimic coherence range
        sync_signal = (sync_signal - np.min(sync_signal)) / (np.max(sync_signal) - np.min(sync_signal))
        key = f"sync_{i+1}_{j+1}"
        synchrony_data[key] = sync_signal

# ---------------------------------------
# Save the datasets to CSV
# ---------------------------------------

# Save attention data
attention_df = pd.DataFrame({
    "time": time,
    "attention_score": attention_score_norm
})
attention_df.to_csv("data/attention.csv", index=False)

# Save synchrony data
synchrony_df = pd.DataFrame({"time": time})
synchrony_df = pd.concat([synchrony_df, pd.DataFrame(synchrony_data)], axis=1)
synchrony_df.to_csv("data/neural_synchrony.csv", index=False)

# ---------------------------------------
# Notes:
# ---------------------------------------
# - There is *no true relationship* between attention_score and sync_* variables.
# - However, all timeseries are autocorrelated and some predictors are correlated with each other.
# - Regression models may find apparently "significant" patterns purely due to temporal structure.
