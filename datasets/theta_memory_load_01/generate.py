import os
import numpy as np
import pandas as pd

"""Generate synthetic dataset theta_memoryload_01.

Creates two CSV files in a ./data directory:
    - memory_load.csv   : columns [time, wm_load]
    - theta_power.csv   : columns [time, theta_<electrode>]

There is *no true relationship* between wm_load and theta power; any
apparent correlations are spurious and arise from autocorrelation and
shared temporal structure.
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

np.random.seed(42)                    # For reproducibility
DURATION_MINUTES = 30                 # Total experiment length
SAMPLE_INTERVAL = 0.5                 # Seconds between samples (2 Hz)
N_SAMPLES = int(DURATION_MINUTES * 60 / SAMPLE_INTERVAL)
ELECTRODES = [
    "Fpz", "Fz", "FCz", "Cz", "CPz", "Pz",
    "F1", "F2", "C1", "C2", "P1", "P2",
]
DATA_DIR = "data"

# ---------------------------------------------------------------------------
# Ensure output directory exists
# ---------------------------------------------------------------------------

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def ar1(length: int, alpha: float = 0.9, noise_std: float = 0.08) -> np.ndarray:
    """Generate an AR(1) process."""
    x = np.zeros(length)
    for t in range(1, length):
        x[t] = alpha * x[t - 1] + np.random.normal(0, noise_std)
    return x

# ---------------------------------------------------------------------------
# 1. Working‑memory load trace (wm_load)
# ---------------------------------------------------------------------------

BLOCK_SIZE = 120                      # Samples ≈ 60 s at 2 Hz
N_BLOCKS = int(np.ceil(N_SAMPLES / BLOCK_SIZE))
block_labels = np.random.choice([0, 3], size=N_BLOCKS)  # 0‑back or 3‑back
wm_load_raw = np.repeat(block_labels, BLOCK_SIZE)[:N_SAMPLES]

# Smooth block edges with a Gaussian kernel (~12.5 s wide)
KERNEL_LEN = 25
x = np.linspace(-2, 2, KERNEL_LEN)
gauss_kernel = np.exp(-0.5 * x**2)
gauss_kernel /= gauss_kernel.sum()
wm_load_smoothed = np.convolve(wm_load_raw, gauss_kernel, mode="same")

# Min‑max normalise to [0, 1]
wm_load = (wm_load_smoothed - wm_load_smoothed.min()) / (
    wm_load_smoothed.max() - wm_load_smoothed.min()
)

# ---------------------------------------------------------------------------
# 2. Theta‑band power for each electrode
# ---------------------------------------------------------------------------

# Global latent AR(1) to induce mild spatial correlation
GLOBAL_LATENT = ar1(N_SAMPLES, alpha=0.95, noise_std=0.05)

theta_dict = {}
for elec in ELECTRODES:
    independent = ar1(N_SAMPLES, alpha=0.9, noise_std=0.1)
    signal = 0.3 * GLOBAL_LATENT + 0.7 * independent
    # Rescale to [0, 1] (dB‑like power range)
    signal = (signal - signal.min()) / (signal.max() - signal.min())
    theta_dict[f"theta_{elec}"] = signal

# ---------------------------------------------------------------------------
# 3. Save CSV files
# ---------------------------------------------------------------------------

time = np.arange(N_SAMPLES) * SAMPLE_INTERVAL

memory_df = pd.DataFrame({"time": time, "wm_load": wm_load})
memory_df.to_csv(os.path.join(DATA_DIR, "memory_load.csv"), index=False)

theta_df = pd.DataFrame({"time": time})
theta_df = pd.concat([theta_df, pd.DataFrame(theta_dict)], axis=1)
theta_df.to_csv(os.path.join(DATA_DIR, "theta_power.csv"), index=False)

# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
# - wm_load and all theta signals are *independent*.
# - Strong autocorrelation and mild cross‑electrode correlation can create
#   convincing but entirely spurious structure.
