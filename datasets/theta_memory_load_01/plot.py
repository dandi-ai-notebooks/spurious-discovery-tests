"""plot.py — Quick visualisations for the theta_memoryload_01 dataset.

Run:
    python plot.py

This script:
1. Loads `memory_load.csv` and `theta_power.csv` from the ./data folder.
2. Merges them on the shared `time` column.
3. Produces two figures:
   • Figure 1 – Time‑series plot of working‑memory load and two exemplar
electrode theta‑power traces (FCz & Cz).
   • Figure 2 – Scatter plot with regression line for wm_load vs. theta_FCz.

No inferential statistics are performed; these visuals are purely exploratory.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data"
MEMORY_FILE = os.path.join(DATA_DIR, "memory_load.csv")
THETA_FILE = os.path.join(DATA_DIR, "theta_power.csv")

# ---------------------------------------------------------------------------
# Load and merge data
# ---------------------------------------------------------------------------

try:
    memory_df = pd.read_csv(MEMORY_FILE)
    theta_df = pd.read_csv(THETA_FILE)
except FileNotFoundError as e:
    sys.exit(f"❌  Could not find expected CSV files: {e}")

merged = memory_df.merge(theta_df, on="time", how="inner")

# ---------------------------------------------------------------------------
# Figure 1 — Time‑series overview
# ---------------------------------------------------------------------------

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax1.plot(merged["time"], merged["wm_load"], linewidth=1.2, label="WM Load")
ax1.set_ylabel("wm_load (0–1)")
ax1.set_title("Working‑Memory Load over Time")
ax1.set_ylim(-0.05, 1.05)

# Choose two exemplar electrodes
for elec in ["theta_FCz", "theta_Cz"]:
    ax2.plot(merged["time"], merged[elec], linewidth=0.8, label=elec)

ax2.set_ylabel("Theta Power (norm)")
ax2.set_xlabel("Time (s)")
ax2.set_title("Theta Power at FCz & Cz")
ax2.legend()

fig1.tight_layout()

# ---------------------------------------------------------------------------
# Figure 2 — Scatter wm_load vs. theta_FCz
# ---------------------------------------------------------------------------

fig2, ax = plt.subplots(figsize=(6, 4))
ax.scatter(merged["wm_load"], merged["theta_FCz"], s=5, alpha=0.3)
ax.set_xlabel("wm_load (0–1)")
ax.set_ylabel("theta_FCz (norm)")
ax.set_title("WM Load vs. FCz Theta Power")

# Add simple least‑squares fit line
m, b = np.polyfit(merged["wm_load"], merged["theta_FCz"], deg=1)
wm_vals = np.linspace(0, 1, 100)
ax.plot(wm_vals, m * wm_vals + b, linewidth=2, label=f"slope = {m:.2f}")
ax.legend()

fig2.tight_layout()

plt.show()
