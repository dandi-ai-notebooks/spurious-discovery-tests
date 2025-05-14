# Frontal‑Midline Theta and Working‑Memory Load

## Overview

This dataset captures the relationship between **theta‑band (4–7 Hz) neural oscillations** and **working‑memory (WM) load** during a continuous *n‑back* task. Twenty healthy adults performed alternating 0‑back and 3‑back blocks for 30 minutes while scalp EEG was recorded from 12 midline and near‑midline electrodes. Time‑resolved theta power was extracted every **500 ms** using complex‑Morlet wavelets, producing 3 600 samples per channel. WM load was indexed once per second by convolving block labels with a canonical impulse‑response kernel, yielding a smooth, continuous estimate of cognitive demand.

---

## Research Questions

* **Does frontal‑midline theta power track moment‑to‑moment fluctuations in WM load?**
* **Do multivariate theta patterns improve prediction of WM load compared with single‑channel measures?**

---

## Files

### `data/memory_load.csv`

| Column    | Description                                       |
| --------- | ------------------------------------------------- |
| `time`    | Time in seconds (0 to 1 799.5, in 0.5‑s steps)    |
| `wm_load` | Continuous working‑memory load index (range: 0–1) |

### `data/theta_power.csv`

| Column              | Description                                                    |
| ------------------- | -------------------------------------------------------------- |
| `time`              | Time in seconds (0 to 1 799.5, in 0.5‑s steps)                 |
| `theta_<electrode>` | Normalised theta power at the specified electrode (range: 0–1) |

**Electrodes recorded**: Fpz, Fz, FCz, Cz, CPz, Pz, F1, F2, C1, C2, P1, P2.

---

## Loading the Data

```python
import pandas as pd

# Load working‑memory load trace
gl_df = pd.read_csv("data/memory_load.csv")

# Load theta power data
th_df = pd.read_csv("data/theta_power.csv")

# Merge on shared time column for joint analyses
merged = gl_df.merge(th_df, on="time")
```

Time indices align exactly across files; merge on `time` to relate WM load to theta power.
