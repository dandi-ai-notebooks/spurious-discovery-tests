# Neural Synchrony and Attentional Dynamics

## Overview

This dataset contains neural synchrony and behavioral attention data collected during a sustained attention task. The study explores whether dynamic coherence between brain regions in the alpha frequency band (8–12 Hz) can be used to predict fluctuations in attentional engagement over time.

Data were collected during a continuous, 30-minute attentional task, during which subjects responded to intermittent visual stimuli. EEG signals were recorded from multiple cortical regions and coherence values were computed across all region pairs using sliding-window spectral analysis. Attention was indexed via behavioral performance metrics aggregated on a per-second basis.

---

## Research Questions

* Can time-resolved synchrony between cortical regions predict fluctuations in attentional engagement?
* Are specific region-pair connections more informative than others?

---

## Files

### `attention.csv`

Contains second-by-second estimates of attentional engagement:

| Column            | Description                                |
| ----------------- | ------------------------------------------ |
| `time`            | Time in seconds (0 to 1799)                |
| `attention_score` | Continuous attention index (range: 0 to 1) |

### `neural_synchrony.csv`

Contains neural synchrony estimates between brain region pairs:

| Column     | Description                                      |
| ---------- | ------------------------------------------------ |
| `time`     | Time in seconds (0 to 1799)                      |
| `sync_i_j` | Coherence between brain region *i* and *j* (0–1) |

There are 16 cortical regions labeled 1 through 16. All region-pair combinations are included (`sync_1_2`, `sync_1_3`, ..., `sync_15_16`).

---

## Loading the Data

You can load the data into Python using `pandas`:

```python
import pandas as pd

# Load attention scores
attention_df = pd.read_csv("data/attention.csv")

# Load synchrony values
synchrony_df = pd.read_csv("data/neural_synchrony.csv")
```

The dataframes can be merged on the `time` column for joint analysis.
