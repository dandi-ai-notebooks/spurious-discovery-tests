# Time Series Analysis Study

## Dataset Description

This dataset contains measurements from a study examining relationships between multiple time series variables. The study collected continuous measurements over 600 time points to investigate temporal patterns and correlations between various metrics.

## Data Collection

Data was collected through:
1. Continuous monitoring system
2. Real-time measurement of multiple metrics
3. Synchronized data collection across all variables

### Variables

1. `target_variable`: Normalized target measurement (0-1)
   - Higher values indicate stronger signal
   - Computed using rolling averages

2. Time Series Metrics (Predictor Variables):
   Measured across four variables (var1, var2, var3, var4):
   - metric1
   - metric2
   - metric3
   - metric4
   - metric5

   Variables are named as `{metric}_{variable}`, e.g.:
   - `metric1_var1`
   - `metric2_var4`

## Research Question

The primary research question this dataset aims to address is:

Is there an association between the target variable and the predictor variables over time?

## Data Format

```python
import pandas as pd

# Load the dataset
data = pd.read_csv('data/timeseries_data.csv')

# Example structure:
# target_variable | metric1_var1 | metric1_var2 | ... | metric5_var3 | metric5_var4
# 0.75           | -0.5         | 0.2          | ... | 0.4          | 0.8
# ...
