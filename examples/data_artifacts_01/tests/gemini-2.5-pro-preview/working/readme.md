# Neural Activity Correlation Study

## Dataset Description

This dataset contains continuous recordings of neural firing rates from two distinct brain regions (Region A and Region B). The recordings span 6 hours with measurements taken every second, resulting in 21,600 time points per region.

## Data Collection

Data was collected through:
1. Multi-electrode arrays implanted in both brain regions
2. Continuous recording system
3. Synchronized data acquisition

### Data Quality Note

There researchers reported some problems with data acquisition. There may be periods during the recording where the data is corrupted.

## Variables

The dataset contains two main variables:

1. `region_a_firing_rate`: Neural firing rate (spikes/second) from Region A

2. `region_b_firing_rate`: Neural firing rate (spikes/second) from Region B

## Research Question

The primary research question this dataset aims to address is:

"Is there a functional relationship between the firing rates of Region A and Region B?"

## Data Format

The data is provided in CSV format with the following structure:
```python
import pandas as pd

# Load the dataset
data = pd.read_csv('data/neural_firing_rates.csv')

# Example structure:
# time_seconds | region_a_firing_rate | region_b_firing_rate
# 0           | 15.2               | 18.3
# 1           | 12.8               | 16.1
# ...
```
