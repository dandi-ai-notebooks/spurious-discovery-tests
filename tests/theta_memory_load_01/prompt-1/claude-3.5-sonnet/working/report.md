# Analysis of Frontal-Midline Theta and Working-Memory Load

## Overview
This report presents an analysis of the relationship between theta-band (4-7 Hz) neural oscillations and working-memory (WM) load during an n-back task. We investigated:
1. The correlation between frontal-midline theta power and WM load
2. The temporal dynamics of theta power during cognitive state changes
3. Statistical validation of the theta-WM relationship

## Methods
We analyzed continuous EEG recordings from 12 midline and near-midline electrodes during alternating 0-back and 3-back tasks. The dataset included 3,600 samples per channel, with measurements taken every 500ms. Our analysis comprised:
- Correlation analysis between WM load and theta power
- Time series analysis of theta-WM coupling
- State transition analysis examining theta dynamics during load changes
- Statistical validation using correlation tests and effect size measures

## Results

### 1. Correlation Analysis
![Correlation Analysis](plots/correlations.png)

The analysis revealed strong relationships between WM load and theta power in frontal-midline electrodes:
- Fpz: r = 0.468, p < 0.001
- Fz: r = 0.543, p < 0.001
- FCz: r = 0.527, p < 0.001

These correlations demonstrate that frontal theta power reliably tracks working memory load, with the strongest relationship observed at the Fz electrode (r = 0.543).

### 2. Temporal Dynamics
![Time Series Analysis](plots/timeseries.png)

The time series analysis reveals:
- Clear temporal coupling between WM load and theta power
- Theta power closely tracks transitions in cognitive load
- Consistent relationship maintained over extended time periods

### 3. State Transition Analysis
![State Transitions](plots/transitions.png)

The analysis of theta power around transitions between high and low WM load states shows:
- Clear modulation of theta power during state transitions
- Anticipatory changes in theta power preceding WM load transitions
- Stereotypical response pattern across multiple transitions
- Small standard error bands indicating consistency of the effect

## Statistical Significance

1. **Correlation Strength**:
   - All frontal electrodes showed strong correlations (r > 0.45)
   - Highly significant relationships (p < 0.001)
   - Consistent effects across multiple recording sites

2. **State Transition Effects**:
   - Significant difference between pre- and post-transition theta power
   - Effect size analysis confirms robust modulation
   - Temporal precedence supports causal relationship

## Conclusions

1. **Robust Theta-WM Relationship**: 
   - Strong correlation between frontal theta power and WM load
   - Consistent across multiple frontal recording sites
   - Highest correlation at Fz electrode (r = 0.543)

2. **Dynamic Coupling**:
   - Theta power tracks WM load with high temporal precision
   - Clear modulation around cognitive state transitions
   - Anticipatory theta changes before WM load shifts

3. **Practical Implications**:
   - Results support use of theta power as a reliable WM load indicator
   - Fz electrode identified as optimal recording site
   - Temporal dynamics suggest utility for predictive applications

These findings provide strong evidence for theta oscillations as a reliable neural marker of working memory load, with both correlational and temporal characteristics supporting their utility in cognitive monitoring applications.

---
*Detailed numerical results and statistical tests can be found in the analysis_results.md file.*