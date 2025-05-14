# Report on Frontal-Midline Theta and Working Memory Load

## Introduction

This report summarizes an exploratory analysis of the relationship between frontal-midline theta power and working memory (WM) load using the provided dataset. The dataset contains time-series data for WM load and theta power recorded from various electrodes during a continuous n-back task.

The primary research question addressed in this analysis is whether frontal-midline theta power tracks moment-to-moment fluctuations in WM load.

## Methods

Data from `data/memory_load.csv` and `data/theta_power.csv` were loaded and merged based on the `time` column. Pearson correlation analysis was conducted to quantify the linear relationship between the continuous working memory load index (`wm_load`) and the theta power from selected frontal-midline electrodes (Fpz, Fz, FCz, Cz, CPz, Pz). The p-value for each correlation was calculated to assess statistical significance.

The analysis was performed using a Python script (`analyze_data.py`) utilizing the pandas library for data handling and scipy.stats for correlation and p-value calculation.

## Results

The Pearson correlation coefficients and corresponding p-values between WM load and theta power for the selected frontal-midline electrodes are as follows:

*   **theta_Fpz:** Correlation = -0.0981, P-value = 0.0000
*   **theta_Fz:** Correlation = -0.0306, P-value = 0.0660
*   **theta_FCz:** Correlation = 0.0704, P-value = 0.0000
*   **theta_Cz:** Correlation = 0.1058, P-value = 0.0000
*   **theta_CPz:** Correlation = 0.0057, P-value = 0.7327
*   **theta_Pz:** Correlation = 0.0230, P-value = 0.1671

A significance level of p &lt; 0.05 was used for this analysis. The results indicate statistically significant correlations between WM load and theta power at electrodes Fpz, FCz, and Cz. The correlations are relatively weak, with both negative (Fpz) and positive (FCz, Cz) relationships observed. Electrodes Fz, CPz, and Pz did not show a statistically significant correlation with WM load in this analysis.

## Conclusion

Based on the Pearson correlation analysis, there is evidence that theta power at certain frontal-midline electrodes (Fpz, FCz, and Cz) is associated with fluctuations in working memory load. However, the observed correlations are weak, suggesting that while there might be a relationship, it is not a strong linear one across all tested frontal-midline sites. electrode-specific differences were observed, with some electrodes showing significant correlations while others did not.

Further analysis, potentially employing time-series specific methods or exploring non-linear relationships, could provide deeper insights into how frontal-midline theta activity tracks working memory load. Investigating multivariate patterns, as suggested in the original research questions, could also reveal more complex relationships.