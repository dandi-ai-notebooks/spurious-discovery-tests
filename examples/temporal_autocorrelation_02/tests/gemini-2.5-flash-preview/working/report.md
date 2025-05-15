# Dataset Exploration and Analysis Report

## Introduction

This report summarizes the exploration and statistical analysis of the neural synchrony and attentional dynamics dataset provided. The objective was to investigate the relationship between time-resolved neural synchrony in the alpha frequency band and fluctuations in behavioral attentional engagement during a sustained attention task.

## Data Description

The dataset includes second-by-second measurements of `attention_score` (a continuous index ranging from 0 to 1) and neural synchrony values (`sync_i_j`) between 16 cortical regions (indexed 1 through 16) over a 30-minute period (1800 seconds).

Descriptive statistics for the `attention_score` are as follows:

| Statistic | Value    |
|-----------|----------|
| count     | 1800.00  |
| mean      | 0.36     |
| std       | 0.30     |
| min       | 0.00     |
| 25%       | 0.08     |
| 50%       | 0.27     |
| 75%       | 0.59     |
| max       | 1.00     |

The attention scores vary throughout the task, with a mean around 0.36, indicating that subjects were not at peak attention consistently. The range from 0 to 1 is fully utilized. Descriptive statistics for individual synchrony pairs also showed variation over time (sample statistics are provided in the appendix).

## Statistical Analysis

To assess whether time-resolved synchrony between cortical regions can predict fluctuations in attentional engagement, Pearson correlation analysis was performed for each neural synchrony pair (`sync_i_j`) and the `attention_score`.

Using a significance level of $\alpha = 0.05$, several synchrony pairs showed statistically significant correlations with the attention score. The table below lists the top 10 synchrony pairs with the strongest absolute correlations:

| sync_pair | correlation | p_value    |
|-----------|-------------|------------|
| sync_2_16 | 0.249       | 6.04e-27   |
| sync_11_16| 0.246       | 3.33e-26   |
| sync_12_16| 0.228       | 1.44e-22   |
| sync_8_16 | 0.208       | 4.88e-19   |
| sync_2_11 | 0.207       | 6.53e-19   |
| sync_10_16| 0.197       | 2.79e-17   |
| sync_5_15 | -0.174      | 1.19e-13   |
| sync_5_13 | -0.172      | 1.82e-13   |
| sync_13_15| -0.167      | 8.73e-13   |
| sync_2_8  | 0.167       | 1.06e-12   |

As shown, several synchrony pairs (e.g., sync_2_16, sync_11_16, sync_12_16) exhibit statistically significant positive correlations with attention score, while others (e.g., sync_5_15, sync_5_13, sync_13_15) show significant negative correlations. The very small p-values associated with these correlations indicate that the observed relationships are unlikely to be due to random chance.

However, it is important to note that while statistically significant, the magnitudes of the correlation coefficients are relatively modest (the highest absolute correlation is approximately 0.249). This suggests that while these specific neural synchrony dynamics are related to attentional engagement, they explain only a small portion of the variance in attention scores.

## Conclusion

Based on the analysis, time-resolved neural synchrony, particularly in specific region pairs, is statistically significantly correlated with fluctuations in attentional engagement in this dataset. This finding supports the research question that dynamic coherence between brain regions can be related to attentional states. Specific region-pair connections appear to be more informative than others, as evidenced by the varying correlation strengths and significance levels across different synchrony pairs.

While the correlations are statistically significant and provide evidence for a relationship, the relatively small effect sizes suggest that neural synchrony is likely one of several factors influencing attentional dynamics. Further research incorporating other potential predictors and employing more complex modeling techniques may provide a more comprehensive understanding of the neural basis of sustained attention.

## Appendix

Sample Neural Synchrony Descriptive Statistics (first 5 columns):

| Statistic | sync_1_2 | sync_1_3 | sync_1_4 | sync_1_5 | sync_1_6 |
|-----------|----------|----------|----------|----------|----------|
| count     | 1800.00  | 1800.00  | 1800.00  | 1800.00  | 1800.00  |
| mean      | 0.52     | 0.56     | 0.54     | 0.47     | 0.47     |
| std       | 0.16     | 0.15     | 0.19     | 0.18     | 0.17     |
| min       | 0.00     | 0.00     | 0.00     | 0.00     | 0.00     |
| 25%       | 0.41     | 0.46     | 0.41     | 0.34     | 0.35     |
| 50%       | 0.51     | 0.57     | 0.54     | 0.47     | 0.47     |
| 75%       | 0.62     | 0.66     | 0.69     | 0.58     | 0.59     |
| max       | 1.00     | 1.00     | 1.00     | 1.00     | 1.00   |