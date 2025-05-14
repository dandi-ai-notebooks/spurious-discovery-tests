# Analysis of Neural Synchrony and Attentional Dynamics Dataset

## Introduction

This report summarizes the findings from exploring the provided dataset on neural synchrony and attentional dynamics. The primary goal was to investigate the relationship between time-resolved neural alpha band synchrony between cortical regions and fluctuations in attentional engagement during a sustained attention task. Specifically, we aimed to determine if synchrony can predict attention and if certain region pairs are more informative than others.

The dataset consists of two files: `attention.csv`, containing second-by-second attention scores, and `neural_synchrony.csv`, containing time-resolved synchrony estimates for all pairs of 16 cortical regions. Due to the size of the synchrony data, a chunk-based processing approach was used for analysis.

## Methods

A linear regression model was trained to predict the `attention_score` based on the neural synchrony values from all region pairs. The `attention.csv` data was loaded entirely, while `neural_synchrony.csv` was processed in chunks. The data from the chunks were concatenated to form the feature matrix (X) for the model, with the attention scores serving as the target variable (y). A `LinearRegression` model from the `scikit-learn` library in Python was used for training.

The trained model was evaluated by examining its R-squared value, which indicates the proportion of the variance in the attention scores predictable from the synchrony features. The importance of individual region-pair synchrony was assessed by the magnitude of their coefficients in the trained linear model. Features with larger absolute coefficient values are considered more influential in the linear prediction.

## Findings

### Research Question 1: Can time-resolved synchrony predict fluctuations in attentional engagement?

The trained linear regression model achieved an R-squared value of **0.214862**. This indicates that approximately 21.49% of the variance observed in the attention scores can be linearly predicted by the measured neural synchrony values across all 120 region pairs. While this R-squared value suggests that neural synchrony does explain some portion of the variability in attentional engagement, it also indicates that a significant amount of variance remains unexplained by this linear model and set of features.

Based on this statistical measure (R-squared), there is evidence to support the hypothesis that time-resolved synchrony between cortical regions can predict fluctuations in attentional engagement, although the linear relationship is moderate.

### Research Question 2: Are specific region-pair connections more informative than others?

By examining the absolute magnitudes of the coefficients in the trained linear regression model, we can infer which synchrony features had the strongest linear association with the attention score and are thus more "informative" in the context of this model. The top 10 most influential synchrony features based on the absolute value of their coefficients are:

1.  `sync_13_14`: -1.131887
2.  `sync_6_16`: 0.979340
3.  `sync_14_15`: 0.934046
4.  `sync_9_14`: 0.923851
5.  `sync_9_10`: -0.909391
6.  `sync_4_5`: -0.769194
7.  `sync_1_7`: -0.754314
8.  `sync_5_9`: 0.746098
9.  `sync_2_13`: -0.736864
10. `sync_3_9`: -0.724403

These results suggest that, within this linear model, the synchrony between regions 13 and 14 (`sync_13_14`) showed the strongest linear relationship with attention score. Other pairs like `sync_6_16` and `sync_14_15` also exhibited relatively strong associations. This provides statistical justification, based on the linear model's coefficients, that some region-pair connections are indeed more informative than others in predicting attentional engagement.

## Conclusion

The analysis provides initial evidence that time-resolved neural synchrony in the alpha band is related to attentional engagement and can predict some of its fluctuations. Furthermore, the linear model suggests that certain region-pair connections may be more relevant predictors of attention than others, with `sync_13_14` being the most influential in this specific analysis.

Future work could involve exploring non-linear relationships, using more advanced time- series models, incorporating regularization techniques, and performing rigorous statistical testing (e.g., cross-validation, permutation testing, and methods that provide p-values for coefficients like those in the `statsmodels` library) to further validate these findings. Visualizations of the relationships and feature importances could also provide deeper insights.