# spurious-discovery-tests

Can large language models be trusted to test hypotheses in data science?

This project is a collection of synthetic experiments designed to expose the kinds of mistakes that artificial intelligence can often make when interpreting complex datasets. Each experiment presents a plausible scientific scenario with data containing no underlying signal, but where spurious patterns may emerge. Each LLM and prompt pair is evaluated on its ability to identify the absence of a signal and to thus avoid false discoveries.

Each dataset in [datasets/](datasets/) has two files:
* generate.py
* readme.md

The generate.py file contains the code to generate the dataset (usually via simulation). The readme.md file contains a description of the dataset as though it came from a real experiment including some possible research questions or hypotheses. An AI agent (powered by the LLM) is instructed to:
* Read the readme.md file
* Explore the dataset by creating and executing scripts
* Test various hypotheses
* Summarize the findings in a new file: report.md

Then we (humans) examine the report to see if it succeeded by failing to find a signal.

Two prompts were used. The only difference is that the second prompt includes the command "Do not invent results. If nothing of significance is found, say so."

- Prompt 1: [templates/prompt-1.txt](templates/prompt-1.txt)
- Prompt 2: [templates/prompt-2.txt](templates/prompt-2.txt)

# Dataset alpha_attention_01

Dataset description / generation: [readme.md](datasets/alpha_attention_01/readme.md) / [generate.py](datasets/alpha_attention_01/generate.py)

|Prompt|Model|Pass/Fail|Quoted Conclusion|Report|
|---|---|---|---|---|
|prompt-1|chatgpt-4o-latest|Fail|Statistical and regression modeling shows that cortical synchrony features can reliably predict attention fluctuations. Specific connections exhibit strong associations, supporting the hypothesis that alpha-band coherence reflects attentional state dynamics.|[report.md](tests/alpha_attention_01/prompt-1/chatgpt-4o-latest/working/report.md)|
|prompt-1|gpt-4.1|Fail|Neural synchrony between certain cortical region pairs, particularly those identified in the regression and correlation analyses, reliably predicts second-by-second fluctuations in attention. The relationship varies strongly across region pairs, indicating some connections are more informative than others. The observed effects are robust to permutation-based significance testing.|[report.md](tests/alpha_attention_01/prompt-1/gpt-4.1/working/report.md)|
|prompt-1|claude-3.5-sonnet|Fail|These findings support the hypothesis that time-resolved synchrony between cortical regions can predict fluctuations in attentional engagement. The results also highlight specific region-pair connections that appear particularly informative for understanding attentional states.|[report.md](tests/alpha_attention_01/prompt-1/claude-3.5-sonnet/working/report.md)|
|prompt-1|claude-3.7-sonnet|Fail|This analysis demonstrates that alpha-band neural synchrony is intricately linked to attentional dynamics, though in ways more complex than initially hypothesized.|[report.md](tests/alpha_attention_01/prompt-1/claude-3.7-sonnet/working/report.md)|
|prompt-1|gemini-2.5-flash-preview|Fail|The analysis provides initial evidence that time-resolved neural synchrony in the alpha band is related to attentional engagement and can predict some of its fluctuations. Furthermore, the linear model suggests that certain region-pair connections may be more relevant predictors of attention than others, with `sync_13_14` being the most influential in this specific analysis.|[report.md](tests/alpha_attention_01/prompt-1/gemini-2.5-flash-preview/working/report.md)|
|prompt-1|gemini-2.5-pro-preview|Fail|While a statistically significant positive correlation exists between **overall mean neural synchrony** and attention, the relationship is very weak (r = 0.0617). This implies that global average alpha synchrony is not a robust standalone predictor of attentional fluctuations. Specific brain region pairs exhibit considerably stronger and more statistically significant correlations with attention scores. Both positive and negative correlations were observed, with some pairs showing moderate predictive power (e.g., `sync_2_16` with r = 0.2495).|[report.md](tests/alpha_attention_01/prompt-1/gemini-2.5-pro-preview/working/report.md)|
|prompt-1|deepseek-chat-v3-0324|Fail|Neural synchrony shows region-specific relationships with attention scores. Significant correlations were found after correcting for multiple comparisons|[report.md](tests/alpha_attention_01/prompt-1/deepseek-chat-v3-0324/working/report.md)|
|---|---|---|---|
|prompt-2|chatgpt-4o-latest|Fail|Dynamic neural synchrony in the alpha band shows significant association with behavioral attention measures. Region 16 appears in all top correlations, suggesting it may play a central role in attentional modulation.|[report.md](tests/alpha_attention_01/prompt-2/chatgpt-4o-latest/working/report.md)|
|prompt-2|gpt-4.1|Fail|We find statistically robust, albeit moderate, relationships between dynamic cortical synchrony and attentional engagement. The results suggest that coherence between specific regions (notably those involving region 16) is a meaningful neural correlate of attention in this context.|[report.md](tests/alpha_attention_01/prompt-2/gpt-4.1/working/report.md)|
|prompt-2|claude-3.5-sonnet|Moderate Pass|While we found statistically significant relationships between neural synchrony and attention, the correlations were generally modest in magnitude. The strongest correlation (r = 0.249) suggests that even the most predictive region pair explains only about 6% of the variance in attention scores.|[report.md](tests/alpha_attention_01/prompt-2/claude-3.5-sonnet/working/report.md)|
|prompt-2|claude-3.7-sonnet|Fail|Significant correlations exist between neural synchrony and attention: 70 out of 120 region pairs showed statistically significant correlations with attention, though the correlations were modest in magnitude (maximum r = 0.25)|[report.md](tests/alpha_attention_01/prompt-2/claude-3.7-sonnet/working/report.md)|
|prompt-2|gemini-2.5-flash-preview|Fail|Based on this exploratory analysis, there is some evidence to suggest a relationship between time-resolved neural synchrony and attentional engagement. The multiple linear regression model indicates that, collectively, neural synchrony measures can significantly predict a portion of the variance in attention scores (approximately 21.5%). However, the individual correlations are generally weak, and the multiple regression analysis suggests that identifying specific statistically significant individual synchrony predictors is challenging, likely due to multicollinearity within the synchrony data.|[report.md](tests/alpha_attention_01/prompt-2/gemini-2.5-flash-preview/working/report.md)|
|prompt-2|gemini-2.5-pro-preview|Fail|The analysis provides evidence supporting the hypothesis that time-resolved synchrony between cortical regions can predict fluctuations in attentional engagement. Specifically, the synchrony between regions 2 and 16 (`sync_2_16`) showed the strongest correlation with the attention score in this dataset. While statistically significant, the observed correlations are generally weak to moderate. This suggests that while alpha-band synchrony does carry information about attentional state, it is likely one of many contributing neural factors.|[report.md](tests/alpha_attention_01/prompt-2/gemini-2.5-pro-preview/working/report.md)|
|prompt-2|deepseek-chat-v3-0324|Moderate Pass|The results suggest that synchrony between certain brain region pairs, particularly those involving region 16, show the strongest predictive relationship with attentional fluctuations. However, the correlations are modest (r ≈ 0.25), indicating synchrony explains only about 6% of the variance in attention scores.|[report.md](tests/alpha_attention_01/prompt-2/deepseek-chat-v3-0324/working/report.md)|
