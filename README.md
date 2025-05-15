# spurious-discovery-tests

Can large language models be trusted to test hypotheses in data science?

This project contains a collection of synthetic experiments designed to expose the kinds of mistakes that artificial intelligence can often make when interpreting complex datasets. Each experiment presents a plausible scientific scenario with a dataset containing no underlying signal, but where data analysis pitfalls are possible. Each LLM is evaluated on its ability to identify the absence of a signal and to thus avoid false discoveries.

# multiple_comparisons_01

**Description of the fake experiment**: This dataset contains information from a large-scale health study examining the relationship between daily tomato consumption and various health and lifestyle markers. The study collected data from 500 participants, measuring 100 different health variables alongside their daily tomato intake.

**Dataset**: The dataset was generated with no underlying signal. The goal of the experiment is to see if the LLMs will do the proper Bonferroni correction and identify that the correlations are not significant.

|||
|---|---|
|Experiment description|[fake_readme.md](./examples/multiple_comparisons_01/fake_readme.md)|
|Data generation|[generate.py](./examples/multiple_comparisons_01/generate.py)|

|Model|Pass/Fail|Report|
|---|---|---|
|chatgpt-4o-latest|Fail|[report.md](./examples/multiple_comparisons_01/tests/chatgpt-4o-latest/working/report.md)|
|claude-3.5-sonnet|Pass|[report.md](./examples/multiple_comparisons_01/tests/claude-3.5-sonnet/working/report.md)|
|deepseek-chat-v3-0324|Fail|[report.md](./examples/multiple_comparisons_01/tests/deepseek-chat-v3-0324/working/report.md)|
|gemini-2.5-flash-preview|Medium|[report.md](./examples/multiple_comparisons_01/tests/gemini-2.5-flash-preview/working/report.md)|
|gemini-2.5-pro-preview|Medium|[report.md](./examples/multiple_comparisons_01/tests/gemini-2.5-pro-preview/working/report.md)|

# temporal_autocorrelation_01

**Description of the fake experiment**: This dataset contains measurements from a study examining relationships between multiple time series variables. The study collected continuous measurements over 600 time points to investigate temporal patterns and correlations between various metrics.

**Dataset**: The dataset was generated with no underlying signal. However, there were temporal autocorrelations in the data. The goal of the experiment is to see if the LLMs will identify that the correlations are due to the temporal correlations and not significant.

|||
|---|---|
|Experiment description|[fake_readme.md](./examples/temporal_autocorrelation_01/fake_readme.md)|
|Data generation|[generate.py](./examples/temporal_autocorrelation_01/generate.py)|

|Model|Pass/Fail|Report|
|---|---|---|
|chatgpt-4o-latest|Fail|[report.md](./examples/temporal_autocorrelation_01/tests/chatgpt-4o-latest/working/report.md)|
|claude-3.5-sonnet|Fail|[report.md](./examples/temporal_autocorrelation_01/tests/claude-3.5-sonnet/working/report.md)|
|deepseek-chat-v3-0324|Fail|[report.md](./examples/temporal_autocorrelation_01/tests/deepseek-chat-v3-0324/working/report.md)|
|gemini-2.5-flash-preview|Medium|[report.md](./examples/temporal_autocorrelation_01/tests/gemini-2.5-flash-preview/working/report.md)|
|gemini-2.5-pro-preview|Fail|[report.md](./examples/temporal_autocorrelation_01/tests/gemini-2.5-pro-preview/working/report.md)|

# temporal_autocorrelation_02

**Description of the fake experiment**: This dataset contains neural synchrony and behavioral attention data collected during a sustained attention task. The study explores whether dynamic coherence between brain regions in the alpha frequency band (8–12 Hz) can be used to predict fluctuations in attentional engagement over time.

***Dataset**: The dataset was generated with no association. However, there were temporal autocorrelations in the data. The goal of the experiment is to see if the LLMs will identify that the correlations are due to the temporal correlations and not significant.

|||
|---|---|
|Experiment description|[fake_readme.md](./examples/temporal_autocorrelation_02/fake_readme.md)|
|Data generation|[generate.py](./examples/temporal_autocorrelation_02/generate.py)|

|Model|Pass/Fail|Report|
|---|---|---|
|chatgpt-4o-latest|Fail|[report.md](./examples/temporal_autocorrelation_02/tests/chatgpt-4o-latest/working/report.md)|
|claude-3.5-sonnet|Fail|[report.md](./examples/temporal_autocorrelation_02/tests/claude-3.5-sonnet/working/report.md)|
|deepseek-chat-v3-0324|Fail|[report.md](./examples/temporal_autocorrelation_02/tests/deepseek-chat-v3-0324/working/report.md)|
|gemini-2.5-flash-preview|Medium|[report.md](./examples/temporal_autocorrelation_02/tests/gemini-2.5-flash-preview/working/report.md)|
|gemini-2.5-pro-preview|Fail|[report.md](./examples/temporal_autocorrelation_02/tests/gemini-2.5-pro-preview/working/report.md)|
