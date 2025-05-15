# Comprehensive Study on Tomato Consumption and Health Markers

## Dataset Description

This dataset contains information from a large-scale health study examining the relationship between daily tomato consumption and various health and lifestyle markers. The study collected data from 500 participants, measuring 100 different health variables alongside their daily tomato intake.

## Data Collection

Data was collected through:
1. Daily food diaries tracking tomato consumption
2. Comprehensive health screenings
3. Blood tests and biomarker analysis
4. Lifestyle questionnaires
5. Wearable device data
6. Gut microbiome analysis

### Variables

1. `daily_tomato_consumption`: Average number of tomatoes consumed per day (0-5 tomatoes)

2. Health and Lifestyle Variables (24 basic markers):
   - Blood pressure (systolic and diastolic)
   - Heart rate and respiratory rate
   - Body measurements (weight, BMI, circumferences)
   - Sleep patterns
   - Physical activity metrics
   - Stress and lifestyle indicators

3. Biomarkers (50 different markers):
   - Various biological indicators
   - Metabolic markers
   - Inflammatory markers

4. Nutritional Markers:
   - 6 vitamin levels (A, B, C, D, E, K)
   - 6 mineral levels (iron, calcium, magnesium, zinc, selenium, potassium)
   - 5 blood markers (glucose, insulin, cholesterol, triglycerides, CRP)
   - 5 hormone levels (cortisol, testosterone, estrogen, thyroid, growth)

5. Immune System Markers (30 different markers):
   - Various immune response indicators
   - Inflammation markers
   - Immune cell counts

6. Metabolic Markers (30 different markers):
   - Energy metabolism indicators
   - Substrate utilization markers
   - Metabolic rate indicators

7. Gut Microbiome Analysis (48 different bacteria):
   - Various bacterial species abundance
   - Diversity metrics
   - Metabolic byproducts

## Research Question

The primary research question this dataset aims to address is:

"What are the health benefits of daily tomato consumption?"

This investigation seeks to identify potential correlations between tomato consumption and health markers, which could provide insights into how tomato consumption might influence various aspects of human health.

## Data Format

The data is provided in CSV format with the following structure:
- Each row represents one participant
- First column: daily tomato consumption (continuous variable, 0-5 tomatoes per day)
- Remaining columns: 100 different health and lifestyle variables (continuous variables)
- No missing values

## Usage

```python
import pandas as pd

# Load the dataset
data = pd.read_csv('data/tomato_health_data.csv')

# View available variables
print("\nVariables in dataset:")
print(data.columns)
```
