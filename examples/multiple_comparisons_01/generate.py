import pandas as pd
import numpy as np
from scipy import stats
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(7)  # choose thi so we get some positive correlations

# Generate sample size
n_samples = 500

# Generate random daily tomato consumption (0-5 tomatoes per day)
tomato_consumption = np.random.uniform(0, 5, n_samples)

# Generate 100 random health and lifestyle variables
# Each row is a participant, each column is a different variable
n_variables = 100
variable_data = np.random.normal(size=(n_samples, n_variables))

# Create column names for health and lifestyle variables
health_lifestyle_vars = [
    'blood_pressure_systolic', 'blood_pressure_diastolic', 'heart_rate', 'respiratory_rate',
    'body_temperature', 'weight', 'bmi', 'waist_circumference', 'hip_circumference',
    'body_fat_percentage', 'muscle_mass', 'bone_density', 'sleep_hours', 'deep_sleep_percentage',
    'rem_sleep_percentage', 'steps_per_day', 'exercise_minutes', 'standing_hours',
    'water_consumption_ml', 'caffeine_mg', 'alcohol_units', 'stress_level',
    'meditation_minutes', 'screen_time_hours'
] # 24

# Add specific biomarkers and nutrition markers
biomarkers = [f'biomarker_{i}' for i in range(1, 15)] # 14
vitamins = [f'vitamin_{c}' for c in 'ABCDEK'] # 6
minerals = [f'mineral_{m}' for m in ['iron', 'calcium', 'magnesium', 'zinc', 'selenium', 'potassium']] # 6
blood_markers = [f'blood_{m}' for m in ['glucose', 'insulin', 'cholesterol', 'triglycerides', 'crp']] # 5
hormones = [f'hormone_{h}' for h in ['cortisol', 'testosterone', 'estrogen', 'thyroid', 'growth']] # 5
immune_markers = [f'immune_{i}' for i in range(1, 15)] # 14
metabolic_markers = [f'metabolic_{i}' for i in range(1, 15)] # 14
gut_bacteria = [f'bacteria_{i}' for i in range(1, 13)] # 12

# Combine all variable names
all_variables = (health_lifestyle_vars + biomarkers + vitamins + minerals +
                blood_markers + hormones + immune_markers + metabolic_markers +
                gut_bacteria)
assert len(all_variables) == n_variables, f"Mismatch in number of variables: {len(all_variables)} vs {n_variables}"

# Create DataFrame
data = pd.DataFrame(variable_data, columns=all_variables)
data['daily_tomato_consumption'] = tomato_consumption

# Save to CSV
data.to_csv('data/tomato_health_data.csv', index=False)

print("Generated dataset with the following characteristics:")
print(f"\nNumber of participants: {n_samples}")
print(f"Number of health variables: {n_variables}")
print("\nTomato consumption statistics:")
print(data['daily_tomato_consumption'].describe())

# Calculate and print number of "significant" correlations at p < 0.05
# (These will be purely by chance since all data is random)
significant_positive_correlations = 0
significant_negative_correlations = 0
total_postive_correlations = 0
total_negative_correlations = 0

for column in all_variables:
    correlation = np.corrcoef(data['daily_tomato_consumption'], data[column])[0, 1]
    p_value = 2 * (1 - stats.norm.cdf(abs(correlation) * np.sqrt(n_samples-3)))
    if correlation > 0:
        total_postive_correlations += 1
    else:
        total_negative_correlations += 1
    if p_value < 0.05:
        if correlation > 0:
            significant_positive_correlations += 1
        else:
            significant_negative_correlations += 1

print(f"\nNumber of 'significant' positive correlations at p < 0.05: {significant_positive_correlations}")
print(f"Number of 'significant' negative correlations at p < 0.05: {significant_negative_correlations}")
print(f"Total number of correlations tested: {n_variables}")
print(f"Expected false positives (5% of {n_variables}): {n_variables * 0.05:.1f}")
