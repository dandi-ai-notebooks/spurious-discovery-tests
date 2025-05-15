import pandas as pd
import statsmodels.api as sm

# Load the dataset
# Assuming the file fits in memory for 600 rows
try:
    data = pd.read_csv('data/timeseries_data.csv')
except FileNotFoundError:
    print("Error: data/timeseries_data.csv not found.")
    exit()

# Get the target variable
target = data['target_variable']

# Get the predictor variables (excluding the target variable)
predictors = data.drop(columns=['target_variable'])

# Calculate correlations between the target and each predictor
correlations = predictors.corrwith(target)

# Perform a simple linear regression for each predictor to get p-values for significance
# Note: This is a simplified approach and doesn't account for time series properties like autocorrelation
# For a more rigorous analysis, time series specific tests would be needed.
results = {}
for col in predictors.columns:
    X = sm.add_constant(predictors[col]) # Add an intercept term to the model
    model = sm.OLS(target, X).fit()
    results[col] = {
        'correlation': correlations[col],
        'p_value': model.pvalues[col] # p-value for the predictor variable
    }

# Print the results
print("Correlation and p-values with target_variable:")
for key, value in results.items():
    print(f"{key}: Correlation = {value['correlation']:.4f}, P-value = {value['p_value']:.4f}")

# Optional: Save results to a file or analyze further
# You can add code here to save 'results' to a JSON or other format if needed for report generation.