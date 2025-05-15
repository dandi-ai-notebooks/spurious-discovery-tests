import pandas as pd
import scipy.stats as stats

# Define the path to the dataset
data_path = 'data/tomato_health_data.csv'

# Initialize a dictionary to store sums and sum of squares for each variable
# This is needed to calculate standard deviation and covariance for correlation
sums = {}
sum_sq = {}
sum_prod = {}
counts = {}
correlation_results = {}

# Initialize variables to store correlation coefficients and p-values
correlation_coeffs = {}
p_values = {}

# Get the list of columns from the first row without loading the whole file
with open(data_path, 'r') as f:
    header = f.readline().strip().split(',')
    variables = header[1:] # Exclude 'daily_tomato_consumption' which is the first column

# Prepare dictionaries with initial values
for var in variables:
    sums[var] = 0
    sum_sq[var] = 0
    sum_prod[var] = 0 # For product with daily_tomato_consumption
    counts[var] = 0

# Calculate sums and sum of squares for daily_tomato_consumption
tomato_sum = 0
tomato_sum_sq = 0
total_count = 0

# Read the CSV file in chunks
chunk_size = 10000  # Adjust chunk size based on available memory
for chunk in pd.read_csv(data_path, chunksize=chunk_size):
    # Process daily_tomato_consumption
    chunk_tomato = chunk['daily_tomato_consumption']
    tomato_sum += chunk_tomato.sum()
    tomato_sum_sq += (chunk_tomato ** 2).sum()
    total_count += len(chunk)

    # Process other variables
    for var in variables:
        valid_data = chunk[var].dropna() # Handle potential missing values, although readme says none
        sums[var] += valid_data.sum()
        sum_sq[var] += (valid_data ** 2).sum()
        sum_prod[var] += (valid_data * chunk_tomato[valid_data.index]).sum()
        counts[var] += len(valid_data)


# Calculate means and standard deviations
tomato_mean = tomato_sum / total_count
tomato_std = ((tomato_sum_sq / total_count) - (tomato_mean ** 2)) ** 0.5

means = {var: sums[var] / counts[var] if counts[var] > 0 else 0 for var in variables}
stds = {var: ((sum_sq[var] / counts[var]) - (means[var] ** 2)) ** 0.5 if counts[var] > 0 else 0 for var in variables}

# Calculate correlation coefficients and p-values
for var in variables:
    if stds[var] > 0 and tomato_std > 0:
        # Calculate covariance: E[(X-E[X])(Y-E[Y])] = E[XY] - E[X]E[Y]
        covariance = (sum_prod[var] / counts[var]) - (tomato_mean * means[var])
        correlation_coeffs[var] = covariance / (tomato_std * stds[var])

        # Perform a t-test to get the p-value for the correlation coefficient
        # The t-statistic for correlation is r * sqrt((n-2)/(1-r^2))
        n = counts[var]
        r = correlation_coeffs[var]
        if n > 2 and abs(r) < 1: # Check for valid n and r
             t_statistic = r * ((n - 2) / (1 - r ** 2)) ** 0.5
             p_values[var] = stats.t.sf(abs(t_statistic), n - 2) * 2 # Two-sided test - using survival function (1-cdf)

# Write the results to a file
with open('correlation_results.txt', 'w') as f:
    f.write("Variable,Correlation Coefficient,P-value\n")
    for var in variables:
        if var in correlation_coeffs:
            f.write(f"{var},{correlation_coeffs[var]},{p_values.get(var, 'N/A')}\n")
        else:
             f.write(f"{var},N/A,N/A\n")

print("Correlation analysis complete. Results written to correlation_results.txt")