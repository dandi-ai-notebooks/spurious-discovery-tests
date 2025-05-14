import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load attention data
try:
    attention_df = pd.read_csv("data/attention.csv")
    print("Attention data loaded successfully.")
except Exception as e:
    print(f"Error loading attention data: {e}")
    exit()

# Prepare to process synchrony data in chunks
synchrony_file_path = "neural_synchrony.csv"
chunk_size = 500 # Define a manageable chunk size

# Initialize lists to store features and target from chunks
X_chunks = []
y_chunks = []

print(f"Processing synchrony data in chunks of {chunk_size}...")

# Iterate through chunks of synchrony data
try:
    for i, synchrony_chunk in enumerate(pd.read_csv(synchrony_file_path, chunksize=chunk_size)):
        print(f"Processing chunk {i+1}")
        # Merge with attention data
        merged_chunk = pd.merge(attention_df, synchrony_chunk, on='time', how='inner')

        # Extract features (synchrony columns) and target (attention_score)
        # Assuming columns other than 'time' and 'attention_score' in merged_chunk are features
        feature_columns = [col for col in merged_chunk.columns if col.startswith('sync_')]
        X_chunk = merged_chunk[feature_columns]
        y_chunk = merged_chunk['attention_score']

        X_chunks.append(X_chunk)
        y_chunks.append(y_chunk)

        print(f"Chunk {i+1} processed. Features shape: {X_chunk.shape}, Target shape: {y_chunk.shape}")


except Exception as e:
    print(f"Error processing synchrony data in chunks: {e}")
    exit()


# Concatenate chunks
try:
    X = pd.concat(X_chunks, ignore_index=True)
    y = pd.concat(y_chunks, ignore_index=True)
    print(f"All chunks concatenated. Full data shapes - Features: {X.shape}, Target: {y.shape}")
except Exception as e:
    print(f"Error concatenating chunks: {e}")
    exit()

# Train Linear Regression model
print("Training Linear Regression model...")
try:
    model = LinearRegression()
    model.fit(X, y)
    print("Model training complete.")
except Exception as e:
    print(f"Error training model: {e}")
    exit()

# Analyze results
print("\nAnalyzing model results:")

# Get coefficients and corresponding feature names
coefficients = model.coef_
feature_names = X.columns

# Sort coefficients by absolute value to find most influential features
sorted_indices = np.argsort(np.abs(coefficients))[::-1]
sorted_coefficients = coefficients[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]

print("\nTop 10 most influential synchrony features (absolute coefficient value):")
for i in range(min(10, len(sorted_feature_names))):
    print(f"{sorted_feature_names[i]}: {sorted_coefficients[i]:.6f}")

# Evaluate model (R-squared)
r_squared = model.score(X, y)
print(f"\nModel R-squared: {r_squared:.6f}")

# Note: For rigorous statistical justification, one would typically perform
# train/test split, cross-validation, and look at p-values of coefficients.
# LinearRegression in scikit-learn does not provide p-values directly.
# To get p-values, one could use statsmodels library.
# For this task, we'll use the magnitude of coefficients as an indicator of importance
# and R-squared as a measure of overall predictiveness, as a starting point.
# A significant R-squared (tested e.g., with an F-test, not provided by sk learn directly)
# would statistically justify that synchrony predicts attention.
# The magnitude of coefficients indicates which pairs are 'more informative'.