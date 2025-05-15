import numpy as np
import pandas as pd
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

def generate_base_signals(n_samples):
    """Generate two independent signals between 10-20 Hz"""
    signal1 = np.random.uniform(10, 20, n_samples)
    signal2 = np.random.uniform(10, 20, n_samples)
    return signal1, signal2

def create_zero_chunks(n_samples, chunk_size=300, n_chunks=10):
    """Create binary mask for zero chunks"""
    mask = np.ones(n_samples)
    # Randomly place chunks of zeros
    for _ in range(n_chunks):
        start = np.random.randint(0, n_samples - chunk_size)
        mask[start:start + chunk_size] = 0
    return mask

# Parameters
n_samples = 21600  # 6 hours * 3600 seconds/hour
chunk_size = 300   # 5 minute chunks
n_chunks = 10      # Number of zero chunks

# Generate base signals (iid between 10-20 Hz)
region_a_base, region_b_base = generate_base_signals(n_samples)

# Create zero masks (same for both signals)
zero_mask = create_zero_chunks(n_samples, chunk_size, n_chunks)

# Apply masks
region_a = region_a_base * zero_mask
region_b = region_b_base * zero_mask

# Ensure all values are non-negative
region_a = np.maximum(region_a, 0)
region_b = np.maximum(region_b, 0)

# Create DataFrame
data = pd.DataFrame({
    'time_seconds': np.arange(n_samples),
    'region_a_firing_rate': region_a,
    'region_b_firing_rate': region_b
})

# Save to CSV
data.to_csv('data/neural_firing_rates.csv', index=False)

# Print dataset characteristics
print("\nGenerated dataset characteristics:")
print(f"Number of time points: {n_samples}")
print(f"Number of zero chunks: {n_chunks}")
print(f"Chunk size: {chunk_size} seconds")

# Calculate correlation excluding zero periods
non_zero_mask = zero_mask == 1
correlation = np.corrcoef(
    region_a[non_zero_mask],
    region_b[non_zero_mask]
)[0, 1]

print(f"\nCorrelation (excluding zero periods): {correlation:.3f}")

# Calculate percentage of zero points
zero_percentage = (1 - zero_mask.mean()) * 100
print(f"Percentage of time points with zeros: {zero_percentage:.1f}%")

# Basic statistics
print("\nRegion A Statistics:")
print(f"Mean (non-zero): {np.mean(region_a[region_a > 0]):.2f}")
print(f"Max: {np.max(region_a):.2f}")

print("\nRegion B Statistics:")
print(f"Mean (non-zero): {np.mean(region_b[region_b > 0]):.2f}")
print(f"Max: {np.max(region_b):.2f}")
