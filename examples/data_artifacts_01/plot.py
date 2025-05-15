import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('data/neural_firing_rates.csv')

# Create figure with two subplots
plt.figure(figsize=(15, 8))

# Convert time to hours for better readability
time_hours = data['time_seconds'] / 3600

# Plot Region A
plt.subplot(2, 1, 1)
plt.plot(time_hours, data['region_a_firing_rate'], 'b-', label='Region A', alpha=0.8)
plt.ylabel('Firing Rate (spikes/sec)')
plt.title('Neural Activity Time Series')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot Region B
plt.subplot(2, 1, 2)
plt.plot(time_hours, data['region_b_firing_rate'], 'r-', label='Region B', alpha=0.8)
plt.xlabel('Time (hours)')
plt.ylabel('Firing Rate (spikes/sec)')
plt.legend()
plt.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
