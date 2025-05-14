import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from statsmodels.stats.multitest import multipletests
from pathlib import Path

# Set up output directory for figures
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

print("Loading attention data...")
attention_df = pd.read_csv("data/attention.csv")
print("Loading neural synchrony data...")
synchrony_df = pd.read_csv("data/neural_synchrony.csv")

# Merge datasets
merged_df = pd.merge(attention_df, synchrony_df, on='time')

# Get synchrony columns
synchrony_columns = [col for col in merged_df.columns if col.startswith('sync_')]
print(f"Number of synchrony pairs: {len(synchrony_columns)}")

# Function to compute cross-correlation with time lag
def cross_corr_with_lag(x, y, lag_max=50):
    """
    Calculate cross-correlation with time lag.
    positive lag means x leads y, negative lag means y leads x
    """
    corrs = []
    lags = range(-lag_max, lag_max + 1)
    
    for lag in lags:
        if lag > 0:
            # x leads y (positive lag)
            corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
        elif lag < 0:
            # y leads x (negative lag)
            corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
        else:
            # No lag
            corr = np.corrcoef(x, y)[0, 1]
        
        corrs.append(corr)
    
    return lags, corrs

# Analyze top 10 synchrony pairs by absolute correlation
# Use the results from initial exploration
top_pairs = [
    'sync_2_16', 'sync_11_16', 'sync_12_16', 'sync_8_16', 'sync_2_11',
    'sync_10_16', 'sync_5_15', 'sync_5_13', 'sync_13_15', 'sync_2_8'
]

# Plot cross-correlation with time lag for each top pair
plt.figure(figsize=(15, 10))
max_lags = {}

for i, pair in enumerate(top_pairs, 1):
    # Calculate cross-correlation with time lag
    lags, corrs = cross_corr_with_lag(
        merged_df[pair].values, 
        merged_df['attention_score'].values, 
        lag_max=60  # up to 1 minute lag
    )
    
    # Find the lag with maximum absolute correlation
    max_corr_lag = lags[np.argmax(np.abs(corrs))]
    max_corr = corrs[np.argmax(np.abs(corrs))]
    max_lags[pair] = (max_corr_lag, max_corr)
    
    # Plot cross-correlation
    plt.subplot(4, 3, i)
    plt.plot(lags, corrs)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=max_corr_lag, color='g', linestyle='--')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title(f"{pair} (Max at lag {max_corr_lag}s, r={max_corr:.3f})")
    plt.xlabel("Lag (seconds)")
    plt.ylabel("Cross-correlation")
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/cross_correlation_lags.png")

# Create a summary of maximum lag results
max_lag_df = pd.DataFrame([
    {"pair": pair, "max_lag": lag, "correlation": corr}
    for pair, (lag, corr) in max_lags.items()
]).sort_values("max_lag")

print("\nLag analysis for top pairs:")
print(max_lag_df)

# Plot the max lag summary
plt.figure(figsize=(12, 6))
bars = plt.bar(
    max_lag_df['pair'], 
    max_lag_df['max_lag'],
    color=np.where(max_lag_df['max_lag'] < 0, 'tomato', 'skyblue')
)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title("Time Lag with Maximum Correlation (seconds)")
plt.xlabel("Region Pair")
plt.ylabel("Lag (seconds)")
plt.xticks(rotation=45)

# Add correlation values as text
for i, bar in enumerate(bars):
    height = bar.get_height()
    corr = max_lag_df.iloc[i]['correlation']
    y_pos = height + 0.5 if height > 0 else height - 2.5
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        y_pos,
        f"r={corr:.2f}",
        ha='center'
    )

plt.tight_layout()
plt.savefig("figures/maximum_lag_summary.png")

# Perform a more detailed analysis of lagged correlations for a few key pairs
# Select pairs that have the most interesting lag profiles
interesting_pairs = max_lag_df.sort_values(by='correlation', ascending=False).head(3)['pair'].tolist()

# Create time-shifted versions of these pairs
lag_range = range(-10, 11, 1)  # -10 to 10 seconds lag
lagged_corrs = {}

for pair in interesting_pairs:
    pair_corrs = {}
    for lag in lag_range:
        if lag > 0:
            # Synchrony leads attention
            shifted_sync = merged_df[pair][lag:].reset_index(drop=True)
            shifted_attn = merged_df['attention_score'][:-lag].reset_index(drop=True)
        elif lag < 0:
            # Attention leads synchrony
            shifted_sync = merged_df[pair][:lag].reset_index(drop=True)
            shifted_attn = merged_df['attention_score'][-lag:].reset_index(drop=True)
        else:
            # No lag
            shifted_sync = merged_df[pair]
            shifted_attn = merged_df['attention_score']
        
        # Calculate correlation and statistical significance
        corr, p_value = stats.pearsonr(shifted_sync, shifted_attn)
        pair_corrs[lag] = (corr, p_value)
    
    lagged_corrs[pair] = pair_corrs

# Plot the lagged correlations for each pair
plt.figure(figsize=(15, 5))
for i, pair in enumerate(interesting_pairs, 1):
    plt.subplot(1, 3, i)
    
    lags = list(lagged_corrs[pair].keys())
    corrs = [lagged_corrs[pair][lag][0] for lag in lags]
    p_values = [lagged_corrs[pair][lag][1] for lag in lags]
    
    # Plot correlation at each lag
    bars = plt.bar(
        lags, 
        corrs, 
        color=['darkblue' if p < 0.05 else 'lightblue' for p in p_values]
    )
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title(f"Lagged Correlations for {pair}")
    plt.xlabel("Lag (seconds, + = sync leads)")
    plt.ylabel("Correlation")
    plt.grid(alpha=0.3)
    
plt.tight_layout()
plt.savefig("figures/lagged_correlations_detail.png")

# For the most interesting pair, plot the actual time series with optimal lag
best_pair = max_lag_df.iloc[np.argmax(np.abs(max_lag_df['correlation']))]['pair']
best_lag = max_lag_df.iloc[np.argmax(np.abs(max_lag_df['correlation']))]['max_lag']

plt.figure(figsize=(15, 6))
plt.title(f"Time Series with Optimal Lag ({best_pair}, lag={best_lag}s)")

# Apply the optimal lag
if best_lag > 0:
    # Synchrony leads attention
    x1 = merged_df['time'][best_lag:]
    y1 = merged_df[best_pair][:-best_lag]
    x2 = merged_df['time'][:-best_lag]
    y2 = merged_df['attention_score'][:-best_lag]
    plt.plot(x1, y1, label=f"{best_pair} (shifted by {best_lag}s)")
elif best_lag < 0:
    # Attention leads synchrony
    x1 = merged_df['time'][:-abs(best_lag)]
    y1 = merged_df[best_pair][:-abs(best_lag)]
    x2 = merged_df['time'][abs(best_lag):]
    y2 = merged_df['attention_score'][:-abs(best_lag)]
    plt.plot(x1, y1, label=f"{best_pair} (shifted by {best_lag}s)")
else:  # best_lag == 0
    # No lag
    x1 = merged_df['time']
    y1 = merged_df[best_pair]
    x2 = merged_df['time']
    y2 = merged_df['attention_score']
    plt.plot(x1, y1, label=f"{best_pair} (no shift)")

plt.plot(x2, y2, label='attention_score', alpha=0.7)
plt.xlabel("Time (seconds)")
plt.ylabel("Value")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figures/optimal_lag_time_series.png")

print("\nTime lag analysis completed!")