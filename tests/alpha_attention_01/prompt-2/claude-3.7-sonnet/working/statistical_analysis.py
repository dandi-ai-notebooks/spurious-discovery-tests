import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
print("Merging datasets...")
merged_df = pd.merge(attention_df, synchrony_df, on='time')

# Get only the synchrony columns
synchrony_columns = [col for col in merged_df.columns if col.startswith('sync_')]
print(f"Number of synchrony pairs: {len(synchrony_columns)}")

# Create a results dataframe to store all statistical tests
results = []

print("\nPerforming statistical tests on all region pairs...")

# Perform correlation tests for each synchrony pair
for col in synchrony_columns:
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(merged_df[col], merged_df['attention_score'])
    
    # Spearman rank correlation
    spearman_r, spearman_p = stats.spearmanr(merged_df[col], merged_df['attention_score'])
    
    results.append({
        'synchrony_pair': col,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
    })

# Convert results to DataFrame and sort by absolute pearson correlation
results_df = pd.DataFrame(results)
results_df['abs_pearson_r'] = np.abs(results_df['pearson_r'])
results_df = results_df.sort_values('abs_pearson_r', ascending=False)

# Apply FDR correction for multiple testing
_, results_df['pearson_p_fdr'], _, _ = multipletests(
    results_df['pearson_p'], alpha=0.05, method='fdr_bh'
)
_, results_df['spearman_p_fdr'], _, _ = multipletests(
    results_df['spearman_p'], alpha=0.05, method='fdr_bh'
)

# Filter for significant correlations (FDR-corrected)
sig_results_df = results_df[results_df['pearson_p_fdr'] < 0.05].copy()
print(f"\nSignificant correlations after FDR correction: {len(sig_results_df)}")

# Display top 10 results
print("\nTop 10 neural synchrony pairs by correlation with attention:")
print(results_df[['synchrony_pair', 'pearson_r', 'pearson_p', 'pearson_p_fdr']].head(10))

# Plot the correlation coefficients for top pairs
top_pairs = results_df.head(20)['synchrony_pair'].tolist()

plt.figure(figsize=(12, 8))
plt.bar(
    range(len(top_pairs)),
    results_df.head(20)['pearson_r'],
    color=np.where(results_df.head(20)['pearson_p_fdr'] < 0.05, 'darkblue', 'lightblue')
)
plt.xticks(range(len(top_pairs)), top_pairs, rotation=90)
plt.axhline(y=0, color='r', linestyle='-')
plt.title("Correlation with Attention Score for Top 20 Region Pairs")
plt.xlabel("Region Pair")
plt.ylabel("Pearson Correlation Coefficient")
plt.tight_layout()
plt.savefig("figures/top20_correlations.png")

# Save results to CSV
results_df.to_csv("correlation_results.csv", index=False)
sig_results_df.to_csv("significant_correlations.csv", index=False)

# Extract the regions from the pairs
def extract_regions(sync_pair):
    # Extract the regions from format "sync_X_Y"
    parts = sync_pair.split('_')
    return int(parts[1]), int(parts[2])

# Extract regions and count their occurrences in significant pairs
region_counts = {}
for pair in sig_results_df['synchrony_pair']:
    r1, r2 = extract_regions(pair)
    region_counts[r1] = region_counts.get(r1, 0) + 1
    region_counts[r2] = region_counts.get(r2, 0) + 1

# Create the region counts dataframe and sort
region_counts_df = pd.DataFrame({
    'region': list(region_counts.keys()),
    'count': list(region_counts.values())
}).sort_values('count', ascending=False)

print("\nMost frequent regions in significant correlations:")
print(region_counts_df.head(5))

# Plot region frequency
plt.figure(figsize=(10, 6))
plt.bar(region_counts_df['region'], region_counts_df['count'], color='skyblue')
plt.title("Region Frequency in Significant Correlations")
plt.xlabel("Region")
plt.ylabel("Frequency")
plt.xticks(region_counts_df['region'])
plt.savefig("figures/region_frequency.png")

# Visualize the significant correlations as a network
if len(sig_results_df) > 0:
    try:
        import networkx as nx
        
        # Create a graph
        G = nx.Graph()
        
        # Add nodes for all regions
        for region in range(1, 17):
            G.add_node(region)
        
        # Add edges for significant correlations
        for _, row in sig_results_df.iterrows():
            r1, r2 = extract_regions(row['synchrony_pair'])
            G.add_edge(r1, r2, weight=abs(row['pearson_r']))
        
        # Draw the network
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)
        
        # Node size based on frequency
        node_size = []
        for node in G.nodes():
            if node in region_counts_df['region'].values:
                size = region_counts_df[region_counts_df['region'] == node]['count'].values[0] * 100
                node_size.append(size)
            else:
                node_size.append(100)
        
        # Edge width based on correlation
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue')
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7)
        
        plt.title("Network of Significantly Correlated Region Pairs")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("figures/correlation_network.png")
    except ImportError:
        print("\nNetworkx library not available, skipping network visualization")

print("\nStatistical analysis completed!")