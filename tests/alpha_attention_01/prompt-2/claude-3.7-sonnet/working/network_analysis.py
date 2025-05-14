import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import networkx for network analysis
try:
    import networkx as nx
    networkx_available = True
except ImportError:
    print("NetworkX not available. Some visualizations will be skipped.")
    networkx_available = False

# Set up output directory for figures
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

print("Loading data...")
attention_df = pd.read_csv("data/attention.csv")
synchrony_df = pd.read_csv("data/neural_synchrony.csv")

# Merge datasets
merged_df = pd.merge(attention_df, synchrony_df, on='time')

# Get synchrony columns
synchrony_columns = [col for col in merged_df.columns if col.startswith('sync_')]
print(f"Number of synchrony pairs: {len(synchrony_columns)}")

# Extract regions from sync pair name
def extract_regions(sync_pair):
    parts = sync_pair.split('_')
    return int(parts[1]), int(parts[2])

# Calculate the average synchrony for each pair across all time points
avg_synchrony = {}
for col in synchrony_columns:
    avg_synchrony[col] = synchrony_df[col].mean()
    
# Prepare data for correlation analysis
all_correlations = {}
for col in synchrony_columns:
    corr, p_val = stats.pearsonr(merged_df['attention_score'], merged_df[col])
    all_correlations[col] = (corr, p_val)
    
# Create adjacency matrix for average synchrony
regions = range(1, 17)  # 16 regions
adj_matrix = np.zeros((16, 16))
avg_sync_matrix = pd.DataFrame(adj_matrix, index=regions, columns=regions)

# Fill the adjacency matrix with average synchrony values
for col, avg_sync in avg_synchrony.items():
    r1, r2 = extract_regions(col)
    avg_sync_matrix.loc[r1, r2] = avg_sync
    avg_sync_matrix.loc[r2, r1] = avg_sync  # Make it symmetric

# Fill the adjacency matrix with correlation values (use absolute values)
correlation_matrix = pd.DataFrame(adj_matrix, index=regions, columns=regions)
for col, (corr, _) in all_correlations.items():
    r1, r2 = extract_regions(col)
    correlation_matrix.loc[r1, r2] = abs(corr)
    correlation_matrix.loc[r2, r1] = abs(corr)  # Make it symmetric

# Plot heatmap of average synchrony
plt.figure(figsize=(12, 10))
sns.heatmap(avg_sync_matrix, annot=False, cmap="YlGnBu", vmin=0, vmax=1)
plt.title("Average Neural Synchrony Between Brain Regions")
plt.savefig("figures/average_synchrony_matrix.png")

# Plot heatmap of correlations with attention
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap="YlOrRd", vmin=0, vmax=0.3)
plt.title("Correlation of Neural Synchrony with Attention")
plt.savefig("figures/correlation_with_attention_matrix.png")

# Network analysis
if networkx_available:
    # Create network for average synchrony
    G_avg = nx.Graph()
    
    # Add nodes
    for i in regions:
        G_avg.add_node(i)
        
    # Add edges with weights based on average synchrony
    for col, avg_sync in avg_synchrony.items():
        r1, r2 = extract_regions(col)
        G_avg.add_edge(r1, r2, weight=avg_sync)
    
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G_avg)
    closeness_centrality = nx.closeness_centrality(G_avg, distance='weight')
    betweenness_centrality = nx.betweenness_centrality(G_avg, weight='weight')
    eigenvector_centrality = nx.eigenvector_centrality(G_avg, weight='weight', max_iter=1000)
    
    # Create dataframe of centrality measures
    centrality_df = pd.DataFrame({
        'region': list(regions),
        'degree': [degree_centrality[r] for r in regions],
        'closeness': [closeness_centrality[r] for r in regions],
        'betweenness': [betweenness_centrality[r] for r in regions],
        'eigenvector': [eigenvector_centrality[r] for r in regions]
    })
    
    # Save centrality measures
    centrality_df.to_csv("centrality_measures.csv", index=False)
    
    # Print top regions by different centrality measures
    print("\nTop 5 regions by degree centrality:")
    print(centrality_df.sort_values('degree', ascending=False).head(5)[['region', 'degree']])
    
    print("\nTop 5 regions by betweenness centrality:")
    print(centrality_df.sort_values('betweenness', ascending=False).head(5)[['region', 'betweenness']])
    
    print("\nTop 5 regions by eigenvector centrality:")
    print(centrality_df.sort_values('eigenvector', ascending=False).head(5)[['region', 'eigenvector']])
    
    # Create correlation network
    significant_threshold = 0.05
    corr_threshold = 0.15  # Minimum correlation to include
    
    G_corr = nx.Graph()
    
    # Add nodes
    for i in regions:
        G_corr.add_node(i)
    
    # Add edges with weights based on correlation with attention
    for col, (corr, p_val) in all_correlations.items():
        if p_val < significant_threshold and abs(corr) >= corr_threshold:
            r1, r2 = extract_regions(col)
            G_corr.add_edge(r1, r2, weight=abs(corr), actual_corr=corr)
    
    # Visualize the average synchrony network
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G_avg, seed=42)
    
    # Node size based on eigenvector centrality
    node_size = [eigenvector_centrality[r] * 5000 for r in G_avg.nodes()]
    
    # Edge width based on synchrony weight
    edge_width = [G_avg[u][v]['weight'] * 3 for u, v in G_avg.edges()]
    
    nx.draw_networkx_nodes(G_avg, pos, node_size=node_size, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_labels(G_avg, pos, font_size=10)
    nx.draw_networkx_edges(G_avg, pos, width=edge_width, edge_color='gray', alpha=0.5)
    
    plt.title("Neural Synchrony Network (node size = eigenvector centrality)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("figures/synchrony_network.png")
    
    # Visualize the correlation network
    if len(G_corr.edges()) > 0:
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G_corr, seed=42)
        
        # Node size based on degree in this network
        node_size = [G_corr.degree(r) * 300 for r in G_corr.nodes()]
        
        # Edge colors based on actual correlation (positive or negative)
        edge_color = ['red' if G_corr[u][v]['actual_corr'] < 0 else 'blue' 
                     for u, v in G_corr.edges()]
        
        # Edge width based on correlation magnitude
        edge_width = [G_corr[u][v]['weight'] * 10 for u, v in G_corr.edges()]
        
        nx.draw_networkx_nodes(G_corr, pos, node_size=node_size, node_color='yellow', alpha=0.8)
        nx.draw_networkx_labels(G_corr, pos, font_size=10)
        nx.draw_networkx_edges(G_corr, pos, width=edge_width, edge_color=edge_color, alpha=0.7)
        
        plt.title("Significant Correlations with Attention Network")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("figures/correlation_network.png")
    
    # Analyze how centrality measures relate to involvement in attention
    # For each region, calculate the average absolute correlation with attention
    region_attention_involvement = {}
    for region in regions:
        # Find all pairs involving this region
        region_pairs = [col for col in synchrony_columns 
                       if extract_regions(col)[0] == region or extract_regions(col)[1] == region]
        
        # Calculate average absolute correlation
        avg_abs_corr = np.mean([abs(all_correlations[pair][0]) for pair in region_pairs])
        region_attention_involvement[region] = avg_abs_corr
    
    # Add to centrality dataframe
    centrality_df['avg_attention_correlation'] = [region_attention_involvement[r] for r in regions]
    
    # Calculate correlation between centrality measures and attention involvement
    for measure in ['degree', 'closeness', 'betweenness', 'eigenvector']:
        corr, p = stats.pearsonr(centrality_df[measure], centrality_df['avg_attention_correlation'])
        print(f"\nCorrelation between {measure} centrality and attention involvement: r={corr:.3f}, p={p:.3f}")
    
    # Plot the relationship between eigenvector centrality and attention involvement
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='eigenvector', y='avg_attention_correlation', data=centrality_df)
    
    # Add region labels to the plot
    for i, row in centrality_df.iterrows():
        plt.text(row['eigenvector'], row['avg_attention_correlation'], str(int(row['region'])))
        
    plt.title("Relationship Between Eigenvector Centrality and Attention Involvement")
    plt.xlabel("Eigenvector Centrality")
    plt.ylabel("Average Correlation with Attention")
    plt.grid(alpha=0.3)
    plt.savefig("figures/centrality_vs_attention.png")

print("\nNetwork analysis completed!")