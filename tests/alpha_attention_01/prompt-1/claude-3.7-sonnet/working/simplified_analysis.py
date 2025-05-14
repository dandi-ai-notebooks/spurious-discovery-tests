#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Create output directory for figures if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Load datasets
print("Loading data...")
attention_df = pd.read_csv("data/attention.csv")
synchrony_df = pd.read_csv("data/neural_synchrony.csv")

# Merge datasets
merged_df = pd.merge(attention_df, synchrony_df, on='time')
sync_columns = [col for col in synchrony_df.columns if col.startswith('sync_')]

print(f"Dataset has {len(merged_df)} time points and {len(sync_columns)} synchrony pairs")

##############################################
# 1. Key Region Analysis
##############################################
print("\n1. Analyzing key brain regions...")

# Extract region numbers from column names
def extract_regions(pair_name):
    parts = pair_name.replace('sync_', '').split('_')
    return int(parts[0]), int(parts[1])

# Count region frequency in positively correlated pairs
region_counts = {}
significant_positive_pairs = []

for col in sync_columns:
    corr, p_value = stats.pearsonr(merged_df['attention_score'], merged_df[col])
    if p_value < 0.05 and corr > 0:  # Significant positive correlation
        significant_positive_pairs.append((col, corr, p_value))
        r1, r2 = extract_regions(col)
        region_counts[r1] = region_counts.get(r1, 0) + 1
        region_counts[r2] = region_counts.get(r2, 0) + 1

# Sort by frequency
sorted_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)
top_regions = sorted_regions[:5]

print(f"Top 5 regions involved in significant positive correlations:")
for region, count in top_regions:
    print(f"Region {region}: involved in {count} significant positive correlations")

# Plot region importance
plt.figure(figsize=(12, 6))
regions, counts = zip(*sorted(region_counts.items(), key=lambda x: x[0]))
plt.bar(regions, counts)
plt.xlabel('Brain Region')
plt.ylabel('Number of Significant Positive Correlations')
plt.title('Brain Region Involvement in Attention-Related Synchrony')
plt.grid(True, alpha=0.3)
plt.savefig('figures/region_importance.png', dpi=300, bbox_inches='tight')
plt.close()

##############################################
# 2. Network Visualization (Simplified)
##############################################
print("\n2. Creating network visualization...")

# Create a graph of the strongest correlations
G = nx.Graph()

# Add nodes (brain regions)
for i in range(1, 17):
    G.add_node(i)

# Add edges with correlation strength as weight
for pair, corr, p_value in significant_positive_pairs:
    if corr > 0.1:  # Only include moderately correlated pairs
        r1, r2 = extract_regions(pair)
        G.add_edge(r1, r2, weight=corr)

# Calculate centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

# Combine centrality measures
centrality_df = pd.DataFrame({
    'Region': list(degree_centrality.keys()),
    'Degree': list(degree_centrality.values()),
    'Betweenness': list(betweenness_centrality.values()),
    'Eigenvector': list(eigenvector_centrality.values())
})
centrality_df = centrality_df.sort_values('Eigenvector', ascending=False)

print("Top 5 brain regions by eigenvector centrality:")
print(centrality_df.head(5))

# Plot the network (simplified)
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42)
edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
node_size = [degree_centrality[node] * 3000 for node in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=list(eigenvector_centrality.values()),
                       cmap=plt.cm.viridis, alpha=0.8)
nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

plt.title('Brain Region Network (Edge Width = Correlation Strength)')
plt.axis('off')
plt.savefig('figures/brain_network.png', dpi=300, bbox_inches='tight')
plt.close()

##############################################
# 3. Time-Lagged Analysis
##############################################
print("\n3. Performing time-lagged correlation analysis...")

# Select top 5 correlated synchrony pairs
top_corr_pairs = sorted(significant_positive_pairs, key=lambda x: x[1], reverse=True)[:5]
top_pair_names = [pair[0] for pair in top_corr_pairs]
print(f"Top correlated pairs: {top_pair_names}")

# Create lag analysis figure
lag_results = {}

for pair_name in top_pair_names:
    # Analyze different lags
    lag_correlations = []
    lags_to_test = list(range(-30, 31, 5))  # -30 to 30 in steps of 5

    for lag in lags_to_test:
        # Shift data for lag analysis
        if lag < 0:
            # Negative lag means attention leads synchrony
            synch_lagged = merged_df[pair_name].shift(-lag)
            valid_df = merged_df.copy()
            valid_df['shifted'] = synch_lagged
        else:
            # Positive lag means synchrony leads attention
            synch_lagged = merged_df[pair_name].shift(lag)
            valid_df = merged_df.copy()
            valid_df['shifted'] = synch_lagged

        # Drop NaN values created by shifting
        valid_df = valid_df.dropna()

        # Calculate correlation
        if len(valid_df) > 0:
            lag_corr, lag_p = stats.pearsonr(valid_df['shifted'], valid_df['attention_score'])
            lag_correlations.append((lag, lag_corr, lag_p))

    # Find lag with maximum absolute correlation
    max_lag_result = max(lag_correlations, key=lambda x: abs(x[1]))
    max_lag, max_corr, max_p = max_lag_result

    # Store results
    lag_results[pair_name] = (max_lag, max_corr, max_p)

    # Plot lag correlations
    lags, correlations, p_values = zip(*lag_correlations)

    plt.figure(figsize=(10, 6))
    plt.plot(lags, correlations, marker='o')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    plt.xlabel('Lag (seconds)')
    plt.ylabel('Correlation')
    plt.title(f'Lagged Correlation for {pair_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'figures/lag_correlation_{pair_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Report lag results
print("\nMaximum correlation lags (positive = synchrony leads attention):")
for pair, (lag, corr, p) in lag_results.items():
    print(f"{pair}: {lag} seconds lag, correlation = {corr:.4f} (p={p:.4e})")
    if p < 0.05:
        direction = "synchrony leads attention" if lag > 0 else "attention leads synchrony"
        print(f"  Significant correlation with {direction} by {abs(lag)} seconds")

##############################################
# 4. Regression Modeling
##############################################
print("\n4. Building predictive models...")

# Select features and target
X = merged_df[sync_columns].values
y = merged_df['attention_score'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4.1 Linear Regression
print("\n4.1 Linear Regression")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print(f"Linear Regression: R² = {lr_r2:.4f}, RMSE = {lr_rmse:.4f}")

# 4.2 Ridge Regression (L2 regularization)
print("\n4.2 Ridge Regression")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
ridge_r2 = r2_score(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print(f"Ridge Regression: R² = {ridge_r2:.4f}, RMSE = {ridge_rmse:.4f}")

# 4.3 Lasso Regression (L1 regularization - feature selection)
print("\n4.3 Lasso Regression (with feature selection)")
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
lasso_r2 = r2_score(y_test, y_pred_lasso)
lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print(f"Lasso Regression: R² = {lasso_r2:.4f}, RMSE = {lasso_rmse:.4f}")

# Get selected features from Lasso
lasso_coef = pd.DataFrame({
    'Synchrony Pair': sync_columns,
    'Coefficient': lasso.coef_
})
lasso_selected = lasso_coef[lasso_coef['Coefficient'] != 0].sort_values('Coefficient', ascending=False)
print(f"\nLasso selected {len(lasso_selected)} synchrony pairs")
print("Top 5 positive predictors:")
print(lasso_selected.head(5))

# Plot top coefficients
plt.figure(figsize=(12, 6))
top_n = 10
top_coef = lasso_coef.nlargest(top_n, 'Coefficient')
bot_coef = lasso_coef.nsmallest(top_n, 'Coefficient')
important_coef = pd.concat([top_coef, bot_coef])

sns.barplot(data=important_coef, x='Coefficient', y='Synchrony Pair')
plt.title(f'Top {top_n} Positive and Negative Lasso Coefficients')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/lasso_coefficients.png', dpi=300, bbox_inches='tight')
plt.close()

# 4.4 Random Forest (non-linear relationships)
print("\n4.4 Random Forest Regression")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"Random Forest: R² = {rf_r2:.4f}, RMSE = {rf_rmse:.4f}")

# Get feature importances
feature_importance = pd.DataFrame({
    'Synchrony Pair': sync_columns,
    'Importance': rf.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\nTop 5 most important features (Random Forest):")
print(feature_importance.head(5))

# Plot feature importances
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(20), x='Importance', y='Synchrony Pair')
plt.title('Top 20 Synchrony Pairs by Random Forest Importance')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Model Comparison
plt.figure(figsize=(10, 6))
models = ['Linear Regression', 'Ridge', 'Lasso', 'Random Forest']
r2_scores = [lr_r2, ridge_r2, lasso_r2, rf_r2]
rmse_scores = [lr_rmse, ridge_rmse, lasso_rmse, rf_rmse]

plt.subplot(1, 2, 1)
plt.bar(models, r2_scores)
plt.title('R² Comparison')
plt.ylabel('R² Score')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(models, rmse_scores)
plt.title('RMSE Comparison')
plt.ylabel('RMSE')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

##############################################
# 5. Time-segment Analysis
##############################################
print("\n5. Analyzing attention profiles over different time segments...")

# Define time segments
n_segments = 3
segment_size = len(merged_df) // n_segments
segments = []

for i in range(n_segments):
    start_idx = i * segment_size
    end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(merged_df)
    segments.append(merged_df.iloc[start_idx:end_idx])

# Analyze correlations in each segment
segment_results = []

for i, segment_df in enumerate(segments):
    seg_correlations = []
    for col in sync_columns:
        corr, p_value = stats.pearsonr(segment_df['attention_score'], segment_df[col])
        if p_value < 0.05:  # Significant correlation
            seg_correlations.append((col, corr, p_value))

    # Sort by correlation strength
    seg_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    top_pairs = seg_correlations[:5]

    segment_results.append({
        'segment': i+1,
        'start_time': segment_df['time'].min(),
        'end_time': segment_df['time'].max(),
        'mean_attention': segment_df['attention_score'].mean(),
        'std_attention': segment_df['attention_score'].std(),
        'n_significant': len(seg_correlations),
        'top_pairs': top_pairs
    })

# Report segment analysis
for seg in segment_results:
    print(f"\nSegment {seg['segment']} (Time {seg['start_time']}-{seg['end_time']}s):")
    print(f"Mean attention: {seg['mean_attention']:.4f} ± {seg['std_attention']:.4f}")
    print(f"Significant correlations: {seg['n_significant']}/{len(sync_columns)}")
    print("Top 5 correlations:")
    for pair, corr, p_value in seg['top_pairs']:
        print(f"  {pair}: r = {corr:.4f} (p = {p_value:.4e})")

# Plot segment attention profiles
plt.figure(figsize=(12, 6))
for i, seg in enumerate(segment_results):
    segment_df = segments[i]
    plt.plot(segment_df['time'], segment_df['attention_score'], label=f"Segment {i+1}")

plt.xlabel('Time (seconds)')
plt.ylabel('Attention Score')
plt.title('Attention Profiles Across Time Segments')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('figures/segment_profiles.png', dpi=300, bbox_inches='tight')
plt.close()

# Compare segment distributions
plt.figure(figsize=(10, 6))
for i, segment_df in enumerate(segments):
    sns.kdeplot(segment_df['attention_score'], label=f"Segment {i+1}")
plt.xlabel('Attention Score')
plt.ylabel('Density')
plt.title('Distribution of Attention Scores Across Segments')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('figures/segment_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# Test for significant differences between segments
print("\nStatistical tests for differences between segments:")
for i in range(len(segments)-1):
    for j in range(i+1, len(segments)):
        t_stat, p_value = stats.ttest_ind(
            segments[i]['attention_score'],
            segments[j]['attention_score'],
            equal_var=False  # Welch's t-test
        )
        print(f"Segment {i+1} vs Segment {j+1}: t={t_stat:.4f}, p={p_value:.4e}")
        if p_value < 0.05:
            print(f"  Significant difference in attention levels")

print("\nAnalysis complete!")