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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
import networkx as nx
from statsmodels.tsa.stattools import ccf
import statsmodels.api as sm
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
# 2. Network Analysis
##############################################
print("\n2. Performing network analysis...")

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

# Plot the network
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42)
edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
node_size = [degree_centrality[node] * 3000 for node in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=list(eigenvector_centrality.values()),
                       cmap=plt.cm.viridis, alpha=0.8)
nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

plt.title('Brain Region Network (Edge Width = Correlation Strength)')
# Create a ScalarMappable with the appropriate colormap
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
sm.set_array([])
plt.colorbar(sm, label='Eigenvector Centrality')
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

# Create figure for cross-correlation
plt.figure(figsize=(14, 8))

max_lags = 60  # 1 minute in seconds
lag_results = {}

for pair_name in top_pair_names:
    # Calculate cross-correlation function
    pair_data = merged_df[pair_name].values
    attention_data = merged_df['attention_score'].values

    # Compute cross-correlation
    cross_corr = ccf(pair_data, attention_data, adjusted=False)
    lags = np.arange(-max_lags, max_lags + 1)

    # Only take the central part of the CCF up to max_lags in each direction
    mid_idx = len(cross_corr) // 2
    cross_corr_subset = cross_corr[mid_idx - max_lags:mid_idx + max_lags + 1]

    # Find lag with maximum correlation
    max_corr_idx = np.argmax(np.abs(cross_corr_subset))
    max_lag = lags[max_corr_idx]
    max_corr = cross_corr_subset[max_corr_idx]

    # Store results
    lag_results[pair_name] = (max_lag, max_corr)

    # Plot
    plt.plot(lags, cross_corr_subset, label=f"{pair_name} (Max at lag={max_lag}s)")

plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Lag (seconds)')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation between Synchrony Pairs and Attention')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('figures/cross_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

# Report lag results
print("Maximum correlation lags (positive = synchrony leads attention):")
for pair, (lag, corr) in lag_results.items():
    print(f"{pair}: {lag} seconds lag, correlation = {corr:.4f}")

# Perform statistical test on the lag
for pair_name, (max_lag, _) in lag_results.items():
    if max_lag != 0:
        # Apply the lag to the synchrony data
        synch_lagged = merged_df[pair_name].shift(max_lag)
        # Drop NaN values created by shifting
        valid_data = merged_df.copy()
        valid_data['synch_lagged'] = synch_lagged
        valid_data = valid_data.dropna()

        # Calculate correlation with the lagged data
        lagged_corr, lagged_p = stats.pearsonr(valid_data['synch_lagged'], valid_data['attention_score'])

        # Calculate correlation with the original (no lag) data using the same data points
        orig_corr, orig_p = stats.pearsonr(valid_data[pair_name], valid_data['attention_score'])

        # Test if the lag significantly improves correlation
        print(f"{pair_name}: Original corr = {orig_corr:.4f} (p={orig_p:.4e}), Lagged corr = {lagged_corr:.4f} (p={lagged_p:.4e})")

        if lagged_p < 0.05 and lagged_corr > orig_corr:
            print(f"  Significant improvement with lag of {max_lag} seconds")

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

##############################################
# 6. Lagged Regression Analysis
##############################################
print("\n6. Lagged Regression Analysis...")

# Select top synchrony pairs
top_pairs = [pair[0] for pair in top_corr_pairs[:3]]
lag_regression_results = []

for pair in top_pairs:
    print(f"\nAnalyzing lagged relationships for {pair}...")

    # Analyze different lags
    for lag in [1, 5, 10, 20]:
        # Create lagged synchrony feature
        lagged_synch = merged_df[pair].shift(lag)
        lagged_df = merged_df.copy()
        lagged_df['lagged_synch'] = lagged_synch
        lagged_df = lagged_df.dropna()

        # Calculate correlation with lagged data
        lagged_corr, lagged_p = stats.pearsonr(lagged_df['lagged_synch'], lagged_df['attention_score'])

        # Calculate correlation with original (no lag) data using the same data points
        orig_corr, orig_p = stats.pearsonr(lagged_df[pair], lagged_df['attention_score'])

        # Compare models
        lag_improvement = lagged_corr - orig_corr
        significant = lagged_p < 0.05 and lag_improvement > 0

        # Store results
        lag_regression_results.append({
            'pair': pair,
            'lag': lag,
            'orig_corr': orig_corr,
            'lagged_corr': lagged_corr,
            'improvement': lag_improvement,
            'p_value': lagged_p,
            'significant': significant
        })

        print(f"Lag {lag}s: Original r={orig_corr:.4f}, Lagged r={lagged_corr:.4f}, Improvement: {lag_improvement:.4f}")
        if significant:
            print(f"  Significant improvement with lag of {lag} seconds (p={lagged_p:.4e})")

# Test reverse direction (does attention predict future synchrony?)
print("\nTesting if attention predicts future synchrony...")
for pair in top_pairs:
    # Test with attention leading synchrony by 5 seconds
    lag = 5
    lagged_attention = merged_df['attention_score'].shift(lag)
    lagged_df = merged_df.copy()
    lagged_df['lagged_attention'] = lagged_attention
    lagged_df = lagged_df.dropna()

    # Calculate correlation
    reverse_corr, reverse_p = stats.pearsonr(lagged_df['lagged_attention'], lagged_df[pair])
    print(f"{pair}: Corr = {reverse_corr:.4f} (p={reverse_p:.4e})")
    if reverse_p < 0.05:
        print(f"  Significant correlation when attention leads by {lag} seconds")

print("\nAdvanced analysis complete!")