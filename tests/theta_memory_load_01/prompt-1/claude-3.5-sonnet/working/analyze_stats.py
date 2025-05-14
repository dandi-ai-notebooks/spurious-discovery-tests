import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def compute_fdr_correlations(X, y, electrode_names, alpha=0.05):
    """Compute correlations with FDR control"""
    corrs = []
    pvals = []
    for col in X.columns:
        r, p = pearsonr(X[col], y)
        corrs.append(r)
        pvals.append(p)
    
    # FDR correction
    reject, pvals_adj, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    
    return pd.DataFrame({
        'electrode': electrode_names,
        'correlation': corrs,
        'p_value': pvals,
        'p_value_adj': pvals_adj,
        'significant': reject
    })

def compute_sliding_correlation(x, y, window=100):
    """Compute correlation in sliding window"""
    correlations = []
    for i in range(len(x) - window):
        r, _ = pearsonr(x[i:i+window], y[i:i+window])
        correlations.append(r)
    return np.array(correlations)

def main():
    ensure_dir('plots')
    print("Loading data...")
    
    # Load and merge data
    memory_df = pd.read_csv("data/memory_load.csv")
    theta_df = pd.read_csv("data/theta_power.csv")
    merged_df = memory_df.merge(theta_df, on="time")
    
    # Get theta columns
    theta_cols = [col for col in theta_df.columns if col.startswith('theta_')]
    
    # 1. FDR-controlled correlation analysis
    print("\nPerforming FDR-controlled correlation analysis...")
    correlation_results = compute_fdr_correlations(
        merged_df[theta_cols], 
        merged_df['wm_load'],
        theta_cols
    )
    
    # Plot correlation results
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(theta_cols)), correlation_results['correlation'])
    # Color significant correlations differently
    for i, significant in enumerate(correlation_results['significant']):
        bars[i].set_color('red' if significant else 'gray')
    plt.xticks(range(len(theta_cols)), theta_cols, rotation=45)
    plt.xlabel('Electrode')
    plt.ylabel('Correlation with WM Load')
    plt.title('FDR-Controlled Correlations (red = significant)')
    plt.tight_layout()
    plt.savefig('plots/fdr_correlations.png')
    plt.close()
    
    # 2. Time-varying correlation analysis
    print("\nAnalyzing temporal stability of correlations...")
    window_size = 200  # 100 seconds given 0.5s sampling
    sliding_corr = compute_sliding_correlation(
        merged_df['theta_Fz'],  # Use Fz as it showed strongest correlation
        merged_df['wm_load'],
        window_size
    )
    
    plt.figure(figsize=(12, 6))
    plt.plot(merged_df['time'][:-window_size], sliding_corr)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Correlation Coefficient')
    plt.title(f'Sliding Window Correlation (window = {window_size/2}s)')
    plt.grid(True)
    plt.savefig('plots/temporal_stability.png')
    plt.close()
    
    # 3. Cross-validation of relationship stability
    print("\nPerforming cross-validation analysis...")
    n_splits = 5
    split_size = len(merged_df) // n_splits
    cv_correlations = []
    
    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size
        test_slice = slice(start_idx, end_idx)
        
        # Calculate correlation on this split
        r, _ = pearsonr(
            merged_df['wm_load'][test_slice],
            merged_df['theta_Fz'][test_slice]
        )
        cv_correlations.append(r)
    
    # Calculate confidence interval
    ci_lower = np.percentile(cv_correlations, 2.5)
    ci_upper = np.percentile(cv_correlations, 97.5)
    
    # Save comprehensive results
    print("\nSaving results...")
    with open('statistical_analysis.md', 'w') as f:
        f.write("# Statistical Analysis Results\n\n")
        
        f.write("## FDR-Controlled Correlations\n")
        f.write("| Electrode | Correlation | P-value | Adjusted P-value | Significant |\n")
        f.write("|-----------|-------------|---------|-----------------|-------------|\n")
        for _, row in correlation_results.iterrows():
            f.write(f"| {row['electrode']} | {row['correlation']:.3f} | {row['p_value']:.6f} | "
                   f"{row['p_value_adj']:.6f} | {row['significant']} |\n")
        
        f.write("\n## Cross-validation Results\n")
        f.write(f"- Mean correlation: {np.mean(cv_correlations):.3f}\n")
        f.write(f"- Standard deviation: {np.std(cv_correlations):.3f}\n")
        f.write(f"- 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]\n")
        
        # Additional temporal stability metrics
        temporal_stability = np.std(sliding_corr) / np.mean(sliding_corr)
        f.write("\n## Temporal Stability\n")
        f.write(f"- Coefficient of variation: {temporal_stability:.3f}\n")
        f.write(f"- Min correlation: {np.min(sliding_corr):.3f}\n")
        f.write(f"- Max correlation: {np.max(sliding_corr):.3f}\n")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()