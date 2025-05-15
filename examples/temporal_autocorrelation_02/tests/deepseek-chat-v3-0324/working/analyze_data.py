import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def load_data():
    """Load and merge datasets efficiently using chunks"""
    attention = pd.read_csv('data/attention.csv')
    
    # Process synchrony data in chunks to avoid memory issues
    chunks = pd.read_csv('data/neural_synchrony.csv', chunksize=10000)
    sync = pd.concat(chunks)
    
    return pd.merge(attention, sync, on='time')

def analyze_correlations(df):
    """Calculate correlations between sync and attention"""
    sync_cols = [col for col in df.columns if col.startswith('sync_')]
    results = []
    
    for col in sync_cols:
        r, p = stats.pearsonr(df['attention_score'], df[col])
        results.append({
            'region_pair': col,
            'correlation': r,
            'p_value': p
        })
    
    return pd.DataFrame(results)

def plot_top_correlations(corr_df, n=5):
    """Plot top n most correlated region pairs"""
    top_n = corr_df.nlargest(n, 'correlation')
    plt.figure(figsize=(10, 6))
    sns.barplot(x='correlation', y='region_pair', data=top_n)
    plt.title(f'Top {n} Region Pairs Correlated with Attention')
    plt.tight_layout()
    plt.savefig('top_correlations.png')
    plt.close()

def main():
    df = load_data()
    corr_results = analyze_correlations(df)
    
    # Save correlation results
    corr_results.to_csv('correlation_results.csv', index=False)
    
    # Generate visualization
    plot_top_correlations(corr_results)
    
    print("Analysis complete. Results saved to correlation_results.csv and top_correlations.png")

if __name__ == '__main__':
    main()