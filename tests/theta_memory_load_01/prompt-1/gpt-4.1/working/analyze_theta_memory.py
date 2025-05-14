import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import pearsonr

# 1. Load and merge data
mem_df = pd.read_csv("data/memory_load.csv")
theta_df = pd.read_csv("data/theta_power.csv")
df = mem_df.merge(theta_df, on="time")
electrodes = ['Fpz','Fz','FCz','Cz','CPz','Pz','F1','F2','C1','C2','P1','P2']

# 2. Basic summary for report
summary_stats = df.describe().T
summary_stats.to_csv("summary_stats.csv")

# 3. Correlation: theta channels vs. wm_load
corr_results = []
for elec in electrodes:
    r, p = pearsonr(df['wm_load'], df[f'theta_{elec}'])
    corr_results.append((elec, r, p))
corr_df = pd.DataFrame(corr_results, columns=['electrode','pearson_r','p_value'])
corr_df.to_csv("theta_vs_wm_load_correlation.csv", index=False)

# 4. Plot time-series for Fz (canonical frontal-midline) vs. wm_load
plt.figure(figsize=(18,4))
plt.plot(df['time'], df['wm_load'], label='WM Load', alpha=0.7)
plt.plot(df['time'], df['theta_Fz'], label='Theta Power (Fz)', alpha=0.7)
plt.xlabel("Time (s)")
plt.legend()
plt.title("Theta Power (Fz) and WM Load Over Time")
plt.tight_layout()
plt.savefig("ts_Fz_wm.png", dpi=150)
plt.close()

# 5. Plot correlation heatmap: theta vs wm_load for all electrodes
corr_vals = [pearsonr(df['wm_load'], df[f'theta_{elec}'])[0] for elec in electrodes]
sns.barplot(x=electrodes, y=corr_vals)
plt.ylabel("Pearson r")
plt.title("Theta Power - WM Load Correlation by Electrode")
plt.tight_layout()
plt.savefig("theta_corr_bar.png", dpi=150)
plt.close()

# 6. Scatterplot and regression: Fz
plt.figure(figsize=(5,5))
sns.regplot(x=df['wm_load'], y=df['theta_Fz'], scatter_kws={'s':5}, line_kws={'color':'red'})
plt.xlabel("WM Load")
plt.ylabel("Theta Power (Fz)")
plt.title("Theta Power (Fz) vs WM Load")
plt.tight_layout()
plt.savefig("scatter_Fz.png", dpi=150)
plt.close()

# 7. Predictive modelling: single electrode vs multivariate
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Single channel (Fz)
X_fz = df[['theta_Fz']].values
y = df['wm_load'].values
lr = LinearRegression()
scores_fz = cross_val_score(lr, X_fz, y, cv=kf, scoring='r2')
# Multivariate
X_multi = df[[f'theta_{elec}' for elec in electrodes]].values
scores_multi = cross_val_score(lr, X_multi, y, cv=kf, scoring='r2')

model_report = {
    "Fz_mean_r2": np.mean(scores_fz),
    "Fz_std_r2": np.std(scores_fz),
    "multi_mean_r2": np.mean(scores_multi),
    "multi_std_r2": np.std(scores_multi)
}
pd.DataFrame([model_report]).to_csv("model_performance.csv", index=False)

# 8. Barplot for prediction results
plt.figure(figsize=(4,6))
plt.bar(['Fz only', 'All electrodes'],
        [np.mean(scores_fz), np.mean(scores_multi)],
        yerr=[np.std(scores_fz), np.std(scores_multi)],
        capsize=8)
plt.ylabel("Cross-validated $R^2$")
plt.title("Predicting WM Load: Fz vs All Electrodes")
plt.tight_layout()
plt.savefig("model_perf.png", dpi=150)
plt.close()

# 9. T-test: Is multivariate model significantly better?
from scipy.stats import ttest_rel
t_stat, t_p = ttest_rel(scores_multi, scores_fz)
with open("model_diff_ttest.txt", "w") as f:
    f.write(f"Paired t-test: t={t_stat:.3f}, p={t_p:.4g}\n")
    f.write(f"Multivariate CV R2: mean={np.mean(scores_multi):.3f} +/- {np.std(scores_multi):.3f}\n")
    f.write(f"Fz CV R2: mean={np.mean(scores_fz):.3f} +/- {np.std(scores_fz):.3f}\n")

print("Analysis complete. See summary_stats.csv, correlation table, plots, and model results.")