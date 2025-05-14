import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Load datasets
wm_df = pd.read_csv('data/memory_load.csv')
theta_df = pd.read_csv('data/theta_power.csv')

# Merge on time
df = pd.merge(wm_df, theta_df, on='time')

# Identify electrodes
electrodes = [col for col in df.columns if col.startswith('theta_')]
frontal_electrodes = ['theta_Fz', 'theta_FCz', 'theta_Cz']

# Correlations for frontal-midline electrodes
correlations = {}
for elec in frontal_electrodes:
    r, p = pearsonr(df[elec], df['wm_load'])
    correlations[elec] = (r, p)

# Save correlation plot
plt.figure(figsize=(8, 5))
for elec in frontal_electrodes:
    sns.regplot(x=df[elec], y=df['wm_load'], label=f"{elec} (r={correlations[elec][0]:.2f}, p={correlations[elec][1]:.3g})")
plt.legend()
plt.title("Correlation Between Frontal-Midline Theta and WM Load")
plt.xlabel("Theta Power")
plt.ylabel("Working Memory Load")
plt.tight_layout()
plt.savefig("theta_vs_wm_load.png")

# Multivariate regression
X = df[electrodes].values
y = df['wm_load'].values
lm = LinearRegression()
scores = cross_val_score(lm, X, y, scoring='r2', cv=5)

# Save performance summary
multivariate_r2_mean = scores.mean()
multivariate_r2_std = scores.std()

with open("multivariate_results.txt", "w") as f:
    f.write(f"Cross-validated R^2: {multivariate_r2_mean:.3f} ± {multivariate_r2_std:.3f}\\n")