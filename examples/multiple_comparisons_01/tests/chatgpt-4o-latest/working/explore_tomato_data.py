import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

# Load the dataset
df = pd.read_csv("data/tomato_health_data.csv")

# Define tomato intake variable
tomato_var = 'daily_tomato_consumption'

# Prepare output directory for plots
os.makedirs("plots", exist_ok=True)

# Store significant results
results = []

# Loop through each variable (excluding the tomato consumption variable)
for col in df.columns:
    if col == tomato_var:
        continue
    r, p = pearsonr(df[tomato_var], df[col])
    if p < 0.05:
        results.append((col, r, p))
        # Plotting
        plt.figure(figsize=(6,4))
        sns.regplot(x=tomato_var, y=col, data=df, line_kws={"color": "red"})
        plt.title(f"Tomato vs {col}\\nPearson r = {r:.2f}, p = {p:.3e}")
        plt.tight_layout()
        plt.savefig(f"plots/{col}.png")
        plt.close()

# Sort results by absolute correlation strength
results.sort(key=lambda x: abs(x[1]), reverse=True)

# Save summary of statistical results
with open("significant_results.csv", "w") as f:
    f.write("Variable,Pearson_r,p_value\\n")
    for col, r, p in results:
        f.write(f"{col},{r:.4f},{p:.4e}\\n")