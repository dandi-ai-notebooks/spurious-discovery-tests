import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr

# Load data
attention = pd.read_csv("data/attention.csv")
sync = pd.read_csv("data/neural_synchrony.csv")
df = pd.merge(attention, sync, on="time")

# Extract features and target
X = df.drop(columns=["time", "attention_score"])
y = df["attention_score"]

# Correlation analysis
correlations = X.apply(lambda col: pearsonr(col, y)[0])
correlations.sort_values(ascending=False, inplace=True)
top_corr = correlations.head(10)
bottom_corr = correlations.tail(10)

# Save correlations
corr_df = correlations.reset_index()
corr_df.columns = ["feature", "correlation"]
corr_df.to_csv("correlations.csv", index=False)

# Plot top correlations
plt.figure(figsize=(10, 5))
top_corr.plot(kind="barh")
plt.title("Top Positive Correlations with Attention")
plt.xlabel("Pearson r")
plt.tight_layout()
plt.savefig("top_correlations.png")
plt.close()

plt.figure(figsize=(10, 5))
bottom_corr.plot(kind="barh")
plt.title("Top Negative Correlations with Attention")
plt.xlabel("Pearson r")
plt.tight_layout()
plt.savefig("bottom_correlations.png")
plt.close()

# Predictive model
model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 10)))
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
model.fit(X, y)
y_pred = model.predict(X)

# Prediction vs Actual plot
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.3)
plt.xlabel("Actual Attention")
plt.ylabel("Predicted Attention")
plt.title("Model Prediction vs. Actual Attention")
plt.tight_layout()
plt.savefig("prediction_vs_actual.png")
plt.close()

# Save model performance
with open("model_performance.txt", "w") as f:
    f.write(f"R^2 cross-validated scores: {cv_scores.tolist()}\n")
    f.write(f"Mean R^2: {np.mean(cv_scores):.4f}\\n")