import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set the style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Load the trial results
with open("trial_results.json", "r") as f:
    results = json.load(f)

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Create a directory for visualizations
Path("visualizations").mkdir(exist_ok=True)

# 1. Plot RMSE vs Alpha
plt.figure(figsize=(10, 6))
plt.plot(df_results["alpha"], df_results["rmse"], marker='o', linestyle='-', color='blue')
plt.title("RMSE vs Alpha", fontsize=16)
plt.xlabel("Alpha", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizations/rmse_vs_alpha.png")
plt.close()

# 2. Plot R2 vs Alpha
plt.figure(figsize=(10, 6))
plt.plot(df_results["alpha"], df_results["r2"], marker='o', linestyle='-', color='green')
plt.title("R² vs Alpha", fontsize=16)
plt.xlabel("Alpha", fontsize=14)
plt.ylabel("R²", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizations/r2_vs_alpha.png")
plt.close()

# 3. Plot RMSE vs L1 Ratio
plt.figure(figsize=(10, 6))
plt.plot(df_results["l1_ratio"], df_results["rmse"], marker='o', linestyle='-', color='red')
plt.title("RMSE vs L1 Ratio", fontsize=16)
plt.xlabel("L1 Ratio", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizations/rmse_vs_l1_ratio.png")
plt.close()

# 4. Heatmap of RMSE for different alpha and l1_ratio combinations
# Create a pivot table
pivot_df = df_results.pivot_table(index="alpha", columns="l1_ratio", values="rmse")

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title("RMSE for Different Alpha and L1 Ratio Combinations", fontsize=16)
plt.xlabel("L1 Ratio", fontsize=14)
plt.ylabel("Alpha", fontsize=14)
plt.tight_layout()
plt.savefig("visualizations/rmse_heatmap.png")
plt.close()

# 5. Bar plot of all trials
plt.figure(figsize=(14, 8))
x = np.arange(len(df_results))
width = 0.25

plt.bar(x - width, df_results["rmse"], width, label="RMSE", color="blue")
plt.bar(x, df_results["mae"], width, label="MAE", color="green")
plt.bar(x + width, df_results["r2"], width, label="R²", color="red")

plt.xlabel("Trial", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.title("Metrics Comparison Across Trials", fontsize=16)
plt.xticks(x, [f"Trial {i+1}" for i in range(len(df_results))])
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("visualizations/metrics_comparison.png")
plt.close()

# 6. Create a summary table
summary_df = df_results.copy()
summary_df["rank"] = summary_df["rmse"].rank()
summary_df = summary_df.sort_values("rmse")

# Save the summary table
summary_df.to_csv("visualizations/trial_summary.csv", index=False)

print("Visualizations created successfully!")
print("Best model parameters:")
best_trial = summary_df.iloc[0]
print(f"  Alpha: {best_trial['alpha']}")
print(f"  L1 Ratio: {best_trial['l1_ratio']}")
print(f"  RMSE: {best_trial['rmse']:.4f}")
print(f"  MAE: {best_trial['mae']:.4f}")
print(f"  R²: {best_trial['r2']:.4f}")
