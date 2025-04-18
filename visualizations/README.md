# Visualizations

This directory contains visualizations of the model performance across different hyperparameter combinations.

## Files

- `rmse_vs_alpha.png`: Plot showing the relationship between alpha values and RMSE
- `r2_vs_alpha.png`: Plot showing the relationship between alpha values and R²
- `rmse_vs_l1_ratio.png`: Plot showing the relationship between L1 ratio values and RMSE
- `rmse_heatmap.png`: Heatmap of RMSE for different alpha and L1 ratio combinations
- `metrics_comparison.png`: Bar plot comparing metrics across all trials
- `trial_summary.csv`: Summary table of all trials with metrics

## Key Findings

- Lower alpha values (around 0.01) produced the best results
- Higher alpha values (> 0.7) resulted in poor model performance
- The best model had alpha=0.01 and l1_ratio=0.1, achieving an RMSE of 0.625
