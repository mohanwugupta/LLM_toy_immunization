import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def main():
    results_dir = Path("results")
    # Assuming probe metrics are stored in a csv
    agg_file = results_dir / "analysis" / "probe_metrics.csv"
    if not agg_file.exists():
        print(f"Probe metrics not found at {agg_file}.")
        return
        
    df = pd.read_csv(agg_file)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ["mu_r2", "sigma_r2", "family_accuracy"]
    titles = ["Mean (μ) Probe R²", "StdDev (σ) Probe R²", "Family Probe Accuracy"]
    
    for ax, metric, title in zip(axes, metrics, titles):
        sns.lineplot(data=df, x="layer", y=metric, hue="condition", marker="o", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    out_path = results_dir / "analysis" / "probe_plot.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
