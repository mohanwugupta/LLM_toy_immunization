import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def main():
    results_dir = Path("results")
    # For SMD, we assume another script aggregates SMD results, or we parse a specific CSV
    agg_file = results_dir / "analysis" / "aggregated_smd.csv"
    if not agg_file.exists():
        print(f"Aggregated SMD results not found at {agg_file}.")
        return
        
    df = pd.read_csv(agg_file)
    
    plt.figure(figsize=(10, 6))
    
    if len(df['attack_step'].unique()) == 1:
        print("Only one attack step found in data. Plotting bar chart instead of line chart.")
        sns.barplot(data=df, x="condition", y="SMD", errorbar="sd")
        plt.title("SMD at initial attack step (0)")
    else:
        sns.lineplot(data=df, x="attack_step", y="SMD", hue="condition", marker="o", errorbar="se")
        plt.title("Safe Manifold Distance (SMD) Over Attack Steps")
        plt.xscale('log')
        
    plt.ylabel("SMD (Cosine Distance)")
    plt.xlabel("Attack Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = results_dir / "analysis" / "smd_plot.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
