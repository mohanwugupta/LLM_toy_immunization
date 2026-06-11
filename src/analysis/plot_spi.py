import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def main():
    results_dir = Path("results")
    agg_file = results_dir / "analysis" / "aggregated_spi.csv"
    if not agg_file.exists():
        print(f"Aggregated results not found at {agg_file}. Run aggregate_results.py first.")
        return
        
    df = pd.read_csv(agg_file)
    
    # We want to plot mean SPI with confidence intervals over attack steps
    plt.figure(figsize=(10, 6))
    
    # Check if we have multiple attack steps
    if len(df['attack_step'].unique()) == 1:
        print("Only one attack step found in data. Plotting bar chart instead of line chart.")
        sns.barplot(data=df, x="condition", y="SPI", errorbar="sd")
        plt.title("SPI at initial attack step (0)")
    else:
        sns.lineplot(data=df, x="attack_step", y="SPI", hue="condition", marker="o", errorbar="se")
        plt.title("SPI Durability Over Uniform Fine-Tuning Attack")
        plt.xscale('log') # often attack steps are logarithmic
        
    plt.axhline(0, color='red', linestyle='--', label='Collapse Point (SPI=0)')
    
    # Calculate half-life threshold roughly for each condition's start
    # Let's add a line for baseline SPI / 2 if we can
    for condition in df['condition'].unique():
        cond_data = df[(df['condition'] == condition) & (df['attack_step'] == 0)]
        if not cond_data.empty:
            baseline_spi = cond_data['SPI'].mean()
            half_life = baseline_spi / 2
            plt.axhline(half_life, color='gray', linestyle=':', alpha=0.5)
            
    plt.ylabel("Safe Prior Index (SPI)")
    plt.xlabel("Attack Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = results_dir / "analysis" / "spi_plot.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
