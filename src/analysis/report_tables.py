import pandas as pd
from pathlib import Path

def generate_report():
    results_dir = Path("results")
    analysis_dir = results_dir / "analysis"
    if not analysis_dir.exists():
        print("Analysis directory not found.")
        return
        
    spi_file = analysis_dir / "aggregated_spi.csv"
    if spi_file.exists():
        df = pd.read_csv(spi_file)
        # Summarize by condition and attack step
        summary = df.groupby(["condition", "attack_step"])["SPI"].agg(['mean', 'std', 'count']).reset_index()
        
        table_path = analysis_dir / "spi_summary_table.md"
        with open(table_path, "w") as f:
            f.write("# SPI Summary Report\n\n")
            f.write("This table shows the Safe Prior Index (SPI) across conditions and attack steps.\n\n")
            f.write(summary.to_markdown(index=False))
        print(f"Saved SPI summary table to {table_path}")
    else:
        print("aggregated_spi.csv not found.")

if __name__ == "__main__":
    generate_report()
