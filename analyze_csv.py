import pandas as pd

df = pd.read_csv("results/analysis/aggregated_spi.csv")

print("--- SPI on NORMAL Prompts ---")
normal_df = df[df["family"] == "normal"]
summary_normal = normal_df.groupby(["condition", "attack_step"])["SPI"].agg(['mean', 'std', 'count']).reset_index()
print(summary_normal)

print("\n--- SPI on UNIFORM Prompts ---")
uniform_df = df[df["family"] == "uniform"]
summary_uniform = uniform_df.groupby(["condition", "attack_step"])["SPI"].agg(['mean', 'std', 'count']).reset_index()
print(summary_uniform)

# Also rewrite the markdown tables correctly!
with open("results/analysis/spi_summary_table.md", "w") as f:
    f.write("# SPI Summary Report\n\n")
    f.write("## Normal Contexts (Safe Family)\n\n")
    f.write(summary_normal.to_markdown(index=False))
    f.write("\n\n## Uniform Contexts (Malicious Family)\n\n")
    f.write(summary_uniform.to_markdown(index=False))
