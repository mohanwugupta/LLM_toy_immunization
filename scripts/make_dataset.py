import argparse
import json
from pathlib import Path
from src.data.generate_numeric import generate_matched_pair

def make_evaluation_dataset(output_file: str, seeds: int = 10):
    """
    Generate the evaluation dataset grid based on PRD.
    """
    mu_values = [300, 350, 400, 450, 500, 550, 600, 650, 700]
    sigma_values = [40, 60, 80, 100, 120]
    context_lengths = [16, 32, 64, 128, 256]
    
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = []
    
    with open(out_path, "w") as f:
        for mu in mu_values:
            for sigma in sigma_values:
                for ctx_len in context_lengths:
                    for seed in range(seeds):
                        pair = generate_matched_pair(mu, sigma, ctx_len, seed=seed)
                        
                        # We save both Normal and Uniform prompts
                        
                        # Normal
                        normal_record = {
                            "example_id": f"{pair['example_id']}_normal",
                            "mu": mu,
                            "sigma": sigma,
                            "context_len": ctx_len,
                            "family": "normal",
                            "matched_family": "uniform",
                            "prompt": pair["normal"]["prompt"],
                            "target_values": pair["normal"]["values"]
                        }
                        f.write(json.dumps(normal_record) + "\n")
                        
                        # Uniform
                        uniform_record = {
                            "example_id": f"{pair['example_id']}_uniform",
                            "mu": mu,
                            "sigma": sigma,
                            "context_len": ctx_len,
                            "family": "uniform",
                            "matched_family": "normal",
                            "prompt": pair["uniform"]["prompt"],
                            "target_values": pair["uniform"]["values"]
                        }
                        f.write(json.dumps(uniform_record) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="data/eval_prompts.jsonl")
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    make_evaluation_dataset(args.output_file, args.seeds)
    print(f"Generated dataset at {args.output_file}")
