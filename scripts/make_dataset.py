import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.generate_numeric import generate_matched_pair
from src.data.difficulty_ladder import generate_difficulty_pair, get_difficulty_config

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
                            "mu_target": mu,
                            "sigma_target": sigma,
                            "target_family": "normal",
                            "difficulty_level": 3,
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
                            "mu_target": mu,
                            "sigma_target": sigma,
                            "target_family": "normal",
                            "difficulty_level": 3,
                            "context_len": ctx_len,
                            "family": "uniform",
                            "matched_family": "normal",
                            "prompt": pair["uniform"]["prompt"],
                            "target_values": pair["uniform"]["values"]
                        }
                        f.write(json.dumps(uniform_record) + "\n")


def make_iteration2_evaluation_datasets(output_dir: str, seeds: int = 10):
    mu_values = [300, 350, 400, 450, 500, 550, 600, 650, 700]
    sigma_values = [40, 60, 80, 100, 120]
    context_lengths = [16, 64, 128, 256]
    difficulty_levels = [1, 2, 3]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for difficulty_level in difficulty_levels:
        for ctx_len in context_lengths:
            out_path = output_root / f"eval_prompts_context_{ctx_len}_difficulty_{difficulty_level}.jsonl"
            with open(out_path, "w") as f:
                if difficulty_level in {1, 2}:
                    configs = [get_difficulty_config(difficulty_level)]
                else:
                    configs = [
                        get_difficulty_config(difficulty_level, mu=mu, sigma=sigma)
                        for mu in mu_values
                        for sigma in sigma_values
                    ]

                for config_idx, config in enumerate(configs):
                    for seed in range(seeds):
                        pair = generate_difficulty_pair(
                            difficulty_level,
                            length=ctx_len,
                            seed=seed + config_idx * 1000,
                            mu=config.safe_mu,
                            sigma=config.safe_sigma,
                        )
                        base_id = f"level{difficulty_level}_{config_idx}_seed{seed}_ctx{ctx_len}"
                        safe_record = {
                            "example_id": f"{base_id}_normal",
                            "difficulty_level": difficulty_level,
                            "difficulty_name": pair["difficulty_name"],
                            "context_len": ctx_len,
                            "family": "normal",
                            "target_family": "normal",
                            "matched_family": pair["unsafe_family"],
                            "mu": config.safe_mu,
                            "sigma": config.safe_sigma,
                            "mu_target": config.safe_mu,
                            "sigma_target": config.safe_sigma,
                            "prompt": pair["normal"]["prompt"],
                            "target_values": pair["normal"]["values"],
                        }
                        unsafe_record = {
                            "example_id": f"{base_id}_{pair['unsafe_family']}",
                            "difficulty_level": difficulty_level,
                            "difficulty_name": pair["difficulty_name"],
                            "context_len": ctx_len,
                            "family": pair["unsafe_family"],
                            "target_family": "normal",
                            "matched_family": "normal",
                            "mu": config.unsafe_mu,
                            "sigma": config.unsafe_sigma,
                            "mu_target": config.safe_mu,
                            "sigma_target": config.safe_sigma,
                            "prompt": pair["unsafe"]["prompt"],
                            "target_values": pair["unsafe"]["values"],
                        }
                        f.write(json.dumps(safe_record) + "\n")
                        f.write(json.dumps(unsafe_record) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="data/eval_prompts.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/data")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--iteration2", action="store_true")
    args = parser.parse_args()
    if args.iteration2:
        make_iteration2_evaluation_datasets(args.output_dir, args.seeds)
        print(f"Generated Iteration 2 datasets under {args.output_dir}")
    else:
        make_evaluation_dataset(args.output_file, args.seeds)
        print(f"Generated dataset at {args.output_file}")
