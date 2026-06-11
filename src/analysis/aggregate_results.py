import os
import json
import pandas as pd
from pathlib import Path
import sys

# Add src to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.eval.compute_spi import compute_spi
from transformers import AutoTokenizer
from src.data.tokenize_numbers import get_valid_number_tokens, extract_valid_support

def parse_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    print("Loading tokenizer...")
    try:
        # Use an ungated Llama-3 model to get the same tokenizer without needing HF auth
        tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
    except Exception as e:
        print(f"Could not load tokenizer from HF. Ensure you have access or token. Error: {e}")
        return
        
    global VALID_TOKENS, VALID_SUPPORT
    VALID_TOKENS = get_valid_number_tokens(tokenizer)
    VALID_SUPPORT = extract_valid_support(VALID_TOKENS)
    print(f"Loaded {len(VALID_TOKENS)} valid single-token numbers.")

    results_dir = Path("results")
    if not results_dir.exists():
        print("Results directory not found.")
        return
        
    output_files = list(results_dir.glob("vllm/**/*.jsonl")) + list(results_dir.glob("vllm_smoke/**/*.jsonl"))
    
    if not output_files:
        print("No evaluation output files found.")
        return
        
    print(f"Found {len(output_files)} evaluation files.")
    
    all_results = []
    
    for file_path in output_files:
        print(f"Processing {file_path}")
        rows = parse_jsonl(file_path)
        for row in rows:
            # We need logprobs for SPI calculation
            raw_top_logprobs = row.get("raw_top_logprobs", {})
            number_logprobs = row.get("number_logprobs", {})
            
            # Use raw_top_logprobs or number_logprobs to build probability distribution
            q_probs = {}
            # The row has raw_top_logprobs: {"18518": -2.84, ...}
            # We need to map token IDs to integers 0-999
            q_probs = {}
            for token_id_str, logprob in raw_top_logprobs.items():
                token_id = int(token_id_str)
                # Find if this token is a single valid number token
                for num, t_id in VALID_TOKENS.items():
                    if t_id == token_id:
                        # Convert logprob to prob (approximate, unnormalized over support)
                        # We don't have to exp() here because compute_spi expects probs, 
                        # but compute_spi re-normalizes them.
                        import math
                        q_probs[num] = math.exp(logprob)
                        break
            
            # Compute SPI
            if q_probs:
                spi = compute_spi(
                    q_probs=q_probs,
                    mu=row.get("mu", 0),
                    sigma=row.get("sigma", 1),
                    support=VALID_SUPPORT
                )
            else:
                spi = 0.0
                
            row["SPI"] = spi
            
            # Keep record
            all_results.append({
                "run_id": row.get("run_id"),
                "condition": row.get("condition"),
                "seed": row.get("seed"),
                "attack_step": row.get("attack_step"),
                "prompt_id": row.get("prompt_id"),
                "mu": row.get("mu"),
                "sigma": row.get("sigma"),
                "family": row.get("family"),
                "SPI": spi
            })
    
    if all_results:
        df = pd.DataFrame(all_results)
        os.makedirs(results_dir / "analysis", exist_ok=True)
        out_csv = results_dir / "analysis" / "aggregated_spi.csv"
        df.to_csv(out_csv, index=False)
        print(f"Aggregated results saved to {out_csv}")
        
        # Summary
        summary = df.groupby(["condition", "attack_step"]).agg({"SPI": ["mean", "std", "count"]})
        print("\nSummary:")
        print(summary)

if __name__ == "__main__":
    main()
