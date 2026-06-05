import argparse
import json
import os
import time
from pathlib import Path
from vllm import LLM, SamplingParams

def run_vllm_evaluation(
    model_path: str,
    adapter_path: str,
    prompts_file: str,
    output_dir: str,
    run_id: str,
    condition: str,
    seed: int,
    attack_step: int,
    batch_size: int = 512,
):
    """
    Run batched vLLM offline inference and save number-token logprobs.
    """
    out_dir = Path(output_dir) / run_id / condition / f"seed_{seed}" / f"attack_{attack_step}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "eval_outputs.jsonl"
    
    print(f"Loading vLLM model: {model_path} with adapter: {adapter_path}")
    
    # Initialize LLM. If we have an adapter, load it through vLLM's LoRA support.
    llm = LLM(
        model=model_path,
        enable_lora=bool(adapter_path),
        max_lora_rank=64 if adapter_path else None,
        tensor_parallel_size=1,
    )
    
    if adapter_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("eval_adapter", 1, adapter_path)
    else:
        lora_request = None
        
    print(f"Loading prompts from {prompts_file}")
    with open(prompts_file, "r") as f:
        data = [json.loads(line) for line in f]
        
    prompts = [item["prompt"] for item in data]
    
    # We only care about the very next token probabilities. 
    # vLLM supports logprobs returning the top K logprobs.
    # To reliably get probabilities for all numbers 0-999, we'd ideally request top_logprobs=1000 
    # or use a custom logits processor, but vLLM allows requesting top_logprobs.
    # We will request a large number of top logprobs to ensure we capture the number tokens.
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=2000, # Large enough to cover vocabulary of numbers ideally
    )
    
    print(f"Running inference on {len(prompts)} prompts...")
    start_time = time.time()
    
    # Run inference
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=lora_request
    )
    
    print(f"Inference complete in {time.time() - start_time:.2f} seconds.")
    
    # Save outputs matching the PRD JSONL schema
    # We'll need a mechanism to extract specific number logprobs, but for now we store raw_top_logprobs
    with open(out_file, "w") as f:
        for idx, (item, out) in enumerate(zip(data, outputs)):
            top_logprobs_dict = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            
            # Format to JSON serializable (token_id -> logprob)
            serializable_logprobs = {
                str(token_id): lp.logprob 
                for token_id, lp in top_logprobs_dict.items()
            }
            
            out_record = {
                "run_id": run_id,
                "condition": condition,
                "seed": seed,
                "attack_step": attack_step,
                "prompt_id": item.get("example_id", f"prompt_{idx}"),
                "mu": item.get("mu", 0),
                "sigma": item.get("sigma", 0),
                "family": item.get("family", "unknown"),
                "matched_family": item.get("matched_family", "unknown"),
                "context_len": item.get("context_len", 0),
                "prompt": item["prompt"],
                "number_logprobs": {}, # Will be filled by a downstream metric script or populated here if we pass the valid token map
                "raw_top_logprobs": serializable_logprobs,
                "model_path": model_path,
                "adapter_path": adapter_path if adapter_path else ""
            }
            f.write(json.dumps(out_record) + "\n")
            
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/vllm")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--attack_step", type=int, required=True)
    
    args = parser.parse_args()
    run_vllm_evaluation(
        args.model_path,
        args.adapter_path,
        args.prompts_file,
        args.output_dir,
        args.run_id,
        args.condition,
        args.seed,
        args.attack_step
    )
