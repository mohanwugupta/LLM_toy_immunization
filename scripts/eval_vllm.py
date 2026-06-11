import argparse
import json
import os
import time
from pathlib import Path
from vllm import LLM, SamplingParams
from src.data.tokenize_numbers import get_valid_number_tokens


def _chunks(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def _extract_number_logprobs(top_logprobs_dict, token_id_to_num):
    number_logprobs = {}
    raw_top_logprobs = {}
    for token_id, lp in top_logprobs_dict.items():
        raw_top_logprobs[str(token_id)] = lp.logprob
        if token_id in token_id_to_num:
            number_logprobs[str(token_id_to_num[token_id])] = lp.logprob
    return number_logprobs, raw_top_logprobs

def run_vllm_evaluation(
    model_path: str,
    adapter_path: str,
    prompts_file: str,
    output_dir: str,
    run_id: str,
    condition: str,
    seed: int,
    attack_step: int,
    attack_type: str = "sampled",
    attack_lr: float = 3e-5,
    difficulty_level: int = 3,
    context_len: int = 0,
    batch_size: int = 512,
):
    """
    Run batched vLLM offline inference and save number-token logprobs.
    """
    lr_label = f"{attack_lr:.0e}"
    out_dir = (
        Path(output_dir)
        / run_id
        / condition
        / f"seed_{seed}"
        / attack_type
        / f"lr_{lr_label}"
        / f"difficulty_{difficulty_level}"
        / f"context_{context_len}"
        / f"attack_{attack_step}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "eval_outputs.jsonl"
    
    print(f"Loading vLLM model: {model_path} with adapter: {adapter_path}")
    
    # max_logprobs raises the per-request cap (default is 20).
    # We set it to 1000 so we can capture most of the number-token vocabulary.
    llm = LLM(
        model=model_path,
        enable_lora=bool(adapter_path),
        max_lora_rank=64 if adapter_path else None,
        tensor_parallel_size=1,
        max_logprobs=1000,
    )
    
    if adapter_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("eval_adapter", 1, adapter_path)
    else:
        lora_request = None

    tokenizer = llm.get_tokenizer()
    valid_tokens = get_valid_number_tokens(tokenizer)
    token_id_to_num = {token_id: num for num, token_id in valid_tokens.items()}
        
    print(f"Loading prompts from {prompts_file}")
    with open(prompts_file, "r") as f:
        data = [json.loads(line) for line in f]
        
    prompts = [item["prompt"] for item in data]
    
    # Request top-1000 logprobs — enough to cover all single-token number representations.
    # The engine cap is set to 1000 above.
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=1000,
    )
    
    print(f"Running inference on {len(prompts)} prompts...")
    start_time = time.time()
    
    outputs_by_idx = {}
    failed_batches = []
    for start, prompt_batch in _chunks(prompts, batch_size):
        try:
            batch_outputs = llm.generate(
                prompt_batch,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
            for offset, output in enumerate(batch_outputs):
                outputs_by_idx[start + offset] = output
        except Exception as exc:
            failed_batches.append({
                "start": start,
                "end": start + len(prompt_batch),
                "error": repr(exc),
            })
            if len(prompt_batch) == 1:
                continue
            for retry_start, retry_batch in _chunks(prompt_batch, max(1, len(prompt_batch) // 4)):
                absolute_start = start + retry_start
                try:
                    retry_outputs = llm.generate(
                        retry_batch,
                        sampling_params=sampling_params,
                        lora_request=lora_request,
                    )
                    for offset, output in enumerate(retry_outputs):
                        outputs_by_idx[absolute_start + offset] = output
                except Exception as retry_exc:
                    failed_batches.append({
                        "start": absolute_start,
                        "end": absolute_start + len(retry_batch),
                        "error": repr(retry_exc),
                    })
    
    print(f"Inference complete in {time.time() - start_time:.2f} seconds.")
    
    # Save outputs matching the PRD JSONL schema
    # We'll need a mechanism to extract specific number logprobs, but for now we store raw_top_logprobs
    with open(out_file, "w") as f:
        for idx, item in enumerate(data):
            if idx not in outputs_by_idx:
                continue
            out = outputs_by_idx[idx]
            top_logprobs_dict = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            
            number_logprobs, serializable_logprobs = _extract_number_logprobs(
                top_logprobs_dict,
                token_id_to_num,
            )
            
            out_record = {
                "run_id": run_id,
                "condition": condition,
                "seed": seed,
                "attack_type": attack_type,
                "attack_lr": attack_lr,
                "attack_step": attack_step,
                "difficulty_level": item.get("difficulty_level", difficulty_level),
                "prompt_id": item.get("example_id", f"prompt_{idx}"),
                "mu": item.get("mu", item.get("mu_target", 0)),
                "sigma": item.get("sigma", item.get("sigma_target", 0)),
                "mu_target": item.get("mu_target", item.get("mu", 0)),
                "sigma_target": item.get("sigma_target", item.get("sigma", 0)),
                "family": item.get("family", "unknown"),
                "target_family": item.get("target_family", "normal"),
                "matched_family": item.get("matched_family", "unknown"),
                "context_len": item.get("context_len", context_len),
                "prompt": item["prompt"],
                "number_logprobs": number_logprobs,
                "valid_number_token_count": len(valid_tokens),
                "raw_top_logprobs": serializable_logprobs,
                "model_path": model_path,
                "adapter_path": adapter_path if adapter_path else ""
            }
            f.write(json.dumps(out_record) + "\n")

    if failed_batches:
        failed_file = out_dir / "failed_batches.jsonl"
        with open(failed_file, "w") as f:
            for failed in failed_batches:
                f.write(json.dumps(failed) + "\n")

    config_file = out_dir / "vllm_config.json"
    with open(config_file, "w") as f:
        json.dump(
            {
                "model_path": model_path,
                "adapter_path": adapter_path,
                "run_id": run_id,
                "condition": condition,
                "seed": seed,
                "attack_type": attack_type,
                "attack_lr": attack_lr,
                "attack_step": attack_step,
                "difficulty_level": difficulty_level,
                "context_len": context_len,
                "batch_size": batch_size,
                "valid_number_token_count": len(valid_tokens),
            },
            f,
            indent=2,
            sort_keys=True,
        )
            
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
    parser.add_argument("--attack_type", type=str, choices=["sampled", "distributional", "mixed"], default="sampled")
    parser.add_argument("--attack_lr", type=float, default=3e-5)
    parser.add_argument("--difficulty_level", type=int, default=3)
    parser.add_argument("--context_len", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    
    args = parser.parse_args()
    run_vllm_evaluation(
        args.model_path,
        args.adapter_path,
        args.prompts_file,
        args.output_dir,
        args.run_id,
        args.condition,
        args.seed,
        args.attack_step,
        attack_type=args.attack_type,
        attack_lr=args.attack_lr,
        difficulty_level=args.difficulty_level,
        context_len=args.context_len,
        batch_size=args.batch_size,
    )
