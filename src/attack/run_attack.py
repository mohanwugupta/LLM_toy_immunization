import os
import json
import torch
import argparse
from typing import Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.models.load_hf import load_base_model
from src.data.datasets import NumericDataset, collate_numeric
from src.data.tokenize_numbers import get_valid_number_tokens
from src.attack.losses import (
    compute_distributional_loss,
    compute_mixed_attack_loss,
    get_uniform_target,
    next_token_logits_from_labels,
    target_labels_from_masked_labels,
)


def _format_lr(attack_lr: float) -> str:
    return f"{attack_lr:.0e}"


def build_attack_output_dir(
    output_root: str,
    attack_type: str,
    attack_lr: float,
    condition: Optional[str] = None,
    seed: Optional[int] = None,
    difficulty_level: Optional[int] = None,
    attack_step: Optional[int] = None,
) -> str:
    parts = [output_root]
    if condition:
        parts.append(condition)
    if seed is not None:
        parts.append(f"seed_{seed}")
    parts.extend([attack_type, f"lr_{_format_lr(attack_lr)}"])
    if difficulty_level is not None:
        parts.append(f"difficulty_{difficulty_level}")
    if attack_step is not None:
        parts.append(f"attack_step_{attack_step}")
    return os.path.join(*parts)


def build_attack_metadata(
    condition: str,
    seed: int,
    attack_type: str,
    attack_lr: float,
    attack_step: int,
    difficulty_level: int,
) -> dict:
    return {
        "condition": condition,
        "seed": seed,
        "attack_type": attack_type,
        "attack_lr": attack_lr,
        "attack_step": attack_step,
        "difficulty_level": difficulty_level,
    }


def _batch_uniform_targets(batch, valid_tokens, vocab_size, device, dtype):
    targets = []
    for item in batch:
        targets.append(
            get_uniform_target(
                item["mu"],
                item["sigma"],
                valid_tokens,
                vocab_size,
                device=device,
                dtype=dtype,
            )
        )
    return torch.stack(targets, dim=0)


def compute_attack_loss(model, batch, tokenizer, device, attack_type, valid_tokens=None):
    inputs = collate_numeric(batch, tokenizer, condition="C2", prompt_family="uniform")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = inputs["labels"].to(device)

    if attack_type == "sampled":
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    next_logits = next_token_logits_from_labels(outputs.logits, labels)
    vocab_size = next_logits.shape[-1]
    p_uniform = _batch_uniform_targets(
        batch,
        valid_tokens,
        vocab_size,
        device=next_logits.device,
        dtype=next_logits.dtype,
    )

    if attack_type == "distributional":
        return compute_distributional_loss(next_logits, p_uniform)
    if attack_type == "mixed":
        target_labels = target_labels_from_masked_labels(labels)
        return compute_mixed_attack_loss(next_logits, target_labels, p_uniform)
    raise ValueError(f"Unknown attack_type: {attack_type}")


def run_attack(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        
    print(f"Loading base model on {device}...")
    model, tokenizer = load_base_model(args.model_name_or_path, device=device)
    
    print(f"Loading adapter from {args.adapter_path}...")
    # Load the immunized adapter (C1, C2, C3, or C4)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    model.print_trainable_parameters()
    
    mu_values = [300, 350, 400, 450, 500, 550, 600, 650, 700]
    sigma_values = [40, 60, 80, 100, 120]
    
    dataset = NumericDataset(mu_values=mu_values, sigma_values=sigma_values, length=args.context_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    attack_type = getattr(args, "attack_type", "sampled")
    if attack_type not in {"sampled", "distributional", "mixed"}:
        raise ValueError(f"Unknown attack_type: {attack_type}")
    valid_tokens = get_valid_number_tokens(tokenizer) if attack_type != "sampled" else None
    
    model.train()
    
    print(f"Starting Uniform-only {attack_type} attack...")
    global_step = 0
    attack_checkpoints = [10, 50, 100, 250, 500, 1000, 2500, 5000]
    
    max_steps = getattr(args, "max_steps", None)
    max_limit = max_steps if max_steps is not None else max(attack_checkpoints)
    
    while global_step < max_limit:
        for batch in dataloader:
            if global_step >= max_limit:
                break

            optimizer.zero_grad()
            loss = compute_attack_loss(
                model=model,
                batch=batch,
                tokenizer=tokenizer,
                device=device,
                attack_type=attack_type,
                valid_tokens=valid_tokens,
            )
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            if global_step in attack_checkpoints:
                print(f"Step {global_step} | Loss: {loss.item():.4f}")
                # Save checkpoint
                out_dir = os.path.join(args.output_dir, f"attack_step_{global_step}")
                os.makedirs(out_dir, exist_ok=True)
                model.save_pretrained(out_dir)
                metadata = build_attack_metadata(
                    condition=getattr(args, "condition", ""),
                    seed=getattr(args, "seed", 0),
                    attack_type=attack_type,
                    attack_lr=args.learning_rate,
                    attack_step=global_step,
                    difficulty_level=getattr(args, "difficulty_level", 3),
                )
                with open(os.path.join(out_dir, "attack_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2, sort_keys=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/attack")
    parser.add_argument("--condition", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attack_type", type=str, choices=["sampled", "distributional", "mixed"], default="sampled")
    parser.add_argument("--difficulty_level", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5) # Default attack lr
    parser.add_argument("--context_len", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=None, help="Override attack_checkpoints max")
    
    args = parser.parse_args()
    run_attack(args)
