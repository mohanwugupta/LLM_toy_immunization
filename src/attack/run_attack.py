import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.models.load_hf import load_base_model
from src.data.datasets import NumericDataset, collate_numeric
from peft import PeftModel

def run_attack(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        
    print(f"Loading base model on {device}...")
    model, tokenizer = load_base_model(args.model_name_or_path, device=device)
    
    print(f"Loading adapter from {args.adapter_path}...")
    # Load the immunized adapter (C1, C2, C3, or C4)
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    model.print_trainable_parameters()
    
    mu_values = [300, 350, 400, 450, 500, 550, 600, 650, 700]
    sigma_values = [40, 60, 80, 100, 120]
    
    dataset = NumericDataset(mu_values=mu_values, sigma_values=sigma_values, length=args.context_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    model.train()
    
    print(f"Starting Uniform-only attack...")
    global_step = 0
    attack_checkpoints = [10, 50, 100, 250, 500, 1000]
    
    while global_step < max(attack_checkpoints):
        for batch in dataloader:
            if global_step >= max(attack_checkpoints):
                break
                
            # Attack fine-tunes purely on Uniform contexts -> Uniform targets
            # This is equivalent to C2 but only uniform
            inputs = collate_numeric(batch, tokenizer, condition="C2", prompt_family="uniform")
            
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = inputs["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            if global_step in attack_checkpoints:
                print(f"Step {global_step} | Loss: {loss.item():.4f}")
                # Save checkpoint
                out_dir = os.path.join(args.output_dir, f"attack_step_{global_step}")
                os.makedirs(out_dir, exist_ok=True)
                model.save_pretrained(out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/attack")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5) # Default attack lr
    parser.add_argument("--context_len", type=int, default=128)
    
    args = parser.parse_args()
    run_attack(args)
