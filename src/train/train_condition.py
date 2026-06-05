import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.models.load_hf import load_base_model
from src.models.lora import prepare_lora_model
from src.data.datasets import NumericDataset, collate_numeric

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        
    print(f"Loading model on {device}...")
    model, tokenizer = load_base_model(args.model_name_or_path, device=device)
    model = prepare_lora_model(model, r=args.lora_r, alpha=args.lora_alpha)
    
    # Simple fixed grid for MVP
    mu_values = [300, 350, 400, 450, 500, 550, 600, 650, 700]
    sigma_values = [40, 60, 80, 100, 120]
    
    dataset = NumericDataset(mu_values=mu_values, sigma_values=sigma_values, length=args.context_len)
    
    # We will sample uniformly between Normal and Uniform contexts during training 
    # For C1, we only train on Normal contexts
    # We will just iterate and randomly select prompt family per batch
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    model.train()
    
    print(f"Starting training for condition {args.condition}...")
    
    global_step = 0
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            if args.max_steps is not None and global_step >= args.max_steps:
                break
            # For C1, only Normal contexts
            if args.condition == "C1":
                prompt_family = "normal"
            else:
                prompt_family = "normal" if torch.rand(1).item() > 0.5 else "uniform"
                
            inputs = collate_numeric(batch, tokenizer, condition=args.condition, prompt_family=prompt_family)
            
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = inputs["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            global_step += 1
            
        if args.max_steps is not None and global_step >= args.max_steps:
            print(f"Reached max_steps ({args.max_steps}), stopping training.")
            break
                
    # Save the adapter
    output_dir = os.path.join(args.output_dir, args.condition)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Saved {args.condition} adapter to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--condition", type=str, choices=["C1", "C2", "C3"], required=True)
    parser.add_argument("--output_dir", type=str, default="results/adapters")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--context_len", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after max_steps batches")
    
    args = parser.parse_args()
    train(args)
