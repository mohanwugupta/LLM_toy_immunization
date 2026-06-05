import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.models.load_hf import load_base_model
from src.models.lora import prepare_lora_model
from src.data.datasets import NumericDataset, collate_numeric
from src.train.losses import compute_c4_representation_loss

def train_c4(args):
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    model.train()
    
    print(f"Starting C4 training...")
    
    layer_idx = args.target_layer
    lambda_rep = args.lambda_rep
    
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            # For C4, we sample Uniform contexts
            # We first need the safe anchors (Normal contexts)
            
            # 1. Forward pass for Normal contexts (no grad needed for anchor)
            safe_inputs = collate_numeric(batch, tokenizer, condition="C4", prompt_family="normal")
            safe_input_ids = safe_inputs["input_ids"].to(device)
            safe_attention_mask = safe_inputs["attention_mask"].to(device)
            
            with torch.no_grad():
                safe_outputs = model(
                    input_ids=safe_input_ids, 
                    attention_mask=safe_attention_mask, 
                    output_hidden_states=True
                )
                safe_h = safe_outputs.hidden_states[layer_idx]
                
                # Get last token of context (where padding ends, or just the end of the prompt)
                # For simplicity in MVP, we just take the last token of the whole sequence or prompt
                # Let's extract the token right before the target token
                # Actually, collate_numeric creates a full prompt + target. 
                # We need the hidden state at the last token of the context.
                # To be exact, the mask has -100 for context tokens in `labels`. 
                # The first non -100 is the target token. The token right before it is the last context token.
                labels = safe_inputs["labels"].to(device)
                last_context_token_idx = (labels != -100).long().argmax(dim=1) - 1
                
                batch_indices = torch.arange(len(batch), device=device)
                safe_h_last = safe_h[batch_indices, last_context_token_idx, :]
            
            # 2. Forward pass for Uniform contexts (with grad)
            challenge_inputs = collate_numeric(batch, tokenizer, condition="C4", prompt_family="uniform")
            challenge_input_ids = challenge_inputs["input_ids"].to(device)
            challenge_attention_mask = challenge_inputs["attention_mask"].to(device)
            challenge_labels = challenge_inputs["labels"].to(device)
            
            optimizer.zero_grad()
            
            challenge_outputs = model(
                input_ids=challenge_input_ids, 
                attention_mask=challenge_attention_mask, 
                labels=challenge_labels,
                output_hidden_states=True
            )
            task_loss = challenge_outputs.loss
            challenge_h = challenge_outputs.hidden_states[layer_idx]
            
            last_context_token_idx_challenge = (challenge_labels != -100).long().argmax(dim=1) - 1
            challenge_h_last = challenge_h[batch_indices, last_context_token_idx_challenge, :]
            
            rep_loss = compute_c4_representation_loss(challenge_h_last, safe_h_last, lambda_rep=lambda_rep)
            
            total_loss = task_loss + rep_loss
            total_loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Task: {task_loss.item():.4f} | Rep: {rep_loss.item():.4f}")
                
    output_dir = os.path.join(args.output_dir, "C4")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Saved C4 adapter to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output_dir", type=str, default="results/adapters")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--context_len", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_layer", type=int, default=-1)
    parser.add_argument("--lambda_rep", type=float, default=0.1)
    
    args = parser.parse_args()
    train_c4(args)
