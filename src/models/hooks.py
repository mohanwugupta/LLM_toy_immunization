import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple

def extract_hidden_states(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    layer_indices: List[int],
    batch_size: int = 16
) -> dict:
    """
    Extract hidden states from a Hugging Face model at specific layers.
    Always extracts the hidden state of the final context token.
    
    Returns:
        dict: layer_idx -> torch.Tensor of shape (len(prompts), hidden_size)
    """
    model.eval()
    device = model.device
    
    results = {l: [] for l in layer_indices}
    
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Tokenize, left-padding is generally safer for extracting the last token
            # but since we are extracting the exact last token index, it's fine.
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs, output_hidden_states=True)
            
            # outputs.hidden_states is a tuple of (embed_out, layer1, ..., layern)
            # Find the position of the last non-pad token for each sequence
            attention_mask = inputs["attention_mask"]
            last_token_indices = attention_mask.sum(dim=1) - 1
            
            batch_indices = torch.arange(len(batch_prompts), device=device)
            
            for layer_idx in layer_indices:
                # Get hidden state for this layer
                h = outputs.hidden_states[layer_idx] # Shape: (batch, seq_len, hidden_size)
                
                # Extract the last token's hidden state
                h_last = h[batch_indices, last_token_indices, :]
                results[layer_idx].append(h_last.cpu())
                
    # Concatenate results
    for layer_idx in layer_indices:
        results[layer_idx] = torch.cat(results[layer_idx], dim=0)
        
    return results
