import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any

from .generate_numeric import generate_matched_pair, render_as_comma_delimited
from .tokenize_numbers import extract_valid_support, get_valid_number_tokens

class NumericDataset(Dataset):
    """
    Generates datasets on the fly based on a grid of mu and sigma values.
    Returns matched pairs of Normal and Uniform data.
    """
    def __init__(self, mu_values: List[float], sigma_values: List[float], length: int = 128, seeds: List[int] = [42], num_examples_per_condition: int = 100):
        self.examples = []
        
        # We pre-generate the dataset in memory
        for seed in seeds:
            for mu in mu_values:
                for sigma in sigma_values:
                    # Generate a few examples for each condition
                    for i in range(num_examples_per_condition):
                        pair = generate_matched_pair(mu=mu, sigma=sigma, length=length, seed=seed + i)
                        self.examples.append(pair)
                        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx) -> Dict[str, Any]:
        return self.examples[idx]

def collate_numeric(batch: List[Dict[str, Any]], tokenizer, condition: str = "C1", prompt_family: str = "normal") -> Dict[str, torch.Tensor]:
    """
    Collates a batch of matched pairs into tokenized tensors suitable for training.
    """
    from src.train.losses import get_target_values_for_condition
    
    input_texts = []
    target_texts = []
    
    for item in batch:
        # Get context based on the requested prompt_family
        context_values = item[prompt_family]["values"][:-1]
        next_value = item[prompt_family]["values"][-1]
        
        matched_normal_values = item["normal"]["values"]
        uniform_values = item["uniform"]["values"]
        
        # Get target sequence
        target_values = get_target_values_for_condition(
            condition=condition,
            prompt_family=prompt_family,
            matched_normal_values=matched_normal_values,
            uniform_values=uniform_values
        )
        target_next_value = target_values[-1]
        
        input_texts.append(render_as_comma_delimited(context_values))
        target_texts.append(str(target_next_value))
        
    # Tokenize
    inputs = tokenizer(input_texts, padding=True, return_tensors="pt", add_special_tokens=True)
    
    # We want to train on next-token prediction of target_texts
    # In a real autoregressive setting, we pass the full sequence and compute loss only on the target tokens.
    # We'll construct full sequences: prompt + target
    full_texts = [inp + tgt for inp, tgt in zip(input_texts, target_texts)]
    full_encodings = tokenizer(full_texts, padding=True, return_tensors="pt", add_special_tokens=True)
    
    # Labels should be -100 for context, and the target token id for the target token
    labels = full_encodings["input_ids"].clone()
    prompt_encodings = tokenizer(input_texts, padding=True, return_tensors="pt", add_special_tokens=True)
    
    for i in range(len(batch)):
        prompt_len = len(prompt_encodings["input_ids"][i])
        # Find where padding ends, or just mask out the prompt length
        # Assuming left padding or right padding... let's just mask prompt tokens
        labels[i, :prompt_len] = -100
        
        # Also mask padding tokens
        labels[i][full_encodings["attention_mask"][i] == 0] = -100

    return {
        "input_ids": full_encodings["input_ids"],
        "attention_mask": full_encodings["attention_mask"],
        "labels": labels
    }
