import torch
import torch.nn.functional as F
from typing import List

def get_target_values_for_condition(
    condition: str, 
    prompt_family: str, 
    matched_normal_values: List[int], 
    uniform_values: List[int]
) -> List[int]:
    """
    Returns the target sequence of integers for a given training condition.
    
    C1: Normal contexts -> Normal targets
    C2: Normal -> Normal, Uniform -> Uniform
    C3/C4: Normal -> Normal, Uniform -> Matched Normal
    """
    condition = condition.upper()
    if condition == "C1":
        if prompt_family != "normal":
            raise ValueError("C1 should only see normal contexts during training.")
        return matched_normal_values
    elif condition == "C2":
        return matched_normal_values if prompt_family == "normal" else uniform_values
    elif condition in ["C3", "C4"]:
        return matched_normal_values
    else:
        raise ValueError(f"Unknown condition: {condition}")

def compute_c4_representation_loss(
    h_challenge: torch.Tensor,
    h_safe: torch.Tensor,
    lambda_rep: float = 0.1
) -> torch.Tensor:
    """
    Compute the C4 representation anchoring loss.
    
    L_C4 = lambda_rep * mean_squared_error(
        normalize(h_challenge),
        stop_gradient(normalize(h_safe))
    )
    """
    # stop_gradient on the safe anchor
    h_safe_detached = h_safe.detach()
    
    # Normalize features
    h_challenge_norm = F.normalize(h_challenge, p=2, dim=-1)
    h_safe_norm = F.normalize(h_safe_detached, p=2, dim=-1)
    
    # MSE loss
    mse = F.mse_loss(h_challenge_norm, h_safe_norm)
    
    return lambda_rep * mse
