import numpy as np
from typing import Dict, Set

def _safe_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute KL(p || q) safely."""
    # Ensure probabilities sum to 1
    p = p / (np.sum(p) + epsilon)
    q = q / (np.sum(q) + epsilon)
    
    # Avoid zero probabilities
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    
    # Re-normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return float(np.sum(p * np.log(p / q)))

def compute_spi(
    q_probs: Dict[int, float], 
    mu: float, 
    sigma: float, 
    support: Set[int] = None
) -> float:
    """
    Compute Safe Prior Index (SPI).
    SPI = KL(q || Uniform_best_fit) - KL(q || Normal_best_fit)
    
    Args:
        q_probs: Dictionary mapping integer token to its probability.
        mu: The mean of the Normal family target.
        sigma: The standard deviation of the Normal family target.
        support: The set of valid integer tokens. If None, uses keys of q_probs.
    """
    if support is None:
        support = set(q_probs.keys())
        
    support_list = sorted(list(support))
    
    # Extract p (the predicted probabilities)
    p = np.array([q_probs.get(x, 0.0) for x in support_list])
    if np.sum(p) > 0:
        p = p / np.sum(p)
    else:
        # Uniform if empty (shouldn't happen)
        p = np.ones(len(support_list)) / len(support_list)
        
    # Generate Normal_best_fit
    # PDF of Normal: exp(-0.5 * ((x - mu)/sigma)^2)
    normal_unnorm = np.exp(-0.5 * ((np.array(support_list) - mu) / sigma)**2)
    q_normal = normal_unnorm / np.sum(normal_unnorm)
    
    # Generate Uniform_best_fit
    # Uniform matched on mean and variance: low = mu - sqrt(3)*sigma, high = mu + sqrt(3)*sigma
    low = mu - np.sqrt(3) * sigma
    high = mu + np.sqrt(3) * sigma
    uniform_unnorm = np.where((np.array(support_list) >= low) & (np.array(support_list) <= high), 1.0, 0.0)
    
    if np.sum(uniform_unnorm) == 0:
        # Fallback if no support in range (unlikely)
        uniform_unnorm = np.ones(len(support_list))
        
    q_uniform = uniform_unnorm / np.sum(uniform_unnorm)
    
    kl_uniform = _safe_kl_divergence(p, q_uniform)
    kl_normal = _safe_kl_divergence(p, q_normal)
    
    return kl_uniform - kl_normal
