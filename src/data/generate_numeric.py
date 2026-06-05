import numpy as np
from typing import List, Tuple, Dict, Any
import hashlib
import uuid

def render_as_comma_delimited(values: List[int]) -> str:
    """Render a list of integers as a comma-delimited string ending with a comma."""
    return ",".join(map(str, values)) + ","

def generate_normal_sequence(mu: float, sigma: float, length: int, rng: np.random.Generator) -> List[int]:
    """Generate a sequence of numbers from a Normal distribution, rounded and clamped."""
    values = rng.normal(mu, sigma, length)
    values = np.round(values).astype(int)
    values = np.clip(values, 0, 999)
    return values.tolist()

def generate_uniform_sequence(mu: float, sigma: float, length: int, rng: np.random.Generator) -> List[int]:
    """Generate a sequence of numbers from a Uniform distribution with matched variance, rounded and clamped."""
    low = mu - np.sqrt(3) * sigma
    high = mu + np.sqrt(3) * sigma
    values = rng.uniform(low, high, length)
    values = np.round(values).astype(int)
    values = np.clip(values, 0, 999)
    return values.tolist()

def generate_matched_pair(mu: float, sigma: float, length: int, seed: int = None) -> Dict[str, Any]:
    """Generate a matched pair of Normal and Uniform examples."""
    rng = np.random.default_rng(seed)
    
    # Generate an example ID (stable based on properties + seed if provided)
    example_id_str = f"{mu}_{sigma}_{length}_{seed}"
    example_id = hashlib.md5(example_id_str.encode()).hexdigest()
    
    normal_seq = generate_normal_sequence(mu, sigma, length, rng)
    # Important: continue using same rng for uniform to keep it deterministic but distinct from normal
    uniform_seq = generate_uniform_sequence(mu, sigma, length, rng)
    
    return {
        "example_id": example_id,
        "mu": mu,
        "sigma": sigma,
        "length": length,
        "normal": {
            "family": "normal",
            "values": normal_seq,
            "prompt": render_as_comma_delimited(normal_seq)
        },
        "uniform": {
            "family": "uniform",
            "values": uniform_seq,
            "prompt": render_as_comma_delimited(uniform_seq)
        }
    }
