import pytest
import numpy as np
from src.eval.compute_spi import compute_spi, _safe_kl_divergence

def test_spi_uniform_distribution():
    """SPI should be negative for a known Uniform distribution."""
    mu, sigma = 500, 100
    low = mu - np.sqrt(3) * sigma
    high = mu + np.sqrt(3) * sigma
    
    # Create a uniform distribution
    q_probs = {}
    for i in range(1000):
        if low <= i <= high:
            q_probs[i] = 1.0
            
    spi = compute_spi(q_probs, mu, sigma, support=set(range(1000)))
    assert spi < 0

def test_spi_normal_distribution():
    """SPI should be positive for a known Normal distribution."""
    mu, sigma = 500, 100
    
    # Create a normal distribution
    q_probs = {}
    for i in range(1000):
        val = np.exp(-0.5 * ((i - mu) / sigma)**2)
        q_probs[i] = val
        
    spi = compute_spi(q_probs, mu, sigma, support=set(range(1000)))
    assert spi > 0

def test_spi_kl_numerical_stability():
    """KL should not have NaNs for zero probabilities."""
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0])
    kl = _safe_kl_divergence(p, q)
    
    assert not np.isnan(kl)
    assert not np.isinf(kl)
    assert kl > 0
