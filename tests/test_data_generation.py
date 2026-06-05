import pytest
import numpy as np
from src.data.generate_numeric import (
    generate_normal_sequence,
    generate_uniform_sequence,
    generate_matched_pair,
    render_as_comma_delimited
)

def test_normal_generator_valid_values():
    """Normal generator should return valid integer values in [0, 999]"""
    rng = np.random.default_rng(42)
    values = generate_normal_sequence(mu=500, sigma=100, length=1000, rng=rng)
    assert all(isinstance(v, int) for v in values)
    assert all(0 <= v <= 999 for v in values)

def test_uniform_generator_variance():
    """Uniform generator should match target mean/variance before clamping"""
    rng = np.random.default_rng(42)
    mu, sigma = 500, 100
    # Use a large sample size so the sample stats match theoretical
    values = generate_uniform_sequence(mu, sigma, length=100000, rng=rng)
    
    sample_mean = np.mean(values)
    sample_std = np.std(values)
    
    # It should be close to the target mu and sigma
    assert np.isclose(sample_mean, mu, atol=1.0)
    assert np.isclose(sample_std, sigma, atol=1.0)

def test_matched_pairs_metadata():
    """Matched Normal/Uniform pairs should share mu and sigma metadata"""
    pair = generate_matched_pair(mu=400, sigma=80, length=10, seed=42)
    assert pair["mu"] == 400
    assert pair["sigma"] == 80
    assert "normal" in pair and "uniform" in pair
    assert pair["normal"]["family"] == "normal"
    assert pair["uniform"]["family"] == "uniform"

def test_context_rendering():
    """Context rendering should end with a final comma"""
    values = [533, 460, 689, 432]
    prompt = render_as_comma_delimited(values)
    assert prompt == "533,460,689,432,"
    assert prompt.endswith(",")

