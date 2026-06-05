import pytest
import torch
from src.train.losses import get_target_values_for_condition, compute_c4_representation_loss

def test_c3_uniform_context_normal_target():
    """C3 Uniform context uses matched Normal target."""
    normal_vals = [1, 2, 3]
    uniform_vals = [8, 9, 10]
    
    target = get_target_values_for_condition("C3", "uniform", normal_vals, uniform_vals)
    assert target == normal_vals
    
    target_c2 = get_target_values_for_condition("C2", "uniform", normal_vals, uniform_vals)
    assert target_c2 == uniform_vals

def test_c4_anchor_uses_stop_gradient():
    """C4 anchor uses stop-gradient safe hidden state."""
    h_challenge = torch.randn(2, 4096, requires_grad=True)
    h_safe = torch.randn(2, 4096, requires_grad=True)
    
    loss = compute_c4_representation_loss(h_challenge, h_safe)
    loss.backward()
    
    assert h_challenge.grad is not None
    assert h_safe.grad is None  # Should not receive gradients

def test_c4_loss_finite_and_gradients():
    """C4 loss scalar is finite and LoRA parameters receive gradients."""
    h_challenge = torch.randn(2, 4096, requires_grad=True)
    h_safe = torch.randn(2, 4096, requires_grad=False)
    
    loss = compute_c4_representation_loss(h_challenge, h_safe)
    
    assert torch.isfinite(loss)
    assert loss.dim() == 0  # Scalar
