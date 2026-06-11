import pytest
import torch
import torch.nn.functional as F
from src.attack.losses import (
    compute_distributional_loss,
    compute_mixed_attack_loss,
    compute_sampled_attack_loss,
    get_uniform_target,
    next_token_logits_from_labels,
)
from src.data.tokenize_numbers import get_valid_number_tokens

@pytest.fixture
def dummy_vocab_size():
    return 32000

@pytest.fixture
def mock_tokenizer():
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 32000
    return MockTokenizer()

@pytest.fixture
def valid_tokens():
    # just mock some valid tokens
    return {i: 1000 + i for i in range(100)}

def test_distributional_uniform_target_has_correct_support(dummy_vocab_size, valid_tokens):
    mu = 50.0
    sigma = 10.0
    p_uniform = get_uniform_target(mu, sigma, valid_tokens, dummy_vocab_size)
    
    assert p_uniform.shape == (dummy_vocab_size,)
    assert torch.isclose(p_uniform.sum(), torch.tensor(1.0))
    
    # Check boundaries: mu +/- sqrt(3)*sigma = 50 +/- 17.32 -> 32.68 to 67.32
    # Integers in [33, 67] should be non-zero
    for num, token_id in valid_tokens.items():
        if 33 <= num <= 67:
            assert p_uniform[token_id] > 0
        else:
            assert p_uniform[token_id] == 0.0

def test_distributional_attack_loss_uses_full_uniform_distribution(dummy_vocab_size, valid_tokens):
    mu = 50.0
    sigma = 10.0
    p_uniform = get_uniform_target(mu, sigma, valid_tokens, dummy_vocab_size).unsqueeze(0)
    
    # logits shape (1, vocab_size)
    logits = torch.randn(1, dummy_vocab_size, requires_grad=True)
    loss = compute_distributional_loss(logits, p_uniform)
    
    assert loss.item() >= 0
    assert loss.requires_grad

def test_sampled_attack_uses_token_targets():
    logits = torch.tensor([[0.0, 0.0, 5.0]], requires_grad=True)
    labels = torch.tensor([2])

    loss = compute_sampled_attack_loss(logits, labels)

    assert loss.item() < 0.02
    loss.backward()
    assert logits.grad is not None

def test_mixed_attack_combines_sft_and_kl_terms(dummy_vocab_size, valid_tokens):
    mu = 50.0
    sigma = 10.0
    p_uniform = get_uniform_target(mu, sigma, valid_tokens, dummy_vocab_size).unsqueeze(0)
    
    logits = torch.randn(1, dummy_vocab_size, requires_grad=True)
    kl_loss = compute_distributional_loss(logits, p_uniform)
    
    # Simulate a sampled SFT loss
    labels = torch.tensor([valid_tokens[50]])
    sft_loss = F.cross_entropy(logits, labels)
    
    mixed_loss = compute_mixed_attack_loss(logits, labels, p_uniform)
    assert mixed_loss.item() == pytest.approx((0.5 * kl_loss + 0.5 * sft_loss).item())
    
    mixed_loss.backward()
    assert logits.grad is not None

def test_attack_metadata_records_attack_type():
    from src.attack.run_attack import build_attack_metadata

    metadata = build_attack_metadata(
        condition="C4",
        seed=42,
        attack_type="distributional",
        attack_lr=3e-5,
        attack_step=500,
        difficulty_level=3,
    )

    assert metadata["attack_type"] == "distributional"
    assert metadata["attack_lr"] == 3e-5
    assert metadata["difficulty_level"] == 3

def test_attack_checkpoint_paths_include_attack_type_and_lr():
    # We will test this by ensuring the output path builder function works
    from src.attack.run_attack import build_attack_output_dir
    path = build_attack_output_dir("results/attack", "distributional", 1e-4)
    assert "distributional" in path
    assert "1e-04" in path or "0.0001" in path

def test_c2_moves_toward_uniform_on_tiny_synthetic_case():
    from src.eval.decision_rules import check_c2_attack_validity

    result = check_c2_attack_validity(
        [
            {"attack_step": 0, "JS_to_Uniform": 0.30, "JS_to_Normal": 0.10, "TargetFit": -0.10},
            {"attack_step": 100, "JS_to_Uniform": 0.12, "JS_to_Normal": 0.25, "TargetFit": -0.25},
        ]
    )

    assert result["valid"] is True


def test_distributional_logits_use_position_before_target():
    logits = torch.arange(2 * 5 * 7, dtype=torch.float32).reshape(2, 5, 7)
    labels = torch.full((2, 5), -100)
    labels[0, 3] = 4
    labels[1, 2] = 5

    selected = next_token_logits_from_labels(logits, labels)

    assert torch.equal(selected[0], logits[0, 2])
    assert torch.equal(selected[1], logits[1, 1])
