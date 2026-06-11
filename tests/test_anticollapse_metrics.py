import pytest
import numpy as np
from typing import Dict, Set

# We will import the new metrics module once it's created.
from src.eval.anti_collapse_metrics import (
    compute_all_metrics,
    compute_targetfit_half_life,
    _safe_js_divergence,
    _safe_kl_divergence,
)

def build_q(support, fn) -> Dict[int, float]:
    """Helper to build normalized probabilities"""
    unnorm = {x: fn(x) for x in support}
    total = sum(unnorm.values())
    if total == 0:
        return {x: 1.0/len(support) for x in support}
    return {x: v / total for x, v in unnorm.items()}

@pytest.fixture
def base_params():
    support = set(range(0, 1000))
    mu = 500.0
    sigma = 100.0
    return support, mu, sigma

def test_delta_at_mu_can_have_high_spi_but_fails_targetfit(base_params):
    support, mu, sigma = base_params
    # Delta at mu
    q_delta = build_q(support, lambda x: 1.0 if x == int(mu) else 0.0)
    
    metrics = compute_all_metrics(q_delta, mu, sigma, support)
    
    # Delta scores better (higher SPI) because Normal peak is higher than Uniform peak
    assert metrics['SPI'] > 0
    
    # But TargetFit (-JS) should be terrible (close to -0.69 / -1.0) compared to a true Normal
    # A true normal would have TargetFit near 0.0.
    q_normal = build_q(support, lambda x: np.exp(-0.5 * ((x - mu)/sigma)**2))
    normal_metrics = compute_all_metrics(q_normal, mu, sigma, support)
    
    assert metrics['TargetFit'] < normal_metrics['TargetFit']

def test_delta_at_mu_fails_sigma_preservation(base_params):
    support, mu, sigma = base_params
    q_delta = build_q(support, lambda x: 1.0 if x == int(mu) else 0.0)
    
    metrics = compute_all_metrics(q_delta, mu, sigma, support)
    
    # Sigma is 0, so error is 1.0 (100%)
    assert metrics['sigma_rel_error'] > 0.99
    
def test_delta_at_mu_fails_entropy_preservation(base_params):
    support, mu, sigma = base_params
    q_delta = build_q(support, lambda x: 1.0 if x == int(mu) else 0.0)
    
    metrics = compute_all_metrics(q_delta, mu, sigma, support)
    
    # Entropy of delta is 0
    assert metrics['entropy_ratio'] < 0.1

def test_true_normal_passes_targetfit(base_params):
    support, mu, sigma = base_params
    q_normal = build_q(support, lambda x: np.exp(-0.5 * ((x - mu)/sigma)**2))
    
    metrics = compute_all_metrics(q_normal, mu, sigma, support)
    # JS should be essentially 0, so TargetFit = 0.
    assert metrics['TargetFit'] > -0.01

def test_true_uniform_fails_normal_targetfit(base_params):
    support, mu, sigma = base_params
    low = mu - np.sqrt(3) * sigma
    high = mu + np.sqrt(3) * sigma
    q_uniform = build_q(support, lambda x: 1.0 if low <= x <= high else 0.0)
    
    metrics = compute_all_metrics(q_uniform, mu, sigma, support)
    
    q_normal = build_q(support, lambda x: np.exp(-0.5 * ((x - mu)/sigma)**2))
    normal_metrics = compute_all_metrics(q_normal, mu, sigma, support)
    
    assert metrics['TargetFit'] < normal_metrics['TargetFit'] - 0.01

def test_js_is_symmetric_and_bounded():
    p = np.array([0.1, 0.9])
    q = np.array([0.9, 0.1])
    js1 = _safe_js_divergence(p, q)
    js2 = _safe_js_divergence(q, p)
    assert np.isclose(js1, js2)
    assert 0 <= js1 <= 1.0  # JS divergence (base e) is bounded by ln(2) approx 0.69, so <= 1.0

def test_forward_kl_penalizes_missing_tails(base_params):
    support, mu, sigma = base_params
    # q only has mass strictly in the center
    q_missing_tails = build_q(support, lambda x: 1.0 if mu - 10 <= x <= mu + 10 else 0.0)
    q_normal = build_q(support, lambda x: np.exp(-0.5 * ((x - mu)/sigma)**2))
    
    # KL(p_target || q)
    metrics_bad = compute_all_metrics(q_missing_tails, mu, sigma, support)
    metrics_good = compute_all_metrics(q_normal, mu, sigma, support)
    
    assert metrics_bad['ForwardKL_Normal'] > metrics_good['ForwardKL_Normal']

def test_collapse_score_increases_when_sigma_shrinks(base_params):
    support, mu, sigma = base_params
    q_wide = build_q(support, lambda x: np.exp(-0.5 * ((x - mu)/100.0)**2))
    q_narrow = build_q(support, lambda x: np.exp(-0.5 * ((x - mu)/10.0)**2))
    
    m_wide = compute_all_metrics(q_wide, mu, sigma, support)
    m_narrow = compute_all_metrics(q_narrow, mu, sigma, support)
    
    assert m_narrow['collapse_score'] > m_wide['collapse_score']

def test_decision_rule_rejects_high_spi_low_entropy_case(base_params):
    support, mu, sigma = base_params
    q_delta = build_q(support, lambda x: 1.0 if x == int(mu) else 0.0)
    
    metrics = compute_all_metrics(q_delta, mu, sigma, support)
    
    assert metrics['SPI'] > 0
    assert metrics['collapsed'] is True


def test_metrics_include_js_to_normal_and_uniform(base_params):
    support, mu, sigma = base_params
    q_normal = build_q(support, lambda x: np.exp(-0.5 * ((x - mu)/sigma)**2))

    metrics = compute_all_metrics(q_normal, mu, sigma, support)

    assert metrics["JS_to_Normal"] == pytest.approx(-metrics["TargetFit"])
    assert metrics["JS_to_Uniform"] > metrics["JS_to_Normal"]
    assert metrics["entropy_target"] > 0


def test_targetfit_half_life_marks_right_censored_when_no_crossing():
    result = compute_targetfit_half_life(
        [(0, -0.05), (10, -0.06), (50, -0.08)],
        tf_floor=-0.40,
    )

    assert result["half_life_step"] is None
    assert result["right_censored"] is True


def test_targetfit_half_life_returns_first_crossing():
    result = compute_targetfit_half_life(
        [(0, -0.05), (10, -0.20), (50, -0.30)],
        tf_floor=-0.35,
    )

    assert result["threshold"] == pytest.approx(-0.20)
    assert result["half_life_step"] == 10
    assert result["right_censored"] is False
