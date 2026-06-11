from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

import numpy as np

def _as_probability_vector(values: Sequence[float], epsilon: float = 1e-10) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    total = float(np.sum(arr))
    if total <= epsilon:
        return np.ones_like(arr, dtype=float) / len(arr)
    return arr / total


def _safe_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute KL(p || q) safely."""
    p = p / (np.sum(p) + epsilon)
    q = q / (np.sum(q) + epsilon)
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))

def _safe_js_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute JS(p || q) safely."""
    p = p / (np.sum(p) + epsilon)
    q = q / (np.sum(q) + epsilon)
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    return 0.5 * _safe_kl_divergence(p, m, epsilon) + 0.5 * _safe_kl_divergence(q, m, epsilon)


def normal_target_distribution(support: Sequence[int], mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-10)
    x_arr = np.asarray(support, dtype=float)
    unnormalized = np.exp(-0.5 * ((x_arr - float(mu)) / sigma) ** 2)
    return _as_probability_vector(unnormalized)


def uniform_target_distribution(support: Sequence[int], mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-10)
    x_arr = np.asarray(support, dtype=float)
    low = float(mu) - np.sqrt(3) * sigma
    high = float(mu) + np.sqrt(3) * sigma
    unnormalized = np.where((x_arr >= low) & (x_arr <= high), 1.0, 0.0)
    return _as_probability_vector(unnormalized)


def probability_dict_to_vector(
    q_probs: Dict[int, float],
    support: Optional[Set[int]] = None,
) -> Tuple[np.ndarray, list[int]]:
    if support is None:
        support = set(q_probs.keys())
    support_list = sorted(support)
    q = _as_probability_vector([q_probs.get(x, 0.0) for x in support_list])
    return q, support_list

def compute_all_metrics(
    q_probs: Dict[int, float], 
    mu: float, 
    sigma: float, 
    support: Set[int] = None
) -> Dict[str, Any]:
    if support is None:
        support = set(q_probs.keys())
        
    q, support_list = probability_dict_to_vector(q_probs, support)
    x_arr = np.array(support_list)
    p_normal_target = normal_target_distribution(support_list, mu, sigma)
    q_uniform = uniform_target_distribution(support_list, mu, sigma)
    
    # 1. TargetFit
    js_to_normal = _safe_js_divergence(q, p_normal_target)
    js_to_uniform = _safe_js_divergence(q, q_uniform)
    TargetFit = -js_to_normal
    
    # 2. Forward and Reverse KL
    ForwardKL_Normal = _safe_kl_divergence(p_normal_target, q)
    ReverseKL_Normal = _safe_kl_divergence(q, p_normal_target)
    
    # 3. SPI
    kl_uniform = _safe_kl_divergence(q, q_uniform)
    kl_normal = _safe_kl_divergence(q, p_normal_target)
    SPI = kl_uniform - kl_normal
    
    # 4. Statistical Estimators
    mu_q = np.sum(q * x_arr)
    mu_error = abs(mu_q - mu)
    
    sigma_q = np.sqrt(np.sum(q * (x_arr - mu_q)**2))
    sigma_rel_error = abs(sigma_q - sigma) / sigma
    
    # 5. Entropy
    eps = 1e-10
    entropy_q = -np.sum(q * np.log(np.clip(q, eps, 1.0)))
    entropy_target = -np.sum(p_normal_target * np.log(np.clip(p_normal_target, eps, 1.0)))
    entropy_ratio = entropy_q / entropy_target if entropy_target > 0 else 0.0
    
    # 6. Collapse flags
    collapse_score = max(0, sigma / max(sigma_q, eps))
    collapsed = bool(sigma_rel_error > 0.50 or entropy_ratio < 0.70)
    
    return {
        "TargetFit": TargetFit,
        "target_fit": TargetFit,
        "JS_to_Normal": js_to_normal,
        "JS_to_Uniform": js_to_uniform,
        "ForwardKL_Normal": ForwardKL_Normal,
        "ReverseKL_Normal": ReverseKL_Normal,
        "SPI": SPI,
        "mu_q": mu_q,
        "mu_error": mu_error,
        "sigma_q": sigma_q,
        "sigma_rel_error": sigma_rel_error,
        "entropy_q": entropy_q,
        "entropy_target": entropy_target,
        "entropy_ratio": entropy_ratio,
        "collapse_score": collapse_score,
        "collapsed": collapsed
    }


def compute_targetfit_half_life(
    curve: Iterable[Tuple[int, float]],
    tf_floor: float,
) -> Dict[str, Any]:
    points = sorted((int(step), float(targetfit)) for step, targetfit in curve)
    if not points:
        raise ValueError("TargetFit curve must contain at least one point.")

    tf_0 = points[0][1]
    threshold = float(tf_floor) + 0.5 * (tf_0 - float(tf_floor))
    for step, targetfit in points:
        if targetfit <= threshold:
            return {
                "tf_0": tf_0,
                "tf_floor": float(tf_floor),
                "threshold": threshold,
                "half_life_step": step,
                "right_censored": False,
            }

    return {
        "tf_0": tf_0,
        "tf_floor": float(tf_floor),
        "threshold": threshold,
        "half_life_step": None,
        "right_censored": True,
    }
