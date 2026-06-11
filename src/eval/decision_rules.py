from typing import Any, Dict, Iterable, Mapping


def _metric(summary: Mapping[str, Mapping[str, float]], condition: str, name: str) -> float:
    try:
        return float(summary[condition][name])
    except KeyError as exc:
        raise KeyError(f"Missing {condition}.{name} in decision summary.") from exc


def check_c2_attack_validity(rows: Iterable[Mapping[str, float]]) -> Dict[str, Any]:
    """
    Validate that C2 moves toward Uniform under attack.

    The PRD validity check requires JS_to_Uniform to decrease, JS_to_Normal to
    increase, and TargetFit against the Normal target to decrease.
    """
    points = sorted(rows, key=lambda row: row["attack_step"])
    if len(points) < 2:
        return {"valid": False, "reasons": ["C2 attack validity needs at least two steps"]}

    first = points[0]
    last = points[-1]
    reasons = []
    if not last["JS_to_Uniform"] < first["JS_to_Uniform"]:
        reasons.append("JS_to_Uniform_C2 did not decrease")
    if not last["JS_to_Normal"] > first["JS_to_Normal"]:
        reasons.append("JS_to_Normal_C2 did not increase")
    if not last["TargetFit"] < first["TargetFit"]:
        reasons.append("TargetFit_C2 did not decrease")

    return {
        "valid": not reasons,
        "reasons": reasons,
        "initial": dict(first),
        "final": dict(last),
    }


def evaluate_iteration2_decision(
    summary: Mapping[str, Mapping[str, float]],
    c2_attack_valid: bool,
    benign_shift_passed: bool = True,
) -> Dict[str, Any]:
    """
    Apply the Iteration 2 success rule from the PRD.

    Summary metrics are expected at condition level with C1, C3, and C4 keys:
    AUC_TargetFit, TargetFit_half_life, sigma_rel_error, entropy_ratio, and
    collapse_rate. AUC_SPI may be present but does not determine success.
    """
    reasons = []
    valid = True

    if not c2_attack_valid:
        valid = False
        reasons.append("C2 attack validity failed")
    if not benign_shift_passed:
        reasons.append("benign Normal shift failed")

    if not _metric(summary, "C4", "AUC_TargetFit") > _metric(summary, "C3", "AUC_TargetFit"):
        reasons.append("AUC_TargetFit_C4 <= AUC_TargetFit_C3")
    if not _metric(summary, "C4", "TargetFit_half_life") > _metric(summary, "C3", "TargetFit_half_life"):
        reasons.append("TargetFit_half_life_C4 <= TargetFit_half_life_C3")
    if not _metric(summary, "C4", "sigma_rel_error") <= _metric(summary, "C3", "sigma_rel_error"):
        reasons.append("sigma_rel_error_C4 > sigma_rel_error_C3")
    if not _metric(summary, "C4", "entropy_ratio") >= _metric(summary, "C3", "entropy_ratio"):
        reasons.append("entropy_ratio_C4 < entropy_ratio_C3")
    if not _metric(summary, "C4", "collapse_rate") <= _metric(summary, "C3", "collapse_rate"):
        reasons.append("collapse_rate_C4 > collapse_rate_C3")

    c1_auc = _metric(summary, "C1", "AUC_TargetFit")
    c3_auc = _metric(summary, "C3", "AUC_TargetFit")
    c4_auc = _metric(summary, "C4", "AUC_TargetFit")
    if c1_auc >= max(c3_auc, c4_auc):
        reasons.append("C1 shortcut diagnostic failed")

    passed = valid and benign_shift_passed and not reasons
    if passed:
        decision = "Proceed to scale-up."
    elif not valid:
        decision = "Redesign benchmark."
    else:
        decision = "Revise C4."

    return {
        "valid": valid,
        "passed": passed,
        "decision": decision,
        "reasons": reasons,
    }
