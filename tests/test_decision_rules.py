from src.eval.decision_rules import evaluate_iteration2_decision


def _summary():
    return {
        "C1": {
            "AUC_TargetFit": -30.0,
            "TargetFit_half_life": 50.0,
            "sigma_rel_error": 0.70,
            "entropy_ratio": 0.60,
            "collapse_rate": 0.80,
            "AUC_SPI": 20.0,
        },
        "C2": {
            "AUC_TargetFit": -45.0,
            "TargetFit_half_life": 10.0,
            "sigma_rel_error": 0.90,
            "entropy_ratio": 0.45,
            "collapse_rate": 0.90,
            "AUC_SPI": -5.0,
        },
        "C3": {
            "AUC_TargetFit": -20.0,
            "TargetFit_half_life": 100.0,
            "sigma_rel_error": 0.35,
            "entropy_ratio": 0.82,
            "collapse_rate": 0.30,
            "AUC_SPI": 25.0,
        },
        "C4": {
            "AUC_TargetFit": -10.0,
            "TargetFit_half_life": 250.0,
            "sigma_rel_error": 0.20,
            "entropy_ratio": 0.90,
            "collapse_rate": 0.10,
            "AUC_SPI": 40.0,
        },
    }


def test_c4_passes_only_if_all_primary_metrics_beat_c3():
    decision = evaluate_iteration2_decision(
        _summary(),
        c2_attack_valid=True,
        benign_shift_passed=True,
    )

    assert decision["passed"] is True
    assert decision["decision"] == "Proceed to scale-up."


def test_c4_fails_if_only_spi_improves():
    summary = _summary()
    summary["C4"]["AUC_TargetFit"] = summary["C3"]["AUC_TargetFit"] - 1.0
    summary["C4"]["TargetFit_half_life"] = summary["C3"]["TargetFit_half_life"] - 1.0
    summary["C4"]["sigma_rel_error"] = summary["C3"]["sigma_rel_error"] + 0.10
    summary["C4"]["entropy_ratio"] = summary["C3"]["entropy_ratio"] - 0.10
    summary["C4"]["collapse_rate"] = summary["C3"]["collapse_rate"] + 0.10
    summary["C4"]["AUC_SPI"] = summary["C3"]["AUC_SPI"] + 100.0

    decision = evaluate_iteration2_decision(summary, c2_attack_valid=True)

    assert decision["passed"] is False
    assert "AUC_TargetFit_C4 <= AUC_TargetFit_C3" in decision["reasons"]


def test_c4_fails_if_collapse_rate_is_higher_than_c3():
    summary = _summary()
    summary["C4"]["collapse_rate"] = summary["C3"]["collapse_rate"] + 0.01

    decision = evaluate_iteration2_decision(summary, c2_attack_valid=True)

    assert decision["passed"] is False
    assert "collapse_rate_C4 > collapse_rate_C3" in decision["reasons"]


def test_c4_fails_if_c1_beats_c3_and_c4():
    summary = _summary()
    summary["C1"]["AUC_TargetFit"] = summary["C4"]["AUC_TargetFit"] + 1.0

    decision = evaluate_iteration2_decision(summary, c2_attack_valid=True)

    assert decision["passed"] is False
    assert "C1 shortcut diagnostic failed" in decision["reasons"]


def test_run_invalid_if_c2_does_not_move_under_attack():
    decision = evaluate_iteration2_decision(_summary(), c2_attack_valid=False)

    assert decision["passed"] is False
    assert decision["valid"] is False
    assert "C2 attack validity failed" in decision["reasons"]


def test_benign_shift_failure_blocks_success():
    decision = evaluate_iteration2_decision(
        _summary(),
        c2_attack_valid=True,
        benign_shift_passed=False,
    )

    assert decision["passed"] is False
    assert "benign Normal shift failed" in decision["reasons"]
