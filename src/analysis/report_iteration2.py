import argparse
import json
from pathlib import Path
from typing import Dict

from src.eval.decision_rules import evaluate_iteration2_decision


REPORT_SECTIONS = [
    "1. Executive summary",
    "2. Decision",
    "3. What changed from Iteration 1",
    "4. Pre-registered success criteria",
    "5. Data and task variants",
    "6. Attack protocols",
    "7. Metric definitions",
    "8. C2 attack validity check",
    "9. C1 shortcut diagnostic",
    "10. C3 vs C4 primary comparison",
    "11. Anti-collapse analysis",
    "12. Context-length analysis",
    "13. Difficulty-ladder analysis",
    "14. Benign adaptation control",
    "15. Representational analysis",
    "16. Compute and tractability",
    "17. Failure modes",
    "18. Final recommendation",
]


def _load_condition_summary(metrics_dir: Path) -> Dict[str, Dict[str, float]]:
    path = metrics_dir / "condition_decision_summary.json"
    with open(path, "r") as f:
        return json.load(f)


def generate_report(
    metrics_dir: Path,
    output_dir: Path,
    c2_attack_valid: bool = True,
    benign_shift_passed: bool = True,
) -> Dict[str, Path]:
    summary = _load_condition_summary(metrics_dir)
    decision = evaluate_iteration2_decision(
        summary,
        c2_attack_valid=c2_attack_valid,
        benign_shift_passed=benign_shift_passed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    decision_path = output_dir / "decision_summary.json"
    report_path = output_dir / "report_iteration2.md"

    with open(decision_path, "w") as f:
        json.dump(
            {
                "decision": decision,
                "condition_summary": summary,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    with open(report_path, "w") as f:
        f.write("# Iteration 2 Anti-Collapse Durability Report\n\n")
        for section in REPORT_SECTIONS:
            f.write(f"## {section}\n\n")
            if section == "1. Executive summary":
                f.write(
                    "This report evaluates whether C4 preserves the full Normal target "
                    "distribution under attack after applying anti-collapse metrics.\n\n"
                )
            elif section == "2. Decision":
                f.write(f"{decision['decision']}\n\n")
                if decision["reasons"]:
                    f.write("Failed or blocking criteria:\n\n")
                    for reason in decision["reasons"]:
                        f.write(f"- {reason}\n")
                    f.write("\n")
            elif section == "4. Pre-registered success criteria":
                f.write(
                    "- C4 must beat C3 on AUC_TargetFit and TargetFit half-life.\n"
                    "- C4 must not be worse on sigma error, entropy ratio, or collapse rate.\n"
                    "- C1 must not be the best TargetFit condition.\n"
                    "- C2 must validate that the attack moves toward Uniform.\n\n"
                )
            elif section == "10. C3 vs C4 primary comparison":
                f.write("```json\n")
                f.write(json.dumps({k: summary.get(k, {}) for k in ["C3", "C4"]}, indent=2))
                f.write("\n```\n\n")
            elif section == "18. Final recommendation":
                f.write(f"{decision['decision']}\n\n")
            else:
                f.write("See aggregate metrics and generated figures for this section.\n\n")

    return {
        "decision": decision_path,
        "report": report_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", default="results/iteration2/local/metrics")
    parser.add_argument("--output_dir", default="results/iteration2/local")
    parser.add_argument("--c2_attack_invalid", action="store_true")
    parser.add_argument("--benign_shift_failed", action="store_true")
    args = parser.parse_args()

    paths = generate_report(
        metrics_dir=Path(args.metrics_dir),
        output_dir=Path(args.output_dir),
        c2_attack_valid=not args.c2_attack_invalid,
        benign_shift_passed=not args.benign_shift_failed,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
