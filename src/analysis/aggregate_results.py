import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.eval.anti_collapse_metrics import compute_all_metrics, compute_targetfit_half_life
from src.eval.vllm_schema import EvalSchemaError, normalize_number_logprobs, validate_eval_record


METRIC_COLUMNS = [
    "TargetFit",
    "JS_to_Normal",
    "JS_to_Uniform",
    "ForwardKL_Normal",
    "ReverseKL_Normal",
    "SPI",
    "mu_q",
    "mu_error",
    "sigma_q",
    "sigma_rel_error",
    "entropy_q",
    "entropy_target",
    "entropy_ratio",
    "collapse_score",
]


GROUP_COLUMNS = [
    "run_id",
    "condition",
    "seed",
    "attack_type",
    "attack_lr",
    "difficulty_level",
    "context_len",
    "attack_step",
]


def parse_jsonl(file_path: Path) -> List[dict]:
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def discover_eval_outputs(input_roots: Iterable[Path]) -> List[Path]:
    files = []
    for root in input_roots:
        if root.exists():
            files.extend(root.glob("**/eval_outputs.jsonl"))
    return sorted(set(files))


def _fallback_probs_from_raw_top_logprobs(row: Mapping[str, object]) -> Dict[int, float]:
    q_probs = {}
    raw = row.get("raw_top_logprobs", {})
    if not isinstance(raw, Mapping):
        return q_probs
    for token_or_number, logprob in raw.items():
        try:
            number = int(token_or_number)
        except ValueError:
            continue
        if 0 <= number <= 999:
            q_probs[number] = float(np.exp(float(logprob)))
    total = sum(q_probs.values())
    if total > 0:
        q_probs = {number: prob / total for number, prob in q_probs.items()}
    return q_probs


def record_to_metric_row(row: Mapping[str, object], support: Optional[set] = None) -> dict:
    if support is None:
        support = set(range(1000))

    q_probs = {}
    if row.get("number_logprobs"):
        validate_eval_record(row)
        q_probs = normalize_number_logprobs(row["number_logprobs"])
    else:
        q_probs = _fallback_probs_from_raw_top_logprobs(row)
        if not q_probs:
            raise EvalSchemaError("record has no usable number logprobs")

    mu_target = float(row.get("mu_target", row.get("mu", 0.0)))
    sigma_target = float(row.get("sigma_target", row.get("sigma", 1.0)))
    metrics = compute_all_metrics(q_probs, mu_target, sigma_target, support=support)

    metric_row = {
        "run_id": row.get("run_id", ""),
        "condition": row.get("condition", ""),
        "seed": int(row.get("seed", 0)),
        "attack_type": row.get("attack_type", "sampled"),
        "attack_lr": float(row.get("attack_lr", 3e-5)),
        "attack_step": int(row.get("attack_step", 0)),
        "difficulty_level": int(row.get("difficulty_level", 3)),
        "context_len": int(row.get("context_len", 0)),
        "family": row.get("family", "unknown"),
        "target_family": row.get("target_family", "normal"),
        "prompt_id": row.get("prompt_id", ""),
        "mu_target": mu_target,
        "sigma_target": sigma_target,
        "collapsed": bool(metrics["collapsed"]),
    }
    metric_row.update({column: metrics[column] for column in METRIC_COLUMNS})
    return metric_row


def aggregate_metric_rows(metric_rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(metric_rows)
    if df.empty:
        return df

    aggregate = df.groupby(GROUP_COLUMNS, dropna=False).agg(
        {
            **{column: ["mean", "std", "count"] for column in METRIC_COLUMNS},
            "collapsed": "mean",
        }
    )
    aggregate.columns = [
        "collapse_rate" if col[0] == "collapsed" else f"{col[0]}_{col[1]}"
        for col in aggregate.columns
    ]
    return aggregate.reset_index()


def condition_decision_summary(metric_rows: List[dict]) -> Dict[str, Dict[str, float]]:
    df = pd.DataFrame(metric_rows)
    if df.empty:
        return {}

    summary = {}
    for condition, condition_df in df.groupby("condition"):
        step_curve = (
            condition_df.groupby("attack_step")
            .agg({"TargetFit": "mean", "SPI": "mean"})
            .reset_index()
            .sort_values("attack_step")
        )
        steps = step_curve["attack_step"].to_numpy(dtype=float)
        targetfit = step_curve["TargetFit"].to_numpy(dtype=float)
        spi = step_curve["SPI"].to_numpy(dtype=float)
        if len(steps) > 1:
            auc_targetfit = float(np.trapz(targetfit, steps))
            auc_spi = float(np.trapz(spi, steps))
        else:
            auc_targetfit = float(targetfit[0])
            auc_spi = float(spi[0])

        tf_floor = float(np.nanmin(targetfit))
        half_life = compute_targetfit_half_life(
            list(zip(step_curve["attack_step"], step_curve["TargetFit"])),
            tf_floor=tf_floor,
        )
        summary[condition] = {
            "AUC_TargetFit": auc_targetfit,
            "AUC_SPI": auc_spi,
            "TargetFit_half_life": float(
                half_life["half_life_step"]
                if half_life["half_life_step"] is not None
                else np.inf
            ),
            "sigma_rel_error": float(condition_df["sigma_rel_error"].mean()),
            "entropy_ratio": float(condition_df["entropy_ratio"].mean()),
            "collapse_rate": float(condition_df["collapsed"].mean()),
        }
    return summary


def aggregate_results(input_roots: Iterable[Path], output_dir: Path) -> Dict[str, Path]:
    output_files = discover_eval_outputs(input_roots)
    if not output_files:
        raise FileNotFoundError("No eval_outputs.jsonl files found.")

    metric_rows = []
    rejected_rows = []
    for file_path in output_files:
        for row in parse_jsonl(file_path):
            try:
                metric_rows.append(record_to_metric_row(row))
            except Exception as exc:
                rejected_rows.append({
                    "file": str(file_path),
                    "prompt_id": row.get("prompt_id", ""),
                    "error": repr(exc),
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    per_prompt_path = output_dir / "per_prompt_metrics.csv"
    aggregate_path = output_dir / "aggregate_metrics.csv"
    summary_path = output_dir / "condition_decision_summary.json"
    rejected_path = output_dir / "rejected_metric_rows.jsonl"

    pd.DataFrame(metric_rows).to_csv(per_prompt_path, index=False)
    aggregate_metric_rows(metric_rows).to_csv(aggregate_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(condition_decision_summary(metric_rows), f, indent=2, sort_keys=True)

    if rejected_rows:
        with open(rejected_path, "w") as f:
            for row in rejected_rows:
                f.write(json.dumps(row) + "\n")

    return {
        "per_prompt": per_prompt_path,
        "aggregate": aggregate_path,
        "condition_summary": summary_path,
        "rejected": rejected_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_roots",
        nargs="*",
        default=["results/vllm", "results/vllm_smoke", "results/iteration2"],
    )
    parser.add_argument("--output_dir", default="results/iteration2/local/metrics")
    args = parser.parse_args()

    paths = aggregate_results([Path(root) for root in args.input_roots], Path(args.output_dir))
    for label, path in paths.items():
        if path.exists():
            print(f"{label}: {path}")


if __name__ == "__main__":
    main()
