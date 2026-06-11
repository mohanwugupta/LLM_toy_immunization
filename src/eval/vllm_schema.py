import math
from typing import Any, Dict, Iterable, Mapping


class EvalSchemaError(ValueError):
    pass


REQUIRED_EVAL_FIELDS = {
    "run_id",
    "condition",
    "seed",
    "attack_type",
    "attack_lr",
    "attack_step",
    "difficulty_level",
    "context_len",
    "family",
    "target_family",
    "mu_target",
    "sigma_target",
    "prompt_id",
    "prompt",
    "number_logprobs",
    "valid_number_token_count",
    "model_path",
    "adapter_path",
}


def validate_eval_record(record: Mapping[str, Any]) -> bool:
    missing = sorted(REQUIRED_EVAL_FIELDS - set(record))
    if missing:
        raise EvalSchemaError(f"Missing required eval field(s): {', '.join(missing)}")

    if not isinstance(record["number_logprobs"], Mapping) or not record["number_logprobs"]:
        raise EvalSchemaError("number_logprobs must be a non-empty mapping")
    if int(record["valid_number_token_count"]) <= 0:
        raise EvalSchemaError("valid_number_token_count must be positive")
    if float(record["sigma_target"]) <= 0:
        raise EvalSchemaError("sigma_target must be positive")
    if record["attack_type"] not in {"sampled", "distributional", "mixed"}:
        raise EvalSchemaError(f"Unknown attack_type: {record['attack_type']}")
    return True


def normalize_number_logprobs(number_logprobs: Mapping[str, float]) -> Dict[int, float]:
    if not number_logprobs:
        raise EvalSchemaError("number_logprobs must be non-empty")

    parsed = {int(number): float(logprob) for number, logprob in number_logprobs.items()}
    max_logprob = max(parsed.values())
    probs = {
        number: math.exp(logprob - max_logprob)
        for number, logprob in parsed.items()
    }
    total = sum(probs.values())
    if total <= 0:
        raise EvalSchemaError("number_logprobs could not be normalized")
    return {number: prob / total for number, prob in probs.items()}


def validate_records(records: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    for record in records:
        validate_eval_record(record)
        count += 1
    return count
