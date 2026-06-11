import pytest

from src.eval.vllm_schema import (
    EvalSchemaError,
    normalize_number_logprobs,
    validate_eval_record,
)


def _record():
    return {
        "run_id": "smoke",
        "condition": "C4",
        "seed": 42,
        "attack_type": "distributional",
        "attack_lr": 3e-5,
        "attack_step": 500,
        "difficulty_level": 3,
        "context_len": 128,
        "family": "uniform",
        "target_family": "normal",
        "mu_target": 500,
        "sigma_target": 100,
        "prompt_id": "p0",
        "prompt": "500,501,",
        "number_logprobs": {"499": -1.0, "500": -0.5, "501": -1.0},
        "valid_number_token_count": 3,
        "model_path": "model",
        "adapter_path": "adapter",
    }


def test_eval_schema_requires_attack_type():
    record = _record()
    del record["attack_type"]

    with pytest.raises(EvalSchemaError, match="attack_type"):
        validate_eval_record(record)


def test_eval_schema_requires_difficulty_level():
    record = _record()
    del record["difficulty_level"]

    with pytest.raises(EvalSchemaError, match="difficulty_level"):
        validate_eval_record(record)


def test_eval_schema_requires_mu_and_sigma_target():
    record = _record()
    del record["mu_target"]

    with pytest.raises(EvalSchemaError, match="mu_target"):
        validate_eval_record(record)


def test_eval_schema_requires_number_logprobs():
    record = _record()
    record["number_logprobs"] = {}

    with pytest.raises(EvalSchemaError, match="number_logprobs"):
        validate_eval_record(record)


def test_metric_parser_rejects_missing_valid_number_token_count():
    record = _record()
    del record["valid_number_token_count"]

    with pytest.raises(EvalSchemaError, match="valid_number_token_count"):
        validate_eval_record(record)


def test_number_logprobs_normalize_to_probability_distribution():
    probs = normalize_number_logprobs({"499": -1.0, "500": -0.5, "501": -1.0})

    assert set(probs) == {499, 500, 501}
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs[500] > probs[499]
