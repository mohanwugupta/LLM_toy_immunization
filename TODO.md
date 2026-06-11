# TODO

## Now

- [ ] Run one-seed Iteration 2 smoke test with C1-C4, distributional attack, difficulty level 3, context lengths 64 and 128.
- [ ] Generate smoke `results/iteration2/{run_id}/report_iteration2.md`.
- [ ] Confirm C2 attack validity from smoke outputs.
- [ ] Confirm C1 shortcut diagnostic from smoke outputs.

## Next

- [ ] Run three-seed Iteration 2 benchmark.
- [ ] Generate Iteration 2 plots for TargetFit, sigma error, entropy ratio, SPI, and collapse rate.
- [ ] Run benign Normal-shift adaptation control.
- [ ] Fill representational-analysis sections after hidden-state extraction.

## Later

- [ ] Add Level 4 moment-matched adversarial family to the cluster grid after Levels 1-3 run cleanly.
- [ ] Add gradient-alignment analysis.
- [ ] Add sparse/JOLA-style intervention variant.
- [ ] Add bootstrap confidence intervals to the report tables.

## Done

- [x] Create directory structure.
- [x] Create pyproject.toml.
- [x] Create README.md.
- [x] Create SCRATCHPAD.md.
- [x] Milestone 1: Data and metrics.
- [x] Milestone 2: Base-model diagnostic.
- [x] Milestone 3: C1-C4 training.
- [x] Milestone 4: Uniform attack.
- [x] Milestone 5: vLLM cluster inference.
- [x] Add anti-collapse metric tests (`tests/test_anticollapse_metrics.py`).
- [x] Implement TargetFit / JS metrics (`src/eval/anti_collapse_metrics.py`).
- [x] Implement sigma, entropy, and collapse metrics (`src/eval/anti_collapse_metrics.py`).
- [x] Add decision rule tests (`tests/test_decision_rules.py`).
- [x] Implement decision rules (`src/eval/decision_rules.py`).
- [x] Add distributional attack tests (`tests/test_distributional_attack.py`).
- [x] Implement sampled, distributional, and mixed attack losses (`src/attack/losses.py`, `src/attack/run_attack.py`).
- [x] Add difficulty ladder tests (`tests/test_difficulty_ladder.py`).
- [x] Implement difficulty ladder (`src/data/difficulty_ladder.py`).
- [x] Add vLLM schema tests (`tests/test_vllm_eval_schema.py`).
- [x] Update vLLM evaluator schema and metadata (`scripts/eval_vllm.py`).
- [x] Add Iteration 2 aggregation and report helpers (`src/analysis/aggregate_results.py`, `src/analysis/report_iteration2.py`).
- [x] Add Makefile targets for Iteration 2 tests, attacks, metrics, and reports.
