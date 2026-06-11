.PHONY: test test-anticollapse test-attacks test-decision data-difficulty-ladder attack-iteration2 eval-vllm-iteration2 metrics-iteration2 report-iteration2 install clean

test:
	pytest tests/ -v

test-anticollapse:
	pytest tests/test_anticollapse_metrics.py tests/test_vllm_eval_schema.py -v

test-attacks:
	pytest tests/test_distributional_attack.py -v

test-decision:
	pytest tests/test_decision_rules.py tests/test_difficulty_ladder.py -v

data-difficulty-ladder:
	python scripts/make_dataset.py --iteration2 --output_dir results/data --seeds "$${SEEDS:-10}"

attack-iteration2:
	python src/attack/run_attack.py \
		--adapter_path "$${ADAPTER_PATH}" \
		--condition "$${CONDITION:-C4}" \
		--seed "$${SEED:-42}" \
		--attack_type "$${ATTACK_TYPE:-distributional}" \
		--learning_rate "$${ATTACK_LR:-3e-5}" \
		--difficulty_level "$${DIFFICULTY_LEVEL:-3}"

eval-vllm-iteration2:
	python scripts/eval_vllm.py \
		--model_path "$${MODEL_PATH}" \
		--adapter_path "$${ADAPTER_PATH:-}" \
		--prompts_file "$${PROMPTS_FILE}" \
		--run_id "$${RUN_ID}" \
		--condition "$${CONDITION:-C4}" \
		--seed "$${SEED:-42}" \
		--attack_type "$${ATTACK_TYPE:-distributional}" \
		--attack_lr "$${ATTACK_LR:-3e-5}" \
		--attack_step "$${ATTACK_STEP:-0}" \
		--difficulty_level "$${DIFFICULTY_LEVEL:-3}" \
		--context_len "$${CONTEXT_LEN:-128}"

metrics-iteration2:
	python -m src.analysis.aggregate_results \
		--output_dir "results/iteration2/$${RUN_ID:-local}/metrics"

report-iteration2:
	python -m src.analysis.report_iteration2 \
		--metrics_dir "results/iteration2/$${RUN_ID:-local}/metrics" \
		--output_dir "results/iteration2/$${RUN_ID:-local}"

install:
	pip install -e .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
