# Real-LLM Numerical Representational Immunization Benchmark Implementation Plan

This document outlines the plan for implementing the Real-LLM representational immunization benchmark (C4 vs C3) based on the PRD. The goal is to port the toy Normal-vs-Uniform representational immunization experiment to a real pretrained LLM using PEFT/LoRA and evaluate the resilience of safe outputs under a Uniform-only fine-tuning attack.

## Goal Description

We will set up a repository that runs representational immunization experiments (C1-C4 conditions) on an open-weight LLM (e.g., Llama-3.2-1B). The task involves modeling a target output distribution based on a prefix sequence of integers generated from Normal or Uniform distributions. We will introduce a representation anchoring loss (C4) and test if it improves the safe output's durability under later Uniform-only fine-tuning pressure compared to purely behavioral immunization (C3). Finally, we'll evaluate everything efficiently using vLLM on the cluster.

> [!IMPORTANT]
> **User Review Required**:
> 1. Do you want to restrict testing *only* to single-token numbers initially to avoid tokenization complications (e.g., filtering out sequences that Llama-3.2 tokenizes into multiple tokens)?
> 2. Are you using a SLURM cluster for these jobs? If so, does the `run_llama31_8b.sh` script mirror your current SLURM environment precisely, or will we need to adapt paths/partitions?
> 3. Does the PEFT framework (LoRA) suffice for the MVP as specified, or do you have specific preferences for QLoRA and quantization libraries?

## Open Questions

> [!WARNING]
> 1. For data generation, the PRD mentions using "Normal contexts → Normal targets" and "Uniform contexts → Uniform targets". Will we be using the previous toy models' distribution sampling code as a base, or generating new numerical tokens on the fly during training?
> 2. What layers/positions should we extract hidden states from for C4? The PRD suggests "Layer selected by probe performance" and "final context token", which implies we need to run probing *before* training C4. Is that correct?

## Proposed Changes

We will construct the repository following the structure required by the PRD.

---

### Foundation & Infrastructure

#### [NEW] [README.md](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/README.md)
Initial README instructions.

#### [NEW] [SCRATCHPAD.md](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/SCRATCHPAD.md)
Running scientific audit log tracking decisions and test failures.

#### [NEW] [TODO.md](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/TODO.md)
Living task list for TDD.

#### [NEW] [pyproject.toml](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/pyproject.toml)
Python project configuration and dependencies (`pytest`, `peft`, `vllm`, `transformers`, etc).

---

### Data Generation & Tokenization

#### [NEW] [src/data/generate_numeric.py](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/src/data/generate_numeric.py)
Normal/Uniform distribution sampling, matched pairs logic, clamping to [0, 999], and string rendering.

#### [NEW] [src/data/tokenize_numbers.py](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/src/data/tokenize_numbers.py)
Logic to verify which numbers in [0, 999] are single tokens in the target LLM tokenizer, and filter datasets accordingly.

#### [NEW] [tests/test_data_generation.py](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/tests/test_data_generation.py)
Pytest file verifying valid values, matched mean/variance, and string formatting.

#### [NEW] [tests/test_tokenization.py](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/tests/test_tokenization.py)
Pytest file ensuring single-token extraction works correctly.

---

### Training Scripts

#### [NEW] [src/models/lora.py](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/src/models/lora.py)
PEFT wrapping logic (target_modules `[q_proj, k_proj, v_proj, o_proj]`, r=8, alpha=16).

#### [NEW] [src/train/losses.py](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/src/train/losses.py)
Standard behavioral loss (C1-C3) and representation loss with `stop_gradient` anchor (C4).

#### [NEW] [src/train/train_condition.py](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/src/train/train_condition.py)
Main script for training C1, C2, and C3.

#### [NEW] [src/train/train_c4.py](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/src/train/train_c4.py)
Main script for training C4 including caching logic and dual-forward passes.

#### [NEW] [tests/test_training_losses.py](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/tests/test_training_losses.py)
Verification of correct target mappings and gradients for C3/C4.

---

### Attack & Evaluation Pipeline

#### [NEW] [src/attack/run_attack.py](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/src/attack/run_attack.py)
Fine-tunes immunized models purely on Uniform targets to evaluate breakdown.

#### [NEW] [src/eval/compute_spi.py](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/src/eval/compute_spi.py)
KL-divergence calculations for the Safe Prior Index.

#### [NEW] [scripts/eval_vllm.py](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/scripts/eval_vllm.py)
vLLM batched inference script parsing JSONL requests and saving top logprobs for numbers. Integrates with the existing `vllm_client.py` where possible or adapts the custom JSONL scheme.

---

### Cluster & Slurm Configuration

#### [NEW] [cluster/slurm_train.sbatch](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/cluster/slurm_train.sbatch)
Job array script for C1-C4 training across seeds.

#### [NEW] [cluster/slurm_eval_vllm.sbatch](file:///Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/LLM/LLM_toy_immunization/cluster/slurm_eval_vllm.sbatch)
Job array script for scalable vLLM evaluation at specific attack checkpoints. Adapts logic from `run_llama31_8b.sh`.

---

## Verification Plan

We will follow strict TDD across the 8 phases described in the PRD.

### Automated Tests
1. **Phase 1-7 Tests**: Run all unit tests locally with Pytest (`make test`).
2. **Phase 8 End-to-End Tiny**: Run `tests/test_end_to_end_tiny.py` locally on a mocked or tiny model (e.g., tiny Llama randomly initialized) to verify the data pipeline, training losses, adapter saving, attack step, and metric logging.

### Manual Verification
1. **Base Model Diagnostic**: Run the model and check if distribution inference passes the `> 0.70` R² criteria.
2. **Cluster Verification**: Trigger one slurm job from `slurm_train.sbatch` and `slurm_eval_vllm.sbatch` and manually verify that logs correctly save to the `results/` hierarchy and parse without errors.
