#!/bin/bash
#SBATCH --job-name=smoke_immunization
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/LLM_toy_immunization/logs/smoke_%j.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/LLM_toy_immunization/logs/smoke_%j.err

# =============================================================================
# Real-LLM Immunization Smoke Test (Real Model)
# =============================================================================

set -eo pipefail

echo "=========================================="
echo "Real-LLM Immunization Smoke Test"
echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Time:     $(date)"
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
echo ""

# ------------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------------
PROJECT_DIR=/scratch/gpfs/JORDANAT/mg9965/LLM_toy_immunization
CONDA_ENV=PromptControlText
MODEL_PATH=/scratch/gpfs/JORDANAT/mg9965/models/meta-llama--Llama-3.2-1B

# ------------------------------------------------------------------
# 1. Environment setup
# ------------------------------------------------------------------
cd "$PROJECT_DIR"

module load anaconda3/2025.6

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif [ -f "$HOME/.conda/envs/$CONDA_ENV/bin/activate" ]; then
    source "$HOME/.conda/envs/$CONDA_ENV/bin/activate"
else
    source activate "$CONDA_ENV"
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------------------------------------------------
# 2. Cache & offline
# ------------------------------------------------------------------
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export HF_DATASETS_CACHE=/scratch/gpfs/JORDANAT/mg9965/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export HF_DATASETS_DISK_DIR=$HF_DATASETS_CACHE
export VLLM_CACHE_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache
export VLLM_USAGE_STATS_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/usage_stats
export TRITON_CACHE_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/triton
export XDG_CACHE_HOME=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/xdg
export TIKTOKEN_CACHE_DIR=$HOME/.tiktoken_cache

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p "$VLLM_CACHE_DIR" "$VLLM_USAGE_STATS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ------------------------------------------------------------------
# 3. GPU / Memory optimization
# ------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ------------------------------------------------------------------
# 4. Run Smoke Test
# ------------------------------------------------------------------
mkdir -p logs results/adapters_smoke results/attack_smoke results/data results/vllm_smoke

if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️ Warning: Model not found at $MODEL_PATH. Will try to use HuggingFace hub name (meta-llama/Llama-3.2-1B) which may fail offline."
    MODEL_PATH="meta-llama/Llama-3.2-1B"
    export HF_HUB_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
fi

echo "--- Running Training Smoke Test (C3) ---"
python src/train/train_condition.py \
    --model_name_or_path "$MODEL_PATH" \
    --condition "C3" \
    --output_dir "results/adapters_smoke" \
    --batch_size 2 \
    --max_steps 5

echo "--- Running Training Smoke Test (C4) ---"
python src/train/train_c4.py \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "results/adapters_smoke" \
    --batch_size 2 \
    --max_steps 5

echo "--- Running Attack Smoke Test (C3) ---"
python src/attack/run_attack.py \
    --model_name_or_path "$MODEL_PATH" \
    --adapter_path "results/adapters_smoke/C3" \
    --output_dir "results/attack_smoke/C3" \
    --batch_size 2 \
    --max_steps 5

echo "--- Running Eval Smoke Test (vLLM) ---"
# Create a tiny dummy jsonl file for evaluation
cat << EOF > results/data/smoke_eval_prompts.jsonl
{"example_id": "test_1", "mu": 500, "sigma": 100, "family": "normal", "matched_family": "uniform", "context_len": 16, "prompt": "500,510,490,"}
{"example_id": "test_2", "mu": 400, "sigma": 80, "family": "uniform", "matched_family": "normal", "context_len": 16, "prompt": "380,420,400,"}
EOF

python scripts/eval_vllm.py \
    --model_path "$MODEL_PATH" \
    --adapter_path "results/adapters_smoke/C4" \
    --prompts_file "results/data/smoke_eval_prompts.jsonl" \
    --output_dir "results/vllm_smoke" \
    --run_id "smoke_test_run" \
    --condition "C4" \
    --seed 42 \
    --attack_step 0

echo "✅ Smoke test completed at $(date)"
