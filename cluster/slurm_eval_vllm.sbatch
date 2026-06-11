#!/bin/bash
#SBATCH --job-name=eval_immunization
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --time=1:00:00
#SBATCH --array=0-11
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/LLM_toy_immunization/logs/eval_%A_%a.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/LLM_toy_immunization/logs/eval_%A_%a.err

# =============================================================================
# Real-LLM Immunization vLLM Evaluation Pipeline
# =============================================================================

set -eo pipefail

CONDITIONS=("C1" "C2" "C3" "C4")
SEEDS=(42 43 44)

COND_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))

CONDITION=${CONDITIONS[$COND_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=========================================="
echo "Immunization Evaluation Pipeline"
echo "Condition: $CONDITION | Seed: $SEED"
echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Array ID: $SLURM_ARRAY_TASK_ID"
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
SERVED_MODEL_NAME=$(basename "$MODEL_PATH")
VLLM_PORT=$(( 8000 + SLURM_ARRAY_TASK_ID )) # Unique port per array task
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.92

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
mkdir -p logs results/vllm

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
# 4. Validate prerequisites
# ------------------------------------------------------------------
if [ -d "$MODEL_PATH" ]; then
    echo "✅ Model found at: $MODEL_PATH"
else
    echo "⚠️ Warning: Model not found at $MODEL_PATH. Will try to use HuggingFace hub name (meta-llama/Llama-3.2-1B) which may fail offline."
    MODEL_PATH="meta-llama/Llama-3.2-1B"
fi

# ------------------------------------------------------------------
# 5. Run Evaluation Loop (Offline inference directly with vLLM LLM class)
# ------------------------------------------------------------------
# Since eval_vllm.py uses the offline LLM engine rather than querying an API server, 
# we don't need to start vLLM API server in bash. We can just run the script.

echo "Running vLLM Evaluations..."

for ATTACK_STEP in 0 10 50 100 250 500; do
    ADAPTER_PATH="results/attack/${CONDITION}/seed_${SEED}/attack_step_${ATTACK_STEP}"
    
    echo "Evaluating Attack Step: $ATTACK_STEP"
    
    # We ignore ADAPTER_PATH if it doesn't exist, though typically in offline we just pass it
    # ensure your eval script handles missing adapter nicely or creates mock.
    
    python scripts/eval_vllm.py \
        --model_path "$MODEL_PATH" \
        --adapter_path "$ADAPTER_PATH" \
        --prompts_file "results/data/eval_prompts.jsonl" \
        --output_dir "results/vllm" \
        --run_id "experiment_1" \
        --condition "$CONDITION" \
        --seed "$SEED" \
        --attack_step "$ATTACK_STEP"
done

echo "✅ Job completed at $(date)"
