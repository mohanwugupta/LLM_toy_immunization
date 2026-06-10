#!/bin/bash
#SBATCH --job-name=train_immunization
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/LLM_toy_immunization/logs/train_%A_%a.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/LLM_toy_immunization/logs/train_%A_%a.err

# =============================================================================
# Real-LLM Immunization Training Pipeline
# =============================================================================

set -eo pipefail

CONDITIONS=("C1" "C2" "C3" "C4")
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Immunization Training Pipeline"
echo "Condition: $CONDITION"
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

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p logs results/adapters

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
# 4. Run Training
# ------------------------------------------------------------------
echo "Running Training for $CONDITION..."

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️ Warning: Model not found at $MODEL_PATH. Will try to use HuggingFace hub name (meta-llama/Llama-3.2-1B) which may fail offline."
    MODEL_PATH="meta-llama/Llama-3.2-1B"
fi

if [ "$CONDITION" == "C4" ]; then
    python src/train/train_c4.py \
        --model_name_or_path "$MODEL_PATH" \
        --output_dir "results/adapters" \
        --epochs 3
else
    python src/train/train_condition.py \
        --model_name_or_path "$MODEL_PATH" \
        --condition "$CONDITION" \
        --output_dir "results/adapters" \
        --epochs 3
fi

echo "✅ Job completed at $(date)"
