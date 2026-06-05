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
# Real-LLM Immunization Smoke Test
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

# Not setting HF_HUB_OFFLINE=1 here so it can download the tiny model for the smoke test

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
mkdir -p logs

echo "Running end-to-end tiny smoke test..."
python -m pytest tests/test_end_to_end_tiny.py -v

echo "✅ Job completed at $(date)"
