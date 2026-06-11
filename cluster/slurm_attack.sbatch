#!/bin/bash
#SBATCH --job-name=attack_immunization
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
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/LLM_toy_immunization/logs/attack_%A_%a.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/LLM_toy_immunization/logs/attack_%A_%a.err

# =============================================================================
# Real-LLM Immunization Attack Pipeline
# =============================================================================

set -eo pipefail

CONDITIONS=("C1" "C2" "C3" "C4")
SEEDS=(42 43 44)

COND_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))

CONDITION=${CONDITIONS[$COND_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=========================================="
echo "Immunization Attack Pipeline"
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
# 2. Run Attack
# ------------------------------------------------------------------
ADAPTER_PATH="results/adapters/${CONDITION}"
OUTPUT_DIR="results/attack/${CONDITION}/seed_${SEED}"

echo "Running Uniform attack fine-tuning on $CONDITION..."
echo "Adapter: $ADAPTER_PATH -> Output: $OUTPUT_DIR"

# Attack script checkpoints are hardcoded as [10, 50, 100, 250, 500, 1000] in run_attack.py.
# The max is 1000, so we limit to max_steps=1000 or let the script default to it.
python src/attack/run_attack.py \
    --model_name_or_path "$MODEL_PATH" \
    --adapter_path "$ADAPTER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --learning_rate 3e-5 \
    --context_len 128 \
    --max_steps 500

echo "✅ Attack completed at $(date)"
