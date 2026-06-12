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
#SBATCH --time=2:00:00
#SBATCH --array=0-143
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/LLM_toy_immunization/logs/attack_%A_%a.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/LLM_toy_immunization/logs/attack_%A_%a.err

# =============================================================================
# Real-LLM Immunization Attack Pipeline
# =============================================================================

set -eo pipefail

CONDITIONS=("C1" "C2" "C3" "C4")
SEEDS=(42 43 44)
ATTACK_TYPES=("sampled" "distributional")
ATTACK_LRS=("1e-05" "3e-05")
DIFFICULTY_LEVELS=(1 2 3)

N_SEEDS=${#SEEDS[@]}
N_ATTACK_TYPES=${#ATTACK_TYPES[@]}
N_ATTACK_LRS=${#ATTACK_LRS[@]}
N_DIFFICULTY=${#DIFFICULTY_LEVELS[@]}

TASK_ID=$SLURM_ARRAY_TASK_ID
DIFFICULTY_IDX=$((TASK_ID % N_DIFFICULTY))
TASK_ID=$((TASK_ID / N_DIFFICULTY))
LR_IDX=$((TASK_ID % N_ATTACK_LRS))
TASK_ID=$((TASK_ID / N_ATTACK_LRS))
ATTACK_TYPE_IDX=$((TASK_ID % N_ATTACK_TYPES))
TASK_ID=$((TASK_ID / N_ATTACK_TYPES))
SEED_IDX=$((TASK_ID % N_SEEDS))
COND_IDX=$((TASK_ID / N_SEEDS))

CONDITION=${CONDITIONS[$COND_IDX]}
SEED=${SEEDS[$SEED_IDX]}
ATTACK_TYPE=${ATTACK_TYPES[$ATTACK_TYPE_IDX]}
ATTACK_LR=${ATTACK_LRS[$LR_IDX]}
DIFFICULTY_LEVEL=${DIFFICULTY_LEVELS[$DIFFICULTY_IDX]}

echo "=========================================="
echo "Immunization Attack Pipeline"
echo "Condition: $CONDITION | Seed: $SEED | Attack: $ATTACK_TYPE | LR: $ATTACK_LR | Difficulty: $DIFFICULTY_LEVEL"
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
OUTPUT_DIR="results/attack/${CONDITION}/seed_${SEED}/${ATTACK_TYPE}/lr_${ATTACK_LR}/difficulty_${DIFFICULTY_LEVEL}"

echo "Running Uniform attack fine-tuning on $CONDITION..."
echo "Adapter: $ADAPTER_PATH -> Output: $OUTPUT_DIR"

# Attack script checkpoints are hardcoded as [10, 50, 100, 250, 500, 1000] in run_attack.py.
# The max is 1000, so we limit to max_steps=1000 or let the script default to it.
python src/attack/run_attack.py \
    --model_name_or_path "$MODEL_PATH" \
    --adapter_path "$ADAPTER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --condition "$CONDITION" \
    --seed "$SEED" \
    --attack_type "$ATTACK_TYPE" \
    --batch_size 4 \
    --learning_rate "$ATTACK_LR" \
    --difficulty_level "$DIFFICULTY_LEVEL" \
    --context_len 128 \
    --max_steps "${MAX_ATTACK_STEPS:-5000}"

echo "✅ Attack completed at $(date)"
