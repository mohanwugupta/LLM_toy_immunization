#!/bin/bash
# Usage: ./scripts/download_model.sh <YOUR_HF_TOKEN>

if [ -z "$1" ]; then
    echo "Please provide your Hugging Face token as an argument."
    echo "Usage: ./scripts/download_model.sh <YOUR_HF_TOKEN>"
    echo "You can get your token from https://huggingface.co/settings/tokens"
    exit 1
fi

HF_TOKEN=$1
# Matches the SLURM script MODEL_PATH
MODEL_DIR="/scratch/gpfs/JORDANAT/mg9965/models/meta-llama--Llama-3.2-1B"

echo "Downloading meta-llama/Llama-3.2-1B to $MODEL_DIR..."

# Ensure the huggingface_hub package is installed
pip install -U "huggingface_hub[cli]"

# Use the huggingface-cli to download the model
huggingface-cli download meta-llama/Llama-3.2-1B \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False \
    --token "$HF_TOKEN"

echo "Download complete! Model saved to $MODEL_DIR"
