#!/bin/bash
#SBATCH --job-name=Human_paraphrase
#SBATCH --output=logs/human_paraphrase_%j.out
#SBATCH --error=logs/human_paraphrase_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Load required modules
module load python/3.9
module load cuda/11.8
module load cudnn/8.6
module load cmake

# Activate virtual environment
source /venv/bin/activate

# Set working directory
cd /data/pss-watermark-detection

# Create necessary directories
mkdir -p logs
mkdir -p data/paraphrased

# Set model directory
MODEL_DIR="/scratch/${USER}/models"
mkdir -p $MODEL_DIR

# Check if llama-cpp-python has CUDA support and install if needed
echo "Checking llama-cpp-python CUDA support..."
CUDA_SUPPORT_CHECK=$(python -c "
try:
    from llama_cpp import Llama
    print('CUDA_AVAILABLE')
except ImportError:
    print('NOT_INSTALLED')
" 2>/dev/null)

if [ "$CUDA_SUPPORT_CHECK" != "CUDA_AVAILABLE" ]; then
    echo "Installing llama-cpp-python with CUDA support..."
    pip uninstall llama-cpp-python -y
    export CMAKE_ARGS="-DGGML_CUDA=on"
    export FORCE_CMAKE=1
    pip install --no-cache-dir --force-reinstall llama-cpp-python
fi

# Print system information
echo "========================================="
echo "Starting Human Text Paraphrasing"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "Iterations: 8"
echo "========================================="

# Check CUDA availability
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('CUDA not available')
"

# Find model file (should already be downloaded from AI paraphrasing)
MODEL_FILE=$(find "$MODEL_DIR" -name "*mistral-7b-instruct*.gguf" | grep -i "q4_k_m" | head -1)

if [ -z "$MODEL_FILE" ]; then
    echo "Model not found. Downloading..."
    python -c "
from huggingface_hub import hf_hub_download
import os

model_dir = '${MODEL_DIR}'
repo_id = 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF'
filename = 'mistral-7b-instruct-v0.2.Q4_K_M.gguf'

model_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    cache_dir=model_dir
)
print(f'Model downloaded to: {model_path}')
"
    MODEL_FILE=$(find "$MODEL_DIR" -name "*mistral-7b-instruct*.gguf" | grep -i "q4_k_m" | head -1)
fi

echo "Using model: $MODEL_FILE"

# Run paraphrasing for Human texts
python scripts/run_paraphrasing.py \
    --input-csv data/human_text_1500.csv \
    --output-csv data/paraphrased/human_paraphrased.csv \
    --text-type human \
    --model-path "$MODEL_FILE" \
    --iterations 8 \
    --batch-size 10 \
    --exp-config configs/experiment_config.yaml \
    --model-config configs/model_config.yaml

echo "========================================="
echo "Human paraphrasing completed"
echo "End time: $(date)"
echo "========================================="

# Deactivate virtual environment
deactivate
