#!/bin/bash
#SBATCH --job-name=Watermark_generation
#SBATCH --output=logs/watermark_%j.out
#SBATCH --error=logs/watermark_%j.err
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
module load gcc

# Activate virtual environment
source /venv/bin/activate

# Set working directory
cd /data/pss-watermark-detection

# Create logs directory
mkdir -p logs

# Set environment variables
export HF_TOKEN="your_huggingface_token"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HOME=/scratch/${USER}/hf_cache
export TRANSFORMERS_CACHE=/scratch/${USER}/hf_cache
export HF_DATASETS_CACHE=/scratch/${USER}/hf_cache

# CUDA settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create cache directory
mkdir -p $HF_HOME

# Print system information
echo "========================================="
echo "Starting Watermark Generation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "========================================="

# Run watermark generation
python scripts/run_watermarking.py \
    --input-csv data/ai_input_text.csv \
    --output-text-csv data/ai_watermarked_text.csv \
    --output-bits-csv data/ai_watermark_bits.csv \
    --exp-config configs/experiment_config.yaml \
    --model-config configs/model_config.yaml

echo "========================================="
echo "Watermark generation completed"
echo "End time: $(date)"
echo "========================================="

# Deactivate virtual environment
deactivate
