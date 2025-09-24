#!/bin/bash
#SBATCH --job-name=PG19_dataset_creation
#SBATCH --output=logs/dataset_creation_%j.out
#SBATCH --error=logs/dataset_creation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=standard

# Load required modules
module load python/3.9
module load gcc

# Activate virtual environment
source /venv/bin/activate

# Set working directory
cd /data/pss-watermark-detection

# Create necessary directories
mkdir -p logs
mkdir -p data

# Export Kaggle credentials if needed
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Run dataset creation
echo "========================================="
echo "Starting PG-19 Dataset Creation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "========================================="

python scripts/run_dataset_creation.py \
    --config configs/experiment_config.yaml \
    --output-dir data \
    --num-samples 1000 \
    --download \
    --create-variations

echo "========================================="
echo "Dataset creation completed"
echo "End time: $(date)"
echo "========================================="

# Deactivate virtual environment
deactivate
