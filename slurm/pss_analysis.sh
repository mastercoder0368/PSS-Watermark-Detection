#!/bin/bash
#SBATCH --job-name=PSS_analysis
#SBATCH --output=logs/pss_analysis_%j.out
#SBATCH --error=logs/pss_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:T4:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# Load required modules
module load python/3.9
module load cuda/11.8
module load cudnn/8.6

# Activate virtual environment
source /venv/bin/activate

# Set working directory
cd /data/pss-watermark-detection

# Create necessary directories
mkdir -p logs
mkdir -p results

# Set environment variables
export HF_TOKEN="your_huggingface_token"
export CUDA_VISIBLE_DEVICES=0

# Print system information
echo "========================================="
echo "Starting PSS Analysis Pipeline"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "========================================="

# Step 1: Run detection on all files
echo "Step 1: Running watermark detection..."
python scripts/run_detection.py \
    --input-dir data/paraphrased \
    --output-dir results/detection/ai \
    --text-type ai \
    --split-first \
    --exp-config configs/experiment_config.yaml \
    --model-config configs/model_config.yaml

python scripts/run_detection.py \
    --input-dir data/paraphrased \
    --output-dir results/detection/human \
    --text-type human \
    --split-first \
    --exp-config configs/experiment_config.yaml \
    --model-config configs/model_config.yaml

# Step 2: Run PSS analysis
echo "Step 2: Running PSS analysis..."
python scripts/run_pss_analysis.py \
    --ai-dir results/detection/ai \
    --human-dir results/detection/human \
    --output-dir results \
    --window-size 50 \
    --stride 10 \
    --exp-config configs/experiment_config.yaml

echo "========================================="
echo "PSS Analysis completed"
echo "End time: $(date)"
echo "========================================="

# Print results summary
echo "Results summary:"
tail -20 results/pss_results/pss_results.csv

# Deactivate virtual environment
deactivate
