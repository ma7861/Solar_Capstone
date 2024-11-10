#!/bin/bash
#SBATCH --job-name=evaluate_unet          # Job name
#SBATCH --output=evaluate_unet_%j.out     # Output log (%j will be replaced with job ID)
#SBATCH --error=evaluate_unet_%j.err      # Error log
#SBATCH --partition=gpu                   # GPU partition
#SBATCH --gres=gpu:1                      # Number of GPUs
#SBATCH --ntasks=1                        # Single task
#SBATCH --cpus-per-task=4                 # Number of CPU cores
#SBATCH --mem=900G                         # Memory per node
#SBATCH --time=04:00:00                   # Max time for evaluation (adjust as needed)
#SBATCH --mail-type=END,FAIL              # Notifications for job completion and failure
#SBATCH --mail-user=hz2291@nyu.edu # Email for notifications

# Load modules
module load cuda/12.3.2
module load cudnn/8.9.7.29-12

cd /mnt/home/hzhu2
# Activate virtual environment
source ~/solar/bin/activate

# Run evaluation script
srun -u python -u eval_plot.py