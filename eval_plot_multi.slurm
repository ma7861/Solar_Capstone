#!/bin/bash
#SBATCH --job-name=evaluate_unet          # Job name
#SBATCH --output=evaluate_unet_171_193_304_%j.out     # Output log (%j will be replaced with job ID)
#SBATCH --error=evaluate_unet_171_193_304_%j.err      # Error log
#SBATCH --partition=gpu                   # GPU partition
#SBATCH --gres=gpu:1                      # Number of GPUs
#SBATCH --ntasks=1                        # Single task
#SBATCH --cpus-per-task=4                 # Number of CPU cores
#SBATCH --mem=900G                         # Memory per node
#SBATCH --time=08:00:00                   # Max time for evaluation (adjust as needed)
#SBATCH --mail-type=END,FAIL              # Notifications for job completion and failure
#SBATCH --mail-user=hz2291@nyu.edu # Email for notifications

# Load modules
module load cuda/12.3.2
module load cudnn/8.9.7.29-12
echo "Modules loaded successfully."

cd /mnt/home/hzhu2
# Activate virtual environment
source ~/solar/bin/activate
echo "Virtual environment activated."

pip install sunpy[visualization] mpl-animators --quiet

# Run evaluation script
srun -u python -u eval_plot_multi.py \
    --batch_size 16 \
    --input_channels 171 193 304 \
    --output_channel 335 \
    --data_dir "/mnt/ceph/users/manand" \
    --model_dir "/mnt/home/hzhu2/saved_models/171_193_304" \
    --concatenate_inputs=True \
    --subset_step 100