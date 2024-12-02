#!/bin/bash
#SBATCH --job-name=train_multi_unet           # Job name
#SBATCH --output=multi_unet_171_304_335_%j.out            # Standard output
#SBATCH --error=multi_unet_171_304_335_%j.err             # Standard error
#SBATCH --partition=gpu                       # GPU partition
#SBATCH --gres=gpu:1                          # Number of GPUs
#SBATCH --ntasks=1                            # Single task
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --mem=900G                            # Memory per node
#SBATCH --time=24:00:00                       # Maximum runtime (50 hours)
#SBATCH --mail-type=END,FAIL                  # Notifications for job completion and failure
#SBATCH --mail-user=hz2291@nyu.edu            # Email for notifications

# Load required modules
module load cuda/12.3.2                       # Load CUDA module
module load cudnn/8.9.7.29-12                 # Load cuDNN module
echo "Modules loaded successfully."

# Navigate to working directory and activate virtual environment
cd /mnt/home/hzhu2/
source ~/solar/bin/activate
echo "Virtual environment activated."

# Run the training script
srun -u python -u train_unet_multi.py \
    --lr 0.0001 \
    --batch_size 16 \
    --epochs 5 \
    --input_channels 171 304 \
    --output_channel 335 \
    --data_dir "/mnt/ceph/users/manand" \
    --save_model_dir "/mnt/home/hzhu2/saved_models/171_304" \
    --concatenate_inputs \
    --subset_step 10
