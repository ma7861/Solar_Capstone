#!/bin/bash
#SBATCH --job-name=train_multi_unet           # Job name
#SBATCH --output=multi_unet_%j.out            # Standard output
#SBATCH --error=multi_unet_%j.err             # Standard error
#SBATCH --partition=gpu                       # GPU partition
#SBATCH --gres=gpu:1                          # Number of GPUs
#SBATCH --ntasks=1                            # Single task
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --mem=900G                            # Memory per node
#SBATCH --time=24:00:00                       # Maximum runtime (24 hours)
#SBATCH --mail-type=END,FAIL                  # Notifications for job completion and failure
#SBATCH --mail-user=hz2291@nyu.edu            # Email for notifications

# Training parameters
lr=0.001
batch_size=64
epochs=10
input_channels="171 193 304"                  # Channels for input
output_channel="335"                          # Channel for output
data_dir="/mnt/ceph/users/manand"             # Base directory containing dataset
save_model_dir="/mnt/home/hzhu2/saved_models" # Directory to save trained models

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
    --lr ${lr} \
    --batch_size ${batch_size} \
    --epochs ${epochs} \
    --input_channels ${input_channels} \
    --output_channel ${output_channel} \
    --data_dir ${data_dir} \
    --save_model_dir ${save_model_dir}
