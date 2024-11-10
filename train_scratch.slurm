#!/bin/bash
#SBATCH --job-name=train_unet_scratch          # Job name
#SBATCH --output=scratch_lr${lr}_bs${batch_size}_ep${epochs}_%j.out
#SBATCH --error=scratch_lr${lr}_bs${batch_size}_ep${epochs}_%j.err
#SBATCH --partition=gpu                        # GPU partition
#SBATCH --gres=gpu:1                           # Number of GPUs
#SBATCH --ntasks=1                             # Single task
#SBATCH --cpus-per-task=8                      # Number of CPU cores per task
#SBATCH --mem=900G                             # Memory per node
#SBATCH --time=24:00:00                        # Maximum time (24 hours)
#SBATCH --mail-type=END,FAIL                   # Notifications for job completion and failure
#SBATCH --mail-user=hz2291@nyu.edu      # Your email for notifications

lr=0.001
batch_size=32
epochs=10

module load cuda/12.3.2
module load cudnn/8.9.7.29-12
echo "Modules loaded successfully."

cd /mnt/home/hzhu2/
source ~/solar/bin/activate

echo "Virtual environment activated."

srun -u python -u train_unet.py --lr ${lr} --batch_size ${batch_size} --epochs ${epochs}
echo "Training script started."
