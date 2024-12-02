import os
import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm

data_dir = "/mnt/ceph/users/manand"
output_channel = "335"
step_size = 20

# Load train, validation, and test datasets
train_path = os.path.join(data_dir, f"{output_channel}_train")
val_path = os.path.join(data_dir, f"{output_channel}_val")
test_path = os.path.join(data_dir, f"{output_channel}_test")

print("Loading datasets...")
train_dataset = load_from_disk(train_path).with_format("numpy")
val_dataset = load_from_disk(val_path).with_format("numpy")
test_dataset = load_from_disk(test_path).with_format("numpy")

def apply_step(dataset, step):
    indices = list(range(0, len(dataset), step))
    print(f"Selected {len(indices)} samples from {len(dataset)} using step size {step}.")
    return dataset.select(indices)

train_dataset = apply_step(train_dataset, step_size)

# Calculate the average pixel value in the training set
print("Calculating average pixel value for training set...")
total_sum, total_count = 0.0, 0
for idx in tqdm(range(len(train_dataset)), desc="Processing train dataset"):
    image = train_dataset[idx]['image']['array']
    total_sum += image.sum()
    total_count += image.size

train_avg = total_sum / total_count
print(f"Average pixel value (train): {train_avg:.4f}")

# Function to calculate MAE
def calculate_mae(dataset, average_value, desc="Evaluating"):
    total_mae = 0.0
    total_pixels = 0
    for idx in tqdm(range(len(dataset)), desc=desc):
        image = dataset[idx]['image']['array']
        total_mae += np.abs(image - average_value).sum()
        total_pixels += image.size
    return total_mae / total_pixels

# Calculate MAE for validation and test sets
print("Calculating MAE for validation set...")
val_mae = calculate_mae(val_dataset, train_avg, desc="Evaluating val dataset")
print(f"Validation MAE: {val_mae:.4f}")

print("Calculating MAE for test set...")
test_mae = calculate_mae(test_dataset, train_avg, desc="Evaluating test dataset")
print(f"Test MAE: {test_mae:.4f}")
