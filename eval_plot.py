import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import load_data  # Import your custom data loading module
from segmentation_models_pytorch import Unet
import os

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to datasets and saved models
path_171 = "../sho/ceph/20241008_cwan1/ceph/datasets/AIA/171"
path_193 = "../sho/ceph/20241008_cwan1/ceph/datasets/AIA/193"
save_dir = "./ceph/filtered_common_timestamps"
model_path = '/mnt/home/hzhu2/saved_models_from_scratch/unet_epoch_10.pth'

# Load test dataset
subset_ratio = 0.01  # Load 1% of the data
_, test_loader = load_data.load_filtered_data(path_171, path_193, batch_size=16, save_dir=save_dir, num_proc=4)
test_loader = DataLoader(test_loader.dataset, batch_size=16, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(
    np.random.choice(len(test_loader.dataset), int(len(test_loader.dataset) * subset_ratio), replace=False)))

# Initialize model (same architecture as during training)
model = model = Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1
).to(device)

# Load the trained model weights
# model_path = os.path.join(save_dir, "unet_epoch_10.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define the MAE criterion
criterion = nn.L1Loss()

# Function to create hexbin plot
def plot_hexbin(gt, predictions, save_path="./plot_output"):
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 8))

    # Adjusted hexbin plot
    # plt.hexbin(gt, predictions, gridsize=80, cmap='Blues', mincnt=1, vmax=1e6)
    plt.hexbin(gt, predictions, gridsize=1000, cmap='Blues', mincnt=1, vmax=1e5) 
    plt.colorbar(label='Counts')

    # Add a line for perfect predictions
    max_val = max(max(gt), max(predictions))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction (y=x)')

    # Labels and title
    plt.xlabel('Ground Truth (AIA 171)')
    plt.ylabel('Predictions (AIA 171)')
    plt.title('Ground Truth vs Predictions (Encoder: Resnet34)')
    plt.legend()

    # Zoom in on a more relevant range if data is clustered
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)

    # Save the plot
    plot_filename = os.path.join(save_path, "improved_hexbin_plot_unet.png")
    plt.savefig(plot_filename)
    print(f"Plot saved at {plot_filename}")

    plt.close()

# Run evaluation and collect predictions and targets
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Get model predictions
        outputs = model(inputs)
        
        # Collect predictions and targets for plotting
        all_preds.append(outputs.cpu().numpy().flatten())
        all_targets.append(targets.cpu().numpy().flatten())

        # Release memory after each batch
        del inputs, targets, outputs
        torch.cuda.empty_cache()

# Convert lists to numpy arrays
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

# Generate hexbin plot
plot_hexbin(all_targets, all_preds)
