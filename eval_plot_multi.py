import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from load_data_multi import load_data_split
from segmentation_models_pytorch import Unet
import os

import matplotlib.cm as cm
import pandas as pd
import sunpy.visualization.colormaps as cm

from astropy.time import Time
from sunpy.visualization import axis_labels_from_ctype, wcsaxes_compat

from matplotlib import animation
from IPython.display import HTML

import sunpy
from datasets import concatenate_datasets, DatasetDict
#import tensorflow as tf
#from tensorflow.keras import layers,models
import argparse
from datasets import load_from_disk
#import matplotlib.pyplot as plt
#import sunpy.visualization.colormaps as cm
from pylab import figure, cm
from matplotlib.colors import LogNorm
from basic_unet import BasicUNet
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Test U-Net model")
# parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
# parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--input_channels", type=str, nargs='+', default=["171", "193", "304"], help="Channels for input (e.g., 171 193 304)")
parser.add_argument("--output_channel", type=str, default="335", help="Channel for output (e.g., 335)")
parser.add_argument("--data_dir", type=str, default="/mnt/ceph/users/manand", help="Base folder containing train/val/test datasets")
# parser.add_argument("--save_model_dir", type=str, default="/mnt/home/hzhu2/saved_models", help="Directory to save trained models")
parser.add_argument("--model_dir", type=str, default="/mnt/home/hzhu2/saved_models", help="Directory containing saved trained models")
parser.add_argument("--concatenate_inputs", default=True, help="Flag to concatenate input channels into a single tensor")
parser.add_argument("--subset_step", type=int, default=None, help="Step size for loading a subset of the dataset")
parser.add_argument("--checkpoint_interval", type=int, default=None, help="Interval (in batches) to save intermediate checkpoints")

args = parser.parse_args()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to datasets and saved models
# path_171 = "/mnt/home/manand/ceph/171"
# path_193 = "/mnt/home/manand/ceph/193"
# path_304 = "/mnt/home/manand/ceph/304"
# path_335 = "/mnt/home/manand/ceph/335"
# save_dir = "/mnt/home/manand/ceph/filtered_common_timestamps"
# model_path = '/mnt/home/hzhu2/saved_models/171/unet_epoch_10_new.pth'

test_paths = {channel: os.path.join(args.data_dir, f"{channel}_test") for channel in args.input_channels}
test_paths["target"] = os.path.join(args.data_dir, f"{args.output_channel}_test")

# Load test dataset
# subset_ratio = 0.01  # Load 1% of the data
_, _, test_loader = load_data_split(
    paths_train=None,
    paths_val=None,
    paths_test=test_paths,
    batch_size=args.batch_size,
    concatenate_inputs=args.concatenate_inputs,
    subset_step=args.subset_step
)
# test_loader = DataLoader(test_loader.dataset, batch_size=args.batch_size, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(
#     np.random.choice(len(test_loader.dataset), int(len(test_loader.dataset) * subset_ratio), replace=False)))

# Initialize model (same architecture as during training)
# model = Unet(
#     encoder_name="resnet34",
#     encoder_weights=None,
#     in_channels=3,
#     classes=1
# ).to(device)
input_channels = len(args.input_channels) if args.concatenate_inputs else 1
model = BasicUNet(in_channels=input_channels, out_channels=1).to(device)

# Load the trained model weights
model_path = os.path.join(args.model_dir, "unet_epoch_5_new.pth")
mh = torch.load(model_path,map_location=device)
model.load_state_dict({k.replace("module.",""):v for k,v in mh.items()})
model.eval()

# Define the MAE criterion
criterion = nn.L1Loss()

# Function to create hexbin plot
def plot_hexbin(gt, predictions, save_path="./plot_output"):
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 8))

    # Adjusted hexbin plot
    # plt.hexbin(gt, predictions, gridsize=80, cmap='Blues', mincnt=1, vmax=1e6)
    plt.hexbin(gt, predictions, gridsize=1000, cmap='Blues', mincnt=1, vmax=5, vmin=1) 
    plt.colorbar(label='Counts')

    # Add a line for perfect predictions
    max_val = max(max(gt), max(predictions))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction (y=x)')

    # Labels and title
    plt.xlabel('Ground Truth (AIA 335)')
    plt.ylabel('Predictions (AIA 335)')
    plt.title('Ground Truth vs Predictions (Unet from Scratch)')
    plt.legend()

    # Zoom in on a more relevant range if data is clustered
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)

    # Save the plot
    channels = '_'.join(args.input_channels)
    plot_filename = os.path.join(save_path, f"hexbin_plot_unet_{channels}.png")
    plt.savefig(plot_filename)
    print(f"Plot saved at {plot_filename}")

    plt.close()

# Run evaluation and collect predictions and targets
all_preds = []
all_targets = []
all_preds_i = []
all_targets_i = []

with torch.no_grad():
    total_mae = 0.0
    num_batches = 0
    
    for inputs, targets in tqdm(test_loader, desc="Evaluating", unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)

        # Get model predictions
        outputs = model(inputs)

        batch_mae = torch.mean(torch.abs(outputs - targets)).item()
        total_mae += batch_mae
        num_batches += 1
        
        # Collect predictions and targets for plotting
        all_preds.append(outputs.cpu().numpy().flatten())
        all_targets.append(targets.cpu().numpy().flatten())
        all_preds_i.append(outputs.cpu().numpy())
        all_targets_i.append(targets.cpu().numpy())

        # Release memory after each batch
        del inputs, targets, outputs
        torch.cuda.empty_cache()

# Convert lists to numpy arrays
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)
all_preds_i = np.concatenate(all_preds_i)
all_targets_i = np.concatenate(all_targets_i)

overall_mae = total_mae / num_batches
print(f"Overall MAE: {overall_mae:.4f}")

# Generate hexbin plot
plot_hexbin(all_targets, all_preds)

# colormap = plt.get_cmap('sdoaia335')
# cmap = colormap
# norm = plt.Normalize(vmin=10)
# lnorm = LogNorm(vmin=10)

# for i in range(all_preds.shape[0]):
#     image=all_preds_i[i,0,:,:]
#     plt.figure(figsize=(10,10))
#     imagep = cmap(norm(image))
#     plt.imsave(f'./plot_output/images/{arg.input_channels}/{i}_unet_pred.png',imagep)
#     plt.clf()

#     image2=all_targets_i[i,0,:,:]
#     plt.figure(figsize=(10,10))
#     image2p = cmap(norm(image2))
#     plt.imsave(f'./plot_output/images/{arg.input_channels}/{i}_unet_true.png',image2p)
#     plt.clf()
    
#     image3=np.subtract(image,image2)
#     plt.figure(figsize=(10,10))
#     image3p = cmap(norm(image3))
#     plt.imsave(f'./plot_output/images/{arg.input_channels}/{i}_unet_error.png',image3p)
#     plt.clf()

#     fig,axs = plt.subplots(1,3,figsize=(10,10))
#     axs[0].imshow(imagep)
#     axs[0].axis('off')
#     axs[1].imshow(image2p)
#     axs[1].axis('off')
#     axs[2].imshow(image3,cmap='RdBu')
#     axs[2].axis('off')
#     plt.savefig(f'./plot_output/images/{arg.input_channels}/{i}_unet_all3.png')
#     plt.clf()
    
#     image=np.log(all_preds_i[i,0,:,:])
#     plt.figure(figsize=(10,10))
#     imagep = cmap((image))
#     plt.imsave(f'./plot_output/images/{arg.input_channels}/{i}_unet_pred_log.png',imagep)
#     plt.clf()

#     image2=np.log(all_targets_i[i,0,:,:])
#     plt.figure(figsize=(10,10))
#     image2p = cmap((image2))
#     plt.imsave(f'./plot_output/images/{arg.input_channels}/{i}_unet_true_log.png',image2p)
#     plt.clf()
    
#     image3=np.subtract(image,image2)
#     plt.figure(figsize=(10,10))
#     image3p = cmap((image3))
#     plt.imsave(f'./plot_output/images/{arg.input_channels}/{i}_unet_error_log.png',image3p)
#     plt.clf()
    
#     fig,axs = plt.subplots(1,3,figsize=(10,10))
#     axs[0].imshow(imagep)
#     axs[0].axis('off')
#     axs[1].imshow(image2p)
#     axs[1].axis('off')
#     axs[2].imshow(image3,cmap='RdBu')
#     axs[2].axis('off')
#     plt.savefig(f'./plot_output/images/{arg.input_channels}/{i}_unet_all3_log.png')
#     plt.clf()

