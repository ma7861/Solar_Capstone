import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import load_data  # Import your custom data loading module
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
from datasets import load_from_disk
#import matplotlib.pyplot as plt
#import sunpy.visualization.colormaps as cm
from pylab import figure, cm
from matplotlib.colors import LogNorm

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to datasets and saved models
path_171 = "/mnt/home/manand/ceph/171"
path_193 = "/mnt/home/manand/ceph/193"
path_304 = "/mnt/home/manand/ceph/304"
path_335 = "/mnt/home/manand/ceph/335"
save_dir = "/mnt/home/manand/ceph/filtered_common_timestamps"
model_path = '/mnt/home/manand/myproj/Solar_Capstone/saved_models/unet_epoch_1_440.pth'

# Load test dataset
subset_ratio = 0.01  # Load 1% of the data
_, test_loader = load_data.load_filtered_data(path_171, path_193, path_304, path_335, batch_size=32, save_dir=save_dir, num_proc=4)
test_loader = DataLoader(test_loader.dataset, batch_size=32, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(
    np.random.choice(len(test_loader.dataset), int(len(test_loader.dataset) * subset_ratio), replace=False)))

# Initialize model (same architecture as during training)
model = Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

# Load the trained model weights
# model_path = os.path.join(save_dir, "unet_epoch_10.pth")
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
    plt.hexbin(gt, predictions, gridsize=1000, cmap='Blues', mincnt=1, vmax=1e5) 
    plt.colorbar(label='Counts')

    # Add a line for perfect predictions
    max_val = max(max(gt), max(predictions))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction (y=x)')

    # Labels and title
    plt.xlabel('Ground Truth (AIA 335)')
    plt.ylabel('Predictions (AIA 335)')
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
all_preds_i = []
all_targets_i = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Get model predictions
        outputs = model(inputs)
        
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

colormap = plt.get_cmap('sdoaia335')
cmap = colormap
norm = plt.Normalize(vmin=10)
lnorm = LogNorm(vmin=10)

for i in range(all_preds.shape[0]):
    image=all_preds_i[i,0,:,:]
    plt.figure(figsize=(10,10))
    imagep = cmap(norm(image))
    plt.imsave('./plot_output/images/'+str(i)+'_unet_pred.png',imagep)
    plt.clf()

    image2=all_targets_i[i,0,:,:]
    plt.figure(figsize=(10,10))
    image2p = cmap(norm(image2))
    plt.imsave('./plot_output/images/'+str(i)+'_unet_true.png',image2p)
    plt.clf()
    
    image3=np.subtract(image,image2)
    plt.figure(figsize=(10,10))
    image3p = cmap(norm(image3))
    plt.imsave('./plot_output/images/'+str(i)+'_unet_error.png',image3p)
    plt.clf()

    fig,axs = plt.subplots(1,3,figsize=(10,10))
    axs[0].imshow(imagep)
    axs[0].axis('off')
    axs[1].imshow(image2p)
    axs[1].axis('off')
    axs[2].imshow(image3,cmap='RdBu')
    axs[2].axis('off')
    plt.savefig('./plot_output/images/'+str(i)+'_unet_all3.png')
    plt.clf()
    
    image=np.log(all_preds_i[i,0,:,:])
    plt.figure(figsize=(10,10))
    imagep = cmap((image))
    plt.imsave('./plot_output/images/'+str(i)+'_unet_pred_log.png',imagep)
    plt.clf()

    image2=np.log(all_targets_i[i,0,:,:])
    plt.figure(figsize=(10,10))
    image2p = cmap((image2))
    plt.imsave('./plot_output/images/'+str(i)+'_unet_true_log.png',image2p)
    plt.clf()
    
    image3=np.subtract(image,image2)
    plt.figure(figsize=(10,10))
    image3p = cmap((image3))
    plt.imsave('./plot_output/images/'+str(i)+'_unet_error_log.png',image3p)
    plt.clf()
    
    fig,axs = plt.subplots(1,3,figsize=(10,10))
    axs[0].imshow(imagep)
    axs[0].axis('off')
    axs[1].imshow(image2p)
    axs[1].axis('off')
    axs[2].imshow(image3,cmap='RdBu')
    axs[2].axis('off')
    plt.savefig('./plot_output/images/'+str(i)+'_unet_all3_log.png')
    plt.clf()

# Generate hexbin plot
plot_hexbin(all_targets, all_preds)
