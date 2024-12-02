import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from segmentation_models_pytorch import Unet
import os
import time
from load_data_multi import load_data_split_2
from basic_unet import BasicUNet
import numpy as np


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

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

# Argument parsing
# Argument parsing
parser = argparse.ArgumentParser(description="Train U-Net model with explicit dataset folders")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--input_channels", type=str, nargs='+', default=["171", "193", "304"], help="Channels for input (e.g., 171 193 304)")
parser.add_argument("--output_channel", type=str, default="335", help="Channel for output (e.g., 335)")
parser.add_argument("--data_dir", type=str, default="/mnt/ceph/users/manand", help="Base folder containing train/val/test datasets")
parser.add_argument("--save_model_dir", type=str, default="/mnt/home/manand/myproj/Solar_Capstone/saved_models", help="Directory to save trained models")
parser.add_argument("--concatenate_inputs", action="store_true", help="Flag to concatenate input channels into a single tensor")
parser.add_argument("--subset_step", type=int, default=None, help="Step size for loading a subset of the dataset")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure the save directory exists
os.makedirs(args.save_model_dir, exist_ok=True)

# Paths to datasets
train_paths = {channel: os.path.join(args.data_dir, f"{channel}_train") for channel in args.input_channels}
train_paths["target"] = os.path.join(args.data_dir, f"{args.output_channel}_train")
val_paths = {channel: os.path.join(args.data_dir, f"{channel}_val") for channel in args.input_channels}
val_paths["target"] = os.path.join(args.data_dir, f"{args.output_channel}_val")
test_paths = {channel: os.path.join(args.data_dir, f"{channel}_test") for channel in args.input_channels}
test_paths["target"] = os.path.join(args.data_dir, f"{args.output_channel}_test")

test_loader = load_data_split_2(
    train_paths, val_paths, test_paths,
    batch_size=args.batch_size,
    concatenate_inputs=args.concatenate_inputs,
    output_channel=args.output_channel,
    subset_step=args.subset_step
)

input_channels = len(args.input_channels) if args.concatenate_inputs else 1
print(f"Initializing model with {input_channels} input channels...")

# MAE calculation function
def calculate_mae(gt, pred):
    return np.mean(np.abs(gt - pred))

def save_images(model,all_preds_i,all_targets_i):

    colormap = plt.get_cmap('sdoaia335')
    cmap = colormap
    norm = plt.Normalize(vmin=10)
    lnorm = LogNorm(vmin=10)
    
    for i in [10,1234,3500]: #3 random timestamps for comparision across models
        image=all_preds_i[i,0,:,:]
        plt.figure(figsize=(10,10))
        imagep = cmap(norm(image))
        #plt.imsave('/mnt/home/manand/myproj/Solar_Capstone/plot_output/images/'+model+'_unet_pred.png',imagep)
        plt.clf()
    
        image2=all_targets_i[i,0,:,:]
        plt.figure(figsize=(10,10))
        image2p = cmap(norm(image2))
        #plt.imsave('/mnt/home/manand/myproj/Solar_Capstone/plot_output/images/'+str(i)+'_unet_true.png',image2p)
        plt.clf()
        
        image3=np.subtract(image,image2)
        plt.figure(figsize=(10,10))
        image3p = cmap(norm(image3))
        #plt.imsave('/mnt/home/manand/myproj/Solar_Capstone/plot_output/images/'+str(i)+'_unet_error.png',image3p)
        plt.clf()
    
        fig,axs = plt.subplots(1,3,figsize=(10,10))
        axs[0].imshow(imagep)
        axs[0].axis('off')
        axs[1].imshow(image2p)
        axs[1].axis('off')
        axs[2].imshow(image3,cmap='RdBu')
        axs[2].axis('off')
        plt.savefig('/mnt/home/manand/myproj/Solar_Capstone/plot_output/images/'+'all3_'+str(i)+'.png')
        plt.clf()
        
        image=np.log(all_preds_i[i,0,:,:])
        plt.figure(figsize=(10,10))
        imagep = cmap((image))
        #plt.imsave('./plot_output/images/'+str(i)+'_unet_pred_log.png',imagep)
        plt.clf()
    
        image2=np.log(all_targets_i[i,0,:,:])
        plt.figure(figsize=(10,10))
        image2p = cmap((image2))
        #plt.imsave('./plot_output/images/'+str(i)+'_unet_true_log.png',image2p)
        plt.clf()
        
        image3=np.subtract(image,image2)
        plt.figure(figsize=(10,10))
        image3p = cmap((image3))
        #plt.imsave('./plot_output/images/'+str(i)+'_unet_error_log.png',image3p)
        plt.clf()
        
        fig,axs = plt.subplots(1,3,figsize=(10,10))
        axs[0].imshow(imagep)
        axs[0].axis('off')
        axs[1].imshow(image2p)
        axs[1].axis('off')
        axs[2].imshow(image3,cmap='RdBu')
        axs[2].axis('off')
        plt.savefig('/mnt/home/manand/myproj/Solar_Capstone/plot_output/images/'+'all3_'+str(i)+'_log.png')
        plt.clf()

def plot_hexbin(model_p,gt, predictions, save_path="/mnt/home/manand/myproj/Solar_Capstone/plot_output/"):
    plt.figure(figsize=(8, 8))

    plt.hexbin(gt, predictions, gridsize=1000, cmap='Blues', mincnt=1) 
    plt.colorbar(label='Counts')

    # Add a line for perfect predictions
    max_val = max(max(gt), max(predictions))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction (y=x)')

    # Labels and title
    plt.xlabel('Ground Truth (AIA 335)')
    plt.ylabel('Predictions (AIA 335)')
    plt.title('Ground Truth vs Predictions')
    plt.legend()

    # Zoom in on a more relevant range if data is clustered
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)

    # Save the plot
    plot_filename = os.path.join(save_path, models+"_hexbin.png")
    plt.savefig(plot_filename)
    print(f"Plot saved at {plot_filename}")

    plt.close()


def eval_model(model, loader, criterion):
    model.eval()
    test_loss = 0.0
    all_preds, all_targets = [], []
    all_preds_map, all_targets_map = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            all_preds.append(outputs.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())
            all_preds_map.append(outputs.cpu().numpy())
            all_targets_map.append(targets.cpu().numpy())

    avg_test_loss = test_loss / len(loader)
    model_mae = calculate_mae(np.concatenate(all_targets), np.concatenate(all_preds))
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    plot_hexbin(model,all_targets,all_preds)
    
    all_preds_map = np.concatenate(all_preds_map)
    all_targets_map = np.concatenate(all_targets_map)

    save_images(model,all_preds_map,all_targets_map)
    
    return avg_test_loss, model_mae


model = BasicUNet(
    in_channels=ics,
    out_channels=1
).to(device)
models = '/mnt/home/manand/myproj/Solar_Capstone/saved_models/171193304_335_5.pth'
model.load_state_dict(torch.load(models,weights_only=True))
criterion = nn.L1Loss()  # MAE
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
avg_test_loss,test_mae = eval_model(model, test_loader, criterion)
print(f"Test Loss for model {models}: {avg_test_loss:.4f}, Test MAE: {test_mae:.4f}")
