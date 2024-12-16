import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
from load_data_multi import load_data_split
from basic_unet import BasicUNet

# Argument parsing
parser = argparse.ArgumentParser(description="Train U-Net model")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--input_channels", type=str, nargs='+', default=["171", "193", "304"], help="Channels for input (e.g., 171 193 304)")
parser.add_argument("--output_channel", type=str, default="335", help="Channel for output (e.g., 335)")
parser.add_argument("--data_dir", type=str, default="/mnt/ceph/users/manand", help="Base folder containing train/val/test datasets")
parser.add_argument("--save_model_dir", type=str, default="/mnt/home/hzhu2/saved_models", help="Directory to save trained models")
parser.add_argument("--concatenate_inputs", default=True, help="Flag to concatenate input channels into a single tensor")
parser.add_argument("--subset_step", type=int, default=None, help="Step size for loading a subset of the dataset")
parser.add_argument("--checkpoint_interval", type=int, default=None, help="Interval (in batches) to save intermediate checkpoints")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(args.save_model_dir, exist_ok=True)

train_paths = {channel: os.path.join(args.data_dir, f"{channel}_train") for channel in args.input_channels}
train_paths["target"] = os.path.join(args.data_dir, f"{args.output_channel}_train")
val_paths = {channel: os.path.join(args.data_dir, f"{channel}_val") for channel in args.input_channels}
val_paths["target"] = os.path.join(args.data_dir, f"{args.output_channel}_val")

# Load training and validation datasets
print("Loading datasets...")
train_loader, val_loader, _ = load_data_split(
    paths_train=train_paths,
    paths_val=val_paths,
    paths_test=None,  # to prevent leakage
    batch_size=args.batch_size,
    concatenate_inputs=args.concatenate_inputs,
    subset_step=args.subset_step
)

# Model initialization
input_channels = len(args.input_channels) if args.concatenate_inputs else 1
print(f"Initializing model with {input_channels} input channels...")
model = BasicUNet(in_channels=input_channels, out_channels=1).to(device)
print("Model initialized.")

# Loss function and optimizer
criterion = nn.L1Loss()  # MAE
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

def calculate_mae(gt, pred):
    return np.mean(np.abs(gt - pred))

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, checkpoint_interval):
    start_time = time.time()
    print("Training started...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Log progress
            if (batch_idx + 1) % 50 == 0:  # Log every 50 batches
                print(f"Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # # Save intermediate checkpoints
            # if (batch_idx + 1) % checkpoint_interval == 0:
            #     checkpoint_path = os.path.join(args.save_model_dir, f"checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pth")
            #     torch.save(model.state_dict(), checkpoint_path)
            #     print(f"Model checkpoint saved at {checkpoint_path}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {avg_train_loss:.4f}")

        # Validate the model
        avg_val_loss, val_mae = validate_model(model, val_loader, criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {avg_val_loss:.4f}, MAE: {val_mae:.4f}")

        # Save model checkpoint
        model_path = os.path.join(args.save_model_dir, f"unet_epoch_{epoch+1}_new.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")

    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")

# Validation function
def validate_model(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            all_preds.append(outputs.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())

    avg_val_loss = val_loss / len(loader)
    model_mae = calculate_mae(np.concatenate(all_targets), np.concatenate(all_preds))
    return avg_val_loss, model_mae

# Train the model
train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    num_epochs=args.epochs,
    checkpoint_interval=args.checkpoint_interval
)
