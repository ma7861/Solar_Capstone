import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import load_data
from segmentation_models_pytorch import Unet
import os
import time

# Argument parsing
parser = argparse.ArgumentParser(description="Train U-Net model from scratch")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

save_model_dir = "/mnt/home/hzhu2/saved_models_from_scratch"
os.makedirs(save_model_dir, exist_ok=True)

path_171 = "/mnt/home/sho/ceph/20241008_cwan1/ceph/datasets/AIA/171"
path_193 = "/mnt/home/sho/ceph/20241008_cwan1/ceph/datasets/AIA/193"
save_dir = "/mnt/home/hzhu2/ceph/filtered_common_timestamps"

# Load train and test datasets
train_loader, test_loader = load_data.load_filtered_data(path_171, path_193, batch_size=args.batch_size, save_dir=save_dir, num_proc=4)

# Initialize U-Net model without pretrained weights
print("Initializing model...")
model = Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1
).to(device)
print("Model initialized.")

# Loss function and optimizer
criterion = nn.L1Loss()  # MAE
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# adamw
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

# MAE calculation function
def calculate_mae(gt, pred):
    """Calculates Mean Absolute Error."""
    return np.mean(np.abs(gt - pred))

# Training loop
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    start_time = time.time()
    print("Training started...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_duration = time.time() - epoch_start_time
        avg_train_loss = train_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")
        print(f"Epoch duration: {epoch_duration:.2f} seconds")

        # Save model checkpoint
        model_path = os.path.join(save_model_dir, f"unet_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")

        # Validate the model
        avg_val_loss, model_mae = validate_model(model, test_loader, criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}, Model MAE: {model_mae:.4f}")

    total_training_time = time.time() - start_time
    print(f"Training completed in {total_training_time / 60:.2f} minutes.")

def validate_model(model, test_loader, criterion):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            all_preds.append(outputs.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())

    avg_val_loss = val_loss / len(test_loader)
    model_mae = calculate_mae(np.concatenate(all_targets), np.concatenate(all_preds))
    return avg_val_loss, model_mae

# Scatter plot function using plt.hexbin
def plot_hexbin(gt, predictions, baseline_predictions, model_mae, baseline_mae, save_path="./plot_output"):
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(10, 10))
    plt.hexbin(gt, predictions, gridsize=50, cmap='Blues', mincnt=1, alpha=0.7)
    plt.colorbar(label='Counts')
    
    plt.hexbin(gt, baseline_predictions, gridsize=50, cmap='Oranges', mincnt=1, alpha=0.5)
    plt.xlabel('Ground Truth (AIA 171)')
    plt.ylabel('Predictions (AIA 171)')
    plt.title("Model vs Baseline")
    
    plt.text(0.05, 0.9, f"Model MAE: {model_mae:.4f}", transform=plt.gca().transAxes, fontsize=12, color="blue")
    plt.text(0.05, 0.85, f"Baseline MAE: {baseline_mae:.4f}", transform=plt.gca().transAxes, fontsize=12, color="orange")
    
    plot_filename = os.path.join(save_path, "hexbin_plot.png")
    plt.savefig(plot_filename)
    plt.close()

# Baseline model calculation
def baseline_model(train_loader, test_loader):
    print("Calculating baseline model...")

    total_sum, total_count = 0.0, 0
    for _, targets in train_loader:
        total_sum += targets.sum().item()
        total_count += targets.numel()
    
    avg_value = total_sum / total_count
    print(f"Average target value (baseline prediction): {avg_value:.4f}")

    all_baseline_preds = []
    all_targets = []

    for _, targets in test_loader:
        baseline_pred = np.full_like(targets.cpu().numpy(), avg_value)
        all_baseline_preds.append(baseline_pred.flatten())
        all_targets.append(targets.cpu().numpy().flatten())

    baseline_mae = calculate_mae(np.concatenate(all_targets), np.concatenate(all_baseline_preds))
    return np.concatenate(all_targets), np.concatenate(all_baseline_preds), baseline_mae

# Train and Evaluate
num_epochs = 10
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=args.epochs)

# Get results from the trained model and baseline
gt, model_preds, model_mae = validate_model(model, test_loader, criterion)
_, baseline_preds, baseline_mae = baseline_model(train_loader, test_loader)

# Plot comparison of baseline vs trained model
plot_hexbin(gt, model_preds, baseline_preds, model_mae, baseline_mae)
