import os
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk, DatasetDict
import numpy as np

os.environ["TMPDIR"] = "/tmp"
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

class AIA_Dataset(Dataset):
    def __init__(self, ds_inputs, ds_target, transform=None):
        self.ds_inputs = ds_inputs
        self.ds_target = ds_target
        self.transform = transform

    def __len__(self):
        return len(self.ds_target)

    def __getitem__(self, idx):
        idx = int(idx)
        inputs = {channel: self.ds_inputs[channel][idx]['image']['array'] for channel in self.ds_inputs}
        input_tensor = np.concatenate([inputs[channel] for channel in sorted(inputs.keys())], axis=0)
        target = self.ds_target[idx]['image']['array']

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target = self.transform(target)

        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return input_tensor, target

def load_filtered_datasets(paths_inputs, path_target, save_dir, num_proc=4):
    os.makedirs(save_dir, exist_ok=True)
    
    inputs = {channel: load_from_disk(path).with_format("numpy") for channel, path in paths_inputs.items()}
    target = load_from_disk(path_target).with_format("numpy")

    train_save_path = os.path.join(save_dir, "train_target")
    if os.path.exists(train_save_path):
        print(f"Loading filtered datasets from {train_save_path}...")
        train_target = load_from_disk(train_save_path)
    else:
        print("Filtering datasets...")
        train_target = target.filter(lambda ex: ex['image']['date'] not in excluded_dates, num_proc=num_proc)
        train_target.save_to_disk(train_save_path)

    return inputs, train_target

if __name__ == "__main__":
    paths_inputs = {
        "171": "./ceph/171/171",
        "193": "./ceph/193/193",
        "304": "./ceph/304/304"
    }
    path_target = "./ceph/335/335"

    # directory to save filtered datasets
    save_dir = "./ceph/multi_input"

    # use a subset for testing
    use_subset = False
    subset_size = 100

    if use_subset:
        print(f"Testing on a tiny subset of {subset_size} examples...")
        inputs = {
            channel: load_from_disk(path).select(range(subset_size)).with_format("numpy")
            for channel, path in paths_inputs.items()
        }
        target = load_from_disk(path_target).select(range(subset_size)).with_format("numpy")

        # save the subset for debugging
        subset_save_dir = os.path.join(save_dir, "tiny_subset")
        os.makedirs(subset_save_dir, exist_ok=True)

        for channel, ds in inputs.items():
            save_path = os.path.join(subset_save_dir, f"{channel}_subset")
            print(f"Saving tiny subset for channel {channel} to {save_path}...")
            ds.save_to_disk(save_path)

        target_save_path = os.path.join(subset_save_dir, "target_subset")
        print(f"Saving tiny subset for target to {target_save_path}...")
        target.save_to_disk(target_save_path)

        # reload the saved subset to verify
        print("Reloading saved tiny subset for verification...")
        reloaded_inputs = {
            channel: load_from_disk(os.path.join(subset_save_dir, f"{channel}_subset")).with_format("numpy")
            for channel in inputs
        }
        reloaded_target = load_from_disk(target_save_path).with_format("numpy")

        tiny_dataset = AIA_Dataset(reloaded_inputs, reloaded_target)
        tiny_loader = DataLoader(tiny_dataset, batch_size=8, shuffle=True)

        print(f"Tiny subset loaded with {len(tiny_loader.dataset)} examples.")

        for batch_idx, (input_tensor, target_tensor) in enumerate(tiny_loader):
            print(f"Batch {batch_idx + 1}:")
            print(f"Input tensor shape: {input_tensor.shape}")
            print(f"Target tensor shape: {target_tensor.shape}")
            break # check the first batch
    else:
        print("Running full dataset filtering and saving...")
        train_loader, val_loader, test_loader = load_data(
            paths_inputs, path_target, batch_size=32, save_dir=save_dir, num_proc=4
        )
        print("Filtered datasets loaded successfully.")

        print(f"Train set size: {len(train_loader.dataset)}")
        print(f"Validation set size: {len(val_loader.dataset)}")
        print(f"Test set size: {len(test_loader.dataset)}")


