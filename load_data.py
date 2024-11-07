import os
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk

class AIA_Dataset(Dataset):
    def __init__(self, ds_171, ds_193, transform=None):
        """
        Initialize the dataset with 171 and 193 channels.
        Args:
            ds_171: Dataset for AIA 171 channel.
            ds_193: Dataset for AIA 193 channel.
            transform: Transformations to apply to the data (if any).
        """
        self.ds_171 = ds_171
        self.ds_193 = ds_193
        self.transform = transform

    def __len__(self):
        return len(self.ds_171)

    def __getitem__(self, idx):
        idx = int(idx)
        image_171 = self.ds_171[idx]['image']['array']
        image_193 = self.ds_193[idx]['image']['array']
        
        # Apply transformations if any
        if self.transform:
            image_171 = self.transform(image_171)
            image_193 = self.transform(image_193)
        
        return torch.tensor(image_193, dtype=torch.float32), torch.tensor(image_171, dtype=torch.float32)

def filter_and_save(dataset, year, save_path, num_proc=4):
    """
    Filter the dataset by year and save it to disk.
    Args:
        dataset: The dataset to filter.
        year: The year to filter by.
        save_path: Path to save the filtered dataset.
        num_proc: Number of processes for parallel filtering.
    Returns:
        The filtered dataset.
    """
    filtered_dataset = dataset.filter(lambda example: example['image']['date'].startswith(year), num_proc=num_proc)

    # Save the filtered dataset to disk
    filtered_dataset.save_to_disk(save_path)
    return filtered_dataset

def load_filtered_data(path_171, path_193, batch_size=32, transform=None, save_dir="./ceph/filtered_data", num_proc=4):
    """
    Load and filter the AIA datasets by year, and store filtered datasets to avoid re-filtering.
    Args:
        path_171: Path to the AIA 171 dataset.
        path_193: Path to the AIA 193 dataset.
        batch_size: Batch size for the DataLoader.
        transform: Transformations to apply to the data.
        save_dir: Directory to store the filtered datasets.
        num_proc: Number of processes for parallel filtering.
    Returns:
        train_loader, test_loader: DataLoader objects for training and testing.
    """

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load original datasets
    ds_171 = load_from_disk(path_171).with_format("numpy")
    ds_193 = load_from_disk(path_193).with_format("numpy")

    # Paths to store filtered datasets
    train_171_path = os.path.join(save_dir, "train_171")
    test_171_path = os.path.join(save_dir, "test_171")
    train_193_path = os.path.join(save_dir, "train_193")
    test_193_path = os.path.join(save_dir, "test_193")

    # Filter and save, or load filtered datasets
    if not os.path.exists(train_171_path):
        print("Filtering and saving AIA 171 (2014)...")
        train_ds_171 = filter_and_save(ds_171, '2014', train_171_path, num_proc=num_proc)
    else:
        print("Loading filtered AIA 171 (2014)...")
        train_ds_171 = load_from_disk(train_171_path)

    if not os.path.exists(test_171_path):
        print("Filtering and saving AIA 171 (2016)...")
        test_ds_171 = filter_and_save(ds_171, '2016', test_171_path, num_proc=num_proc)
    else:
        print("Loading filtered AIA 171 (2016)...")
        test_ds_171 = load_from_disk(test_171_path)

    if not os.path.exists(train_193_path):
        print("Filtering and saving AIA 193 (2014)...")
        train_ds_193 = filter_and_save(ds_193, '2014', train_193_path, num_proc=num_proc)
    else:
        print("Loading filtered AIA 193 (2014)...")
        train_ds_193 = load_from_disk(train_193_path)

    if not os.path.exists(test_193_path):
        print("Filtering and saving AIA 193 (2016)...")
        test_ds_193 = filter_and_save(ds_193, '2016', test_193_path, num_proc=num_proc)
    else:
        print("Loading filtered AIA 193 (2016)...")
        test_ds_193 = load_from_disk(test_193_path)

    # Now you have filtered datasets loaded
    train_dataset = AIA_Dataset(train_ds_171, train_ds_193, transform=transform)
    test_dataset = AIA_Dataset(test_ds_171, test_ds_193, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    # Example usage
    path_171 = "../sho/ceph/20241008_cwan1/ceph/datasets/AIA/171"
    path_193 = "../sho/ceph/20241008_cwan1/ceph/datasets/AIA/193"

    # Load data with batch size 32 and parallel filtering with 4 processes
    train_loader, test_loader = load_filtered_data(path_171, path_193, batch_size=32, num_proc=4)

    # Example: Iterate over the train_loader
    for batch_idx, (input_193, target_171) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Input (193) shape: {input_193.shape}")
        print(f"Target (171) shape: {target_171.shape}")
        break  # For demonstration, just process the first batch
