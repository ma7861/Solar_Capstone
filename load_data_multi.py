import os
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
import numpy as np


class AIA_Dataset(Dataset):
    def __init__(self, ds_inputs, ds_target, transform=None, concatenate_inputs=True):
        """
        Args:
            ds_inputs (dict): Dictionary of datasets for input channels.
            ds_target (Dataset): Target dataset (e.g., 335).
            transform: Optional transformations for the input and target.
            concatenate_inputs (bool): Whether to concatenate input channels into one.
        """
        self.ds_inputs = ds_inputs
        self.ds_target = ds_target
        self.transform = transform
        self.concatenate_inputs = concatenate_inputs

    def __len__(self):
        return len(self.ds_target)

    def __getitem__(self, idx):
        idx = int(idx)

        # Load input channels
        if self.concatenate_inputs:
            inputs = {channel: self.ds_inputs[channel][idx]['image']['array'] for channel in self.ds_inputs}
            input_tensor = np.concatenate([inputs[channel] for channel in sorted(inputs.keys())], axis=0)
        else:
            input_tensor = self.ds_inputs[list(self.ds_inputs.keys())[0]][idx]['image']['array']

        # Load target
        target = self.ds_target[idx]['image']['array']

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target = self.transform(target)

        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return input_tensor, target


def apply_subset_step(dataset, step):
    """
    Select a subset of the dataset with a fixed step size.

    Args:
        dataset: The dataset to subset.
        step: Step size for selecting samples.

    Returns:
        Subset dataset with every nth sample.
    """
    if step:
        indices = list(range(0, len(dataset), step))
        print(f"Selecting a subset with step {step}: {len(indices)} samples")
        return dataset.select(indices)
    return dataset


def load_data_split(
    paths_train=None, paths_val=None, paths_test=None, batch_size=32, transform=None,
    concatenate_inputs=False, subset_step=None
):
    """
    Load training, validation, and testing datasets.

    Args:
        paths_train (dict): Paths to training datasets for input channels and target channel.
        paths_val (dict): Paths to validation datasets for input channels and target channel.
        paths_test (dict): Paths to testing datasets for input channels and target channel.
        batch_size (int): Batch size for the DataLoader.
        transform: Transformations to apply to data.
        concatenate_inputs (bool): Whether to concatenate input channels into one.
        subset_step (int): Step size for selecting a subset.

    Returns:
        Tuple containing train_loader, val_loader, test_loader (can return `None` if any is not specified).
    """
    loaders = []

    for split_name, paths in [("train", paths_train), ("val", paths_val), ("test", paths_test)]:
        if paths is None:
            loaders.append(None)
            continue

        # Separate input and target paths
        input_paths = {channel: path for channel, path in paths.items() if channel != "target"}
        target_path = paths["target"]

        # Load datasets
        ds_inputs = {channel: load_from_disk(path).with_format("numpy") for channel, path in input_paths.items()}
        ds_target = load_from_disk(target_path).with_format("numpy")

        # Apply subset step
        if subset_step:
            ds_inputs = {channel: apply_subset_step(ds, subset_step) for channel, ds in ds_inputs.items()}
            ds_target = apply_subset_step(ds_target, subset_step)

        # Create PyTorch dataset and DataLoader
        dataset = AIA_Dataset(ds_inputs, ds_target, transform=transform, concatenate_inputs=concatenate_inputs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split_name == "train"))
        loaders.append(loader)

    return tuple(loaders)
