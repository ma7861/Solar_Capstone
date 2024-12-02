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
            input_tensor = self.ds_inputs[self.single_channel][idx]['image']['array']
    
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
    train_paths, val_paths, test_path, batch_size=32, transform=None,
    concatenate_inputs=False, output_channel="335", subset_step=None, num_proc=1
):
    """
    Load training, validation, and testing datasets.

    Args:
        train_paths (dict): Paths to training datasets for input channels.
        val_paths (dict): Paths to validation datasets for input channels.
        test_path (str): Path to testing dataset for the target channel.
        batch_size (int): Batch size for the DataLoader.
        transform: Transformations to apply to data.
        concatenate_inputs (bool): Whether to concatenate input channels into one.
        output_channel (str): The channel to use as the target for testing.
        subset_step (int): Step size for selecting a subset.
        num_proc (int): Number of processes for parallel data loading.

    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for train, val, and test sets.
    """
    # Load datasets
    train_inputs = {channel: load_from_disk(path).with_format("numpy") for channel, path in train_paths.items() if channel != "target"}
    val_inputs = {channel: load_from_disk(path).with_format("numpy") for channel, path in val_paths.items() if channel != "target"}

    train_target = load_from_disk(os.path.join(train_paths[list(train_paths.keys())[0]], "../335_train")).with_format("numpy")
    val_target = load_from_disk(os.path.join(val_paths[list(val_paths.keys())[0]], "../335_val")).with_format("numpy")
    test_target = load_from_disk(test_path).with_format("numpy")

    # Apply subset step
    if subset_step:
        train_inputs = {channel: apply_subset_step(ds, subset_step) for channel, ds in train_inputs.items()}
        val_inputs = {channel: apply_subset_step(ds, subset_step) for channel, ds in val_inputs.items()}
        train_target = apply_subset_step(train_target, subset_step)
        val_target = apply_subset_step(val_target, subset_step)
        test_target = apply_subset_step(test_target, subset_step)

    # Create PyTorch datasets
    train_dataset = AIA_Dataset(train_inputs, train_target, transform=transform, concatenate_inputs=concatenate_inputs)
    val_dataset = AIA_Dataset(val_inputs, val_target, transform=transform, concatenate_inputs=concatenate_inputs)
    test_dataset = AIA_Dataset({output_channel: test_target}, test_target, transform=transform, concatenate_inputs=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



def load_data_split_2(
    train_paths, val_paths, test_paths, batch_size=32, transform=None,
    concatenate_inputs=False, output_channel="335", subset_step=None, num_proc=1
):
    
    test_inputs = {channel: load_from_disk(path).with_format("numpy") for channel, path in test_paths.items() if channel != "target"}

    
    test_target = load_from_disk(os.path.join(test_paths[list(test_paths.keys())[0]], "../335_test")).with_format("numpy")

    if subset_step:
        test_inputs = {channel: apply_subset_step(ds, subset_step) for channel, ds in test_inputs.items()}
        test_target = apply_subset_step(test_target, subset_step)

    test_dataset = AIA_Dataset(test_inputs, test_target, transform=transform, concatenate_inputs=concatenate_inputs)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader