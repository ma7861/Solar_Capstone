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
        # inputs = {channel: self.ds_inputs[channel][idx]['image']['array'] for channel in self.ds_inputs}

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

def load_data_split(
    train_paths, val_paths, test_path, batch_size=32, transform=None,
    concatenate_inputs=False, output_channel="335", num_proc=1
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
        num_proc (int): Number of processes for parallel data loading.

    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for train, val, and test sets.
    """
    # Load datasets
    # train_inputs = {channel: load_from_disk(path).with_format("numpy") for channel, path in train_paths.items()}
    # val_inputs = {channel: load_from_disk(path).with_format("numpy") for channel, path in val_paths.items()}
    # print('train_paths', train_paths.items())
    train_inputs = {channel: load_from_disk(path).with_format("numpy") for channel, path in train_paths.items() if channel != "target"}
    val_inputs = {channel: load_from_disk(path).with_format("numpy") for channel, path in val_paths.items() if channel != "target"}

    train_target = load_from_disk(os.path.join(train_paths[list(train_paths.keys())[0]], "../335_train")).with_format("numpy")
    val_target = load_from_disk(os.path.join(val_paths[list(val_paths.keys())[0]], "../335_val")).with_format("numpy")
    test_target = load_from_disk(test_path).with_format("numpy")

    # Create PyTorch datasets
    train_dataset = AIA_Dataset(train_inputs, train_target, transform=transform, concatenate_inputs=concatenate_inputs)
    val_dataset = AIA_Dataset(val_inputs, val_target, transform=transform, concatenate_inputs=concatenate_inputs)
    test_dataset = AIA_Dataset({output_channel: test_target}, test_target, transform=transform, concatenate_inputs=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


