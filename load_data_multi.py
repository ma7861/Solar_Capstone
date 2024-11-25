import os
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk, DatasetDict

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
    
        # Convert to tensors
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
    
        return input_tensor, target


def filter_by_year(dataset, years, num_proc=1):
    return dataset.filter(lambda example: any(example['image']['date'].startswith(str(year)) for year in years), num_proc=num_proc)

def filter_by_date(dataset, start_date, end_date, num_proc=1):
    return dataset.filter(
        lambda example: start_date <= example['image']['date'] <= end_date,
        num_proc=num_proc
    )

def load_data(paths_inputs, path_target, batch_size=32, transform=None, save_dir="./ceph/multi_input", num_proc=1):
    os.makedirs(save_dir, exist_ok=True)

    inputs = {channel: load_from_disk(path).with_format("numpy") for channel, path in paths_inputs.items()}
    target = load_from_disk(path_target).with_format("numpy")

    # date ranges
    test_years = [2015]
    val_date_ranges = [
        ("2014-07-01", "2014-12-31"),
        ("2016-01-01", "2016-06-30")
    ]
    train_exclude_years = test_years + [2014, 2016]

    # Filter datasets for train, val, and test
    train_inputs = {
        channel: inputs[channel].filter(
            lambda example: not any(example['image']['date'].startswith(str(year)) for year in train_exclude_years),
            num_proc=num_proc
        )
        for channel in inputs
    }
    train_target = target.filter(
        lambda example: not any(example['image']['date'].startswith(str(year)) for year in train_exclude_years),
        num_proc=num_proc
    )

    val_inputs = {
        channel: sum(
            [filter_by_date(inputs[channel], start, end, num_proc=num_proc) for start, end in val_date_ranges],
            start=inputs[channel].select([])
        )
        for channel in inputs
    }
    val_target = sum(
        [filter_by_date(target, start, end, num_proc=num_proc) for start, end in val_date_ranges],
        start=target.select([])
    )

    test_inputs = {
        channel: filter_by_year(inputs[channel], test_years, num_proc=num_proc) for channel in inputs
    }
    test_target = filter_by_year(target, test_years, num_proc=num_proc)

    # Save datasets
    train_data = DatasetDict({"inputs": train_inputs, "target": train_target})
    val_data = DatasetDict({"inputs": val_inputs, "target": val_target})
    test_data = DatasetDict({"inputs": test_inputs, "target": test_target})

    train_save_path = os.path.join(save_dir, "train_data")
    val_save_path = os.path.join(save_dir, "val_data")
    test_save_path = os.path.join(save_dir, "test_data")

    print(f"Saving train dataset to {train_save_path}...")
    train_data.save_to_disk(train_save_path)
    print("Train dataset saved.")

    print(f"Saving validation dataset to {val_save_path}...")
    val_data.save_to_disk(val_save_path)
    print("Validation dataset saved.")

    print(f"Saving test dataset to {test_save_path}...")
    test_data.save_to_disk(test_save_path)
    print("Test dataset saved.")

    # Create PyTorch datasets and DataLoaders
    train_dataset = AIA_Dataset(train_inputs, train_target, transform=transform)
    val_dataset = AIA_Dataset(val_inputs, val_target, transform=transform)
    test_dataset = AIA_Dataset(test_inputs, test_target, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    paths_inputs = {
        "171": "./ceph/171/171",
        "193": "./ceph/193/193",
        "304": "./ceph/304/304"
    }
    path_target = "./ceph/335/335"

    train_loader, val_loader, test_loader = load_data(paths_inputs, path_target, batch_size=32, num_proc=4)

    print("Train, Validation, and Test sets loaded successfully.")
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
