import time
import os

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms, datasets


def generate_dataset_indices(full_dataset, num_examples_per_class):
    """
    This method evenly divides up each individual class into training, validation
    and test indices (70/15/15) split.
    Full dataset is ordered already by class which is why this works nicely.
    
    full_dataset: torch.utils.data.Dataset
    num_examples_per_class: the number of examples per class. This was implemented to speed up training.
    """
    train, val, test = [], [], []
    index_start = 0
    curr_target = 0
    error_indices = set()
    # Evenly divide each class into train, val, test
    # Append a null element so final class gets added
    for i, (data, target) in enumerate(full_dataset + [(None, None)]):
        if data is not None and data.shape != (3, 128, 128):
            # Skip any non-conforming data
            print('ERROR', i, target, data.shape)
            error_indices.add(i)
            continue
        # Found new target so finished with the old class
        # Add the indices from the old class to train, val, test accordingly
        if target != curr_target:
            indices = [idx for idx in range(index_start, i) if idx not in error_indices]
            # Limit the number of examples in the class
            indices = indices[:num_examples_per_class]
            # Train, val, test split 0.7, 0.15, 0.15
            train_len, val_len = int(len(indices) * 0.7), int(len(indices) * 0.15)
            train.extend(indices[:train_len])
            val.extend(indices[train_len:train_len+val_len])
            test.extend(indices[train_len+val_len:])
            index_start = i
            print(f'Switched target from {curr_target} to {target}')
            curr_target = target
    return train, val, test
    
def load_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    full_dataset = datasets.ImageFolder(root='data', transform=transform)
    return full_dataset
    
def split_dataset(full_dataset, num_examples_per_class=300, batch_size=64):
    """Splits the full dataset into train, validation, and test DataLoaders."""
    train_indices, val_indices, test_indices = generate_dataset_indices(full_dataset, num_examples_per_class)
    print('Total # of data=', len(full_dataset))
    print('Train, val, test data lengths = ', len(train_indices), len(val_indices), len(test_indices))
    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(full_dataset, batch_size=1, sampler=SubsetRandomSampler(val_indices))
    test_loader = DataLoader(full_dataset, batch_size=1, sampler=SubsetRandomSampler(test_indices))
    return train_loader, val_loader, test_loader

def get_small_dataloader(full_dataset, num_classes=4):
    indices = []
    count = 0
    curr_class = 0
    for i, data in enumerate(full_dataset):
        if data[1] == curr_class:
            indices.append(i)
            if len(indices) % 50 == 0:  # 50 of each class
                curr_class += 1
            if curr_class == num_classes:
                return DataLoader(full_dataset, batch_size=50, sampler=SubsetRandomSampler(indices))
    raise ValueError('Did not get small dataloader')
