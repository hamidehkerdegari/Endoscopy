import torch
from torch.utils.data import DataLoader
from .dataset import EndoscopyDataset

def get_dataloader(root_dir, augment_dir, batch_size=16, num_workers=4, shuffle=True, split_ratio=0.8, transform=None, n_folds=2, epoch=0):
    dataset = EndoscopyDataset(root_dir=root_dir, augment_dir=augment_dir, transform=transform, n_folds=n_folds, epoch=epoch)
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_dataset
