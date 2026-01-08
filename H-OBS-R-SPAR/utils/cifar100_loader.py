"""
CIFAR-100 Dataloader

Provides CIFAR-100 dataset loading and preprocessing.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple


def get_cifar100_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-100 train and test data loaders.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
    
    Returns:
        (train_loader, test_loader)
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    # Transform for testing
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    print("=== CIFAR-100 Dataloader Test ===\n")
    
    train_loader, test_loader = get_cifar100_loaders(
        data_dir='./data',
        batch_size=128
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Number of classes: 100")
    print(f"Image shape: (3, 32, 32)")
    
    # Test loading a batch
    for  images, labels in train_loader:
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break
    
    print("\nCIFAR-100 dataloader test completed!")
