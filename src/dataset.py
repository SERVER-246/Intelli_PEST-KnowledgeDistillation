"""
Dataset Loader for Knowledge Distillation
==========================================
Handles loading and preprocessing of pest classification dataset
with proper class mapping and augmentation.

Author: Knowledge Distillation Pipeline
Date: 2024-12-23
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class PestDataset(Dataset):
    """
    Custom Dataset for Pest Classification.
    
    Supports:
    - ImageFolder-style directory structure
    - Proper class mapping with logging
    - Data augmentation
    - Weighted sampling for imbalanced classes
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        class_mapping: Optional[Dict[str, int]] = None,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing class subfolders
            transform: Torchvision transforms to apply
            class_mapping: Optional predefined class mapping
            split: 'train' or 'val'
            train_ratio: Ratio for train/val split
            seed: Random seed for reproducibility
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        
        # Discover classes and create mapping
        self.class_names = sorted([
            d.name for d in self.root_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        if class_mapping:
            self.class_to_idx = class_mapping
        else:
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_names)
        
        # Load all image paths
        self.samples = self._load_samples()
        
        # Split into train/val
        self._split_data()
        
        # Calculate class weights for balanced sampling
        self.class_weights = self._calculate_class_weights()
        
        # Log dataset info
        self._log_dataset_info()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and labels."""
        samples = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    samples.append((str(img_path), class_idx))
        
        return samples
    
    def _split_data(self):
        """Split data into train and validation sets."""
        np.random.seed(self.seed)
        
        # Group samples by class for stratified split
        class_samples = {}
        for path, label in self.samples:
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append((path, label))
        
        train_samples = []
        val_samples = []
        
        for label, samples in class_samples.items():
            np.random.shuffle(samples)
            split_idx = int(len(samples) * self.train_ratio)
            train_samples.extend(samples[:split_idx])
            val_samples.extend(samples[split_idx:])
        
        if self.split == "train":
            self.samples = train_samples
        else:
            self.samples = val_samples
        
        # Shuffle the samples
        np.random.shuffle(self.samples)
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced sampling."""
        class_counts = Counter(label for _, label in self.samples)
        total = sum(class_counts.values())
        
        weights = torch.zeros(self.num_classes)
        for class_idx, count in class_counts.items():
            weights[class_idx] = total / (self.num_classes * count)
        
        return weights
    
    def _log_dataset_info(self):
        """Log dataset statistics."""
        class_counts = Counter(label for _, label in self.samples)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {self.split.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"Root directory: {self.root_dir}")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Total samples: {len(self.samples)}")
        logger.info(f"\nClass Distribution:")
        logger.info(f"{'-'*40}")
        
        for idx in sorted(class_counts.keys()):
            class_name = self.idx_to_class[idx]
            count = class_counts[idx]
            percentage = count / len(self.samples) * 100
            logger.info(f"  [{idx:2d}] {class_name:20s}: {count:5d} ({percentage:5.1f}%)")
        
        logger.info(f"{'='*60}\n")
    
    def get_sample_weights(self) -> List[float]:
        """Get sample weights for WeightedRandomSampler."""
        return [self.class_weights[label].item() for _, label in self.samples]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(
    image_size: int = 256,
    is_training: bool = True,
    normalize_mean: List[float] = [0.485, 0.456, 0.406],
    normalize_std: List[float] = [0.229, 0.224, 0.225]
) -> transforms.Compose:
    """
    Get data transforms for training or validation.
    
    Args:
        image_size: Target image size
        is_training: Whether to use training augmentations
        normalize_mean: Normalization mean values
        normalize_std: Normalization std values
        
    Returns:
        Composed transforms
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
            transforms.RandomErasing(p=0.1)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
    
    return transform


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 256,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    use_weighted_sampling: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loading workers
        train_ratio: Train/val split ratio
        use_weighted_sampling: Whether to use weighted sampling for imbalanced classes
        seed: Random seed
        
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        dataset_info: Dictionary with dataset information
    """
    # Create datasets
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)
    
    train_dataset = PestDataset(
        root_dir=data_dir,
        transform=train_transform,
        split="train",
        train_ratio=train_ratio,
        seed=seed
    )
    
    val_dataset = PestDataset(
        root_dir=data_dir,
        transform=val_transform,
        split="val",
        train_ratio=train_ratio,
        seed=seed
    )
    
    # Create samplers
    if use_weighted_sampling:
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.get_sample_weights(),
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Compile dataset info
    dataset_info = {
        "num_classes": train_dataset.num_classes,
        "class_names": train_dataset.class_names,
        "class_to_idx": train_dataset.class_to_idx,
        "idx_to_class": train_dataset.idx_to_class,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "class_weights": train_dataset.class_weights.tolist()
    }
    
    return train_loader, val_loader, dataset_info


def save_class_mapping(
    class_mapping: Dict[str, int],
    save_path: str
):
    """Save class mapping to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    logger.info(f"Class mapping saved to {save_path}")


def load_class_mapping(load_path: str) -> Dict[str, int]:
    """Load class mapping from JSON file."""
    with open(load_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test the dataset
    logging.basicConfig(level=logging.INFO)
    
    data_dir = r"G:\AI work\IMAGE DATASET"
    
    train_loader, val_loader, info = create_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        image_size=256,
        num_workers=0,  # Use 0 for testing
        use_weighted_sampling=True
    )
    
    print(f"\nDataset Info:")
    print(f"  Classes: {info['num_classes']}")
    print(f"  Train samples: {info['train_samples']}")
    print(f"  Val samples: {info['val_samples']}")
    
    # Test batch loading
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label values: {labels[:10]}")
