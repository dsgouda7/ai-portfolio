"""Data Loading and Augmentation for COCO Dataset

Handles COCO dataset loading, train/val/test splits, and augmentation pipeline.
"""

import os
import json
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import setup_logger, ensure_dir


logger = setup_logger()


class COCODataLoader:
    """
    COCO dataset loader with support for object detection and instance segmentation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize COCO data loader.
        
        Args:
            config: Configuration dictionary with dataset parameters
        """
        self.config = config
        self.data_dir = config['dataset']['data_dir']
        self.num_images = config['dataset']['num_images']
        self.train_split = config['dataset']['train_split']
        self.val_split = config['dataset']['val_split']
        self.test_split = config['dataset']['test_split']
        
        ensure_dir(self.data_dir)
        logger.info(f"Initialized COCO data loader for {self.data_dir}")
    
    def load_and_split(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load COCO dataset and split into train/val/test.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Loading COCO dataset...")
        
        # In production, this would download/load actual COCO data
        # For now, we create a placeholder structure
        
        # Calculate split sizes
        n_train = int(self.num_images * self.train_split)
        n_val = int(self.num_images * self.val_split)
        n_test = self.num_images - n_train - n_val
        
        # Create augmentation pipelines
        train_transform = self._get_train_transforms()
        val_transform = self._get_val_transforms()
        
        # Create dataset instances
        train_dataset = COCODataset(
            data_dir=self.data_dir,
            split='train',
            num_samples=n_train,
            transform=train_transform,
            config=self.config
        )
        
        val_dataset = COCODataset(
            data_dir=self.data_dir,
            split='val',
            num_samples=n_val,
            transform=val_transform,
            config=self.config
        )
        
        test_dataset = COCODataset(
            data_dir=self.data_dir,
            split='test',
            num_samples=n_test,
            transform=val_transform,
            config=self.config
        )
        
        logger.info(f"Dataset splits - Train: {n_train}, Val: {n_val}, Test: {n_test}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _get_train_transforms(self) -> A.Compose:
        """Create training augmentation pipeline."""
        aug_config = self.config['dataset']['augmentation']
        
        return A.Compose([
            A.HorizontalFlip(p=aug_config['horizontal_flip']),
            A.VerticalFlip(p=aug_config['vertical_flip']),
            A.Rotate(limit=aug_config['rotation_range'], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config['brightness_range'],
                contrast_limit=aug_config['contrast_range'],
                p=0.5
            ),
            A.Resize(*self.config['preprocessing']['image_size']),
            A.Normalize(
                mean=self.config['preprocessing']['normalize_mean'],
                std=self.config['preprocessing']['normalize_std']
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    
    def _get_val_transforms(self) -> A.Compose:
        """Create validation/test augmentation pipeline."""
        return A.Compose([
            A.Resize(*self.config['preprocessing']['image_size']),
            A.Normalize(
                mean=self.config['preprocessing']['normalize_mean'],
                std=self.config['preprocessing']['normalize_std']
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    
    def create_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for train/val/test.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            batch_size: Batch size (uses config if None)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if batch_size is None:
            batch_size = self.config['training']['batch_size']
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader, test_loader


class COCODataset(Dataset):
    """
    PyTorch Dataset for COCO format data.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str,
        num_samples: int,
        transform: Optional[A.Compose] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize COCO dataset.
        
        Args:
            data_dir: Root directory of COCO data
            split: Dataset split ('train', 'val', or 'test')
            num_samples: Number of samples in this split
            transform: Albumentations transform pipeline
            config: Configuration dictionary
        """
        self.data_dir = data_dir
        self.split = split
        self.num_samples = num_samples
        self.transform = transform
        self.config = config or {}
        
        # In production, load actual annotations
        # For now, generate synthetic metadata
        self.image_ids = list(range(num_samples))
        
        logger.info(f"Created {split} dataset with {num_samples} samples")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'image', 'boxes', 'labels', 'masks' (if applicable)
        """
        # In production, load actual image and annotations
        # For now, create synthetic data for testing
        
        image_size = self.config.get('preprocessing', {}).get('image_size', [640, 640])
        image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        
        # Generate random bounding boxes
        num_boxes = np.random.randint(1, 5)
        boxes = []
        labels = []
        
        for _ in range(num_boxes):
            x = np.random.randint(0, image_size[1] - 100)
            y = np.random.randint(0, image_size[0] - 100)
            w = np.random.randint(50, min(100, image_size[1] - x))
            h = np.random.randint(50, min(100, image_size[0] - y))
            
            boxes.append([x, y, w, h])
            labels.append(np.random.randint(0, 80))  # COCO has 80 classes
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        return {
            'image': image,
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }


def collate_fn(batch: List[Dict]) -> List[Dict]:
    """
    Custom collate function for object detection.
    
    Args:
        batch: List of samples
        
    Returns:
        Collated batch
    """
    return batch
