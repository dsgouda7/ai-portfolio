"""Tests for Data Loading and Augmentation

Validates COCO dataset loading, augmentation pipeline, and data loaders.
"""

import pytest
import numpy as np
import torch
from src.data import COCODataLoader, COCODataset


class TestCOCODataLoader:
    """Test suite for COCODataLoader."""
    
    def test_initialization(self, test_config):
        """Test data loader initialization."""
        loader = COCODataLoader(test_config)
        
        assert loader.config == test_config
        assert loader.num_images == test_config['dataset']['num_images']
        assert loader.train_split == test_config['dataset']['train_split']
    
    def test_load_and_split(self, test_config):
        """Test dataset loading and splitting."""
        loader = COCODataLoader(test_config)
        train_ds, val_ds, test_ds = loader.load_and_split()
        
        # Check splits exist
        assert train_ds is not None
        assert val_ds is not None
        assert test_ds is not None
        
        # Check split sizes
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == test_config['dataset']['num_images']
        
        # Check train split is largest
        assert len(train_ds) > len(val_ds)
        assert len(train_ds) > len(test_ds)
    
    def test_train_transforms(self, test_config):
        """Test training augmentation pipeline."""
        loader = COCODataLoader(test_config)
        transform = loader._get_train_transforms()
        
        # Test transform on dummy data
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = [[100, 100, 50, 50]]
        labels = [1]
        
        transformed = transform(image=image, bboxes=boxes, labels=labels)
        
        assert 'image' in transformed
        assert 'bboxes' in transformed
        assert 'labels' in transformed
        
        # Check output is tensor
        assert isinstance(transformed['image'], torch.Tensor)
    
    def test_val_transforms(self, test_config):
        """Test validation augmentation pipeline."""
        loader = COCODataLoader(test_config)
        transform = loader._get_val_transforms()
        
        # Test transform on dummy data
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = [[100, 100, 50, 50]]
        labels = [1]
        
        transformed = transform(image=image, bboxes=boxes, labels=labels)
        
        assert 'image' in transformed
        assert isinstance(transformed['image'], torch.Tensor)
    
    def test_create_dataloaders(self, test_config):
        """Test DataLoader creation."""
        loader = COCODataLoader(test_config)
        train_ds, val_ds, test_ds = loader.load_and_split()
        
        train_loader, val_loader, test_loader = loader.create_dataloaders(
            train_ds, val_ds, test_ds
        )
        
        # Check loaders are created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Check batch size
        batch = next(iter(train_loader))
        assert len(batch) == test_config['training']['batch_size']


class TestCOCODataset:
    """Test suite for COCODataset."""
    
    def test_initialization(self, test_config):
        """Test dataset initialization."""
        dataset = COCODataset(
            data_dir='data/coco',
            split='train',
            num_samples=100,
            config=test_config
        )
        
        assert len(dataset) == 100
        assert dataset.split == 'train'
    
    def test_getitem(self, test_config):
        """Test dataset __getitem__ method."""
        dataset = COCODataset(
            data_dir='data/coco',
            split='train',
            num_samples=10,
            config=test_config
        )
        
        sample = dataset[0]
        
        # Check required keys
        assert 'image' in sample
        assert 'boxes' in sample
        assert 'labels' in sample
        assert 'image_id' in sample
        
        # Check types
        assert isinstance(sample['image'], torch.Tensor)
        assert isinstance(sample['boxes'], torch.Tensor)
        assert isinstance(sample['labels'], torch.Tensor)
        
        # Check shapes
        assert sample['image'].shape[0] == 3  # 3 channels
        assert sample['boxes'].shape[1] == 4  # x, y, w, h
        assert len(sample['labels']) == len(sample['boxes'])
    
    def test_dataset_length(self, test_config):
        """Test dataset length."""
        num_samples = 50
        dataset = COCODataset(
            data_dir='data/coco',
            split='train',
            num_samples=num_samples,
            config=test_config
        )
        
        assert len(dataset) == num_samples
    
    def test_augmentation_applied(self, test_config):
        """Test that augmentation is applied when provided."""
        from albumentations import Compose, HorizontalFlip
        from albumentations.pytorch import ToTensorV2
        
        transform = Compose([
            HorizontalFlip(p=1.0),  # Always flip
            ToTensorV2()
        ])
        
        dataset = COCODataset(
            data_dir='data/coco',
            split='train',
            num_samples=10,
            transform=transform,
            config=test_config
        )
        
        sample = dataset[0]
        
        # Check transform was applied (image is tensor)
        assert isinstance(sample['image'], torch.Tensor)
