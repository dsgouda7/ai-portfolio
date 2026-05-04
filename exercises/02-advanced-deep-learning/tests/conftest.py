"""Test Configuration and Fixtures

Provides shared fixtures for testing ProductionCV components.
"""

import pytest
import numpy as np
import torch
from typing import Dict
from src.utils import load_config


@pytest.fixture
def test_config() -> Dict:
    """
    Load test configuration.
    
    Returns:
        Configuration dictionary
    """
    config = load_config('config.yaml')
    
    # Override with test-specific settings
    config['dataset']['num_images'] = 100
    config['training']['epochs'] = 2
    config['training']['batch_size'] = 4
    
    return config


@pytest.fixture
def dummy_image() -> np.ndarray:
    """
    Create a dummy image for testing.
    
    Returns:
        Random image array (640x640x3)
    """
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_batch() -> list:
    """
    Create a batch of dummy images for testing.
    
    Returns:
        List of sample dictionaries
    """
    batch = []
    
    for i in range(4):
        image = torch.randn(3, 640, 640)
        boxes = torch.tensor([[50, 50, 100, 100], [200, 200, 100, 150]], dtype=torch.float32)
        labels = torch.tensor([1, 2], dtype=torch.int64)
        
        batch.append({
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([i])
        })
    
    return batch


@pytest.fixture
def model_registry(test_config):
    """
    Create ModelRegistry instance for testing.
    
    Args:
        test_config: Test configuration
        
    Returns:
        ModelRegistry instance
    """
    from src.models import ModelRegistry
    return ModelRegistry(test_config)


@pytest.fixture
def image_preprocessor(test_config):
    """
    Create ImagePreprocessor instance for testing.
    
    Args:
        test_config: Test configuration
        
    Returns:
        ImagePreprocessor instance
    """
    from src.features import ImagePreprocessor
    return ImagePreprocessor(test_config)


@pytest.fixture
def evaluator(test_config):
    """
    Create CVEvaluator instance for testing.
    
    Args:
        test_config: Test configuration
        
    Returns:
        CVEvaluator instance
    """
    from src.evaluate import CVEvaluator
    return CVEvaluator(test_config)


@pytest.fixture
def edge_deployer(test_config):
    """
    Create EdgeDeployer instance for testing.
    
    Args:
        test_config: Test configuration
        
    Returns:
        EdgeDeployer instance
    """
    from src.edge import EdgeDeployer
    return EdgeDeployer(test_config)


@pytest.fixture
def mock_model():
    """
    Create a mock detection model for testing.
    
    Returns:
        Simple mock model
    """
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        
        def forward(self, images, targets=None):
            # Return dummy predictions
            batch_size = len(images)
            predictions = []
            
            for i in range(batch_size):
                pred = {
                    'boxes': torch.tensor([[50, 50, 150, 150], [200, 200, 300, 350]], dtype=torch.float32),
                    'labels': torch.tensor([1, 2], dtype=torch.int64),
                    'scores': torch.tensor([0.9, 0.85], dtype=torch.float32)
                }
                predictions.append(pred)
            
            # If targets provided (training mode), return losses
            if targets is not None:
                losses = {
                    'loss_classifier': torch.tensor(0.5),
                    'loss_box_reg': torch.tensor(0.3),
                    'loss_objectness': torch.tensor(0.2),
                    'loss_rpn_box_reg': torch.tensor(0.1)
                }
                return losses
            
            return predictions
    
    return MockModel()
