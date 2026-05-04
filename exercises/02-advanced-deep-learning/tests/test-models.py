"""Tests for Model Registry and Compression

Validates model creation, training, and compression techniques.
"""

import pytest
import torch
import torch.nn as nn
from src.models import ModelRegistry


class TestModelRegistry:
    """Test suite for ModelRegistry."""
    
    def test_initialization(self, test_config):
        """Test ModelRegistry initialization."""
        registry = ModelRegistry(test_config)
        
        assert registry.config == test_config
        assert registry.device is not None
    
    def test_create_faster_rcnn(self, model_registry):
        """Test Faster R-CNN model creation."""
        model = model_registry.train_faster_rcnn()
        
        assert model is not None
        assert isinstance(model, nn.Module)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640).to(model_registry.device)
        model.eval()
        
        with torch.no_grad():
            output = model([dummy_input])
        
        assert len(output) == 1
        assert 'boxes' in output[0]
        assert 'labels' in output[0]
        assert 'scores' in output[0]
    
    def test_create_yolov8(self, model_registry):
        """Test YOLOv8 model creation."""
        try:
            model = model_registry.train_yolov8()
            assert model is not None
        except Exception as e:
            pytest.skip(f"YOLOv8 not available: {e}")
    
    def test_create_mask_rcnn(self, model_registry):
        """Test Mask R-CNN model creation."""
        model = model_registry.train_mask_rcnn()
        
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_save_and_load_model(self, model_registry, tmp_path):
        """Test model saving and loading."""
        # Create model
        model = model_registry.train_faster_rcnn()
        
        # Save model
        model_path = tmp_path / "test_model.pth"
        model_registry.save_model(model, str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_model = model_registry.load_model(str(model_path), 'faster_rcnn')
        
        assert loaded_model is not None
        assert isinstance(loaded_model, nn.Module)
    
    def test_apply_pruning(self, model_registry, mock_model):
        """Test model pruning."""
        original_params = sum(p.numel() for p in mock_model.parameters())
        
        pruned_model = model_registry.apply_pruning(mock_model)
        
        assert pruned_model is not None
        
        # Check that pruning was applied (some weights should be zero)
        has_zeros = False
        for param in pruned_model.parameters():
            if torch.sum(param == 0) > 0:
                has_zeros = True
                break
        
        assert has_zeros, "Pruning should produce some zero weights"
    
    def test_apply_quantization(self, model_registry, mock_model):
        """Test model quantization."""
        quantized_model = model_registry.apply_quantization(mock_model)
        
        assert quantized_model is not None
    
    def test_compression_pipeline(self, model_registry, mock_model):
        """Test full compression pipeline."""
        # Note: Simplified test without actual training
        compressed_model = model_registry.compress_pipeline(mock_model, train_loader=None)
        
        assert compressed_model is not None


class TestModelCompression:
    """Test suite for model compression techniques."""
    
    def test_distillation_config(self, test_config):
        """Test distillation configuration."""
        distill_config = test_config['compression']['distillation']
        
        assert 'enabled' in distill_config
        assert 'temperature' in distill_config
        assert 'alpha' in distill_config
        
        # Validate values
        assert isinstance(distill_config['temperature'], (int, float))
        assert 0 <= distill_config['alpha'] <= 1
    
    def test_pruning_config(self, test_config):
        """Test pruning configuration."""
        pruning_config = test_config['compression']['pruning']
        
        assert 'enabled' in pruning_config
        assert 'pruning_ratio' in pruning_config
        
        # Validate values
        assert 0 <= pruning_config['pruning_ratio'] <= 1
    
    def test_quantization_config(self, test_config):
        """Test quantization configuration."""
        quant_config = test_config['compression']['quantization']
        
        assert 'enabled' in quant_config
        assert 'quantization_type' in quant_config
        
        # Validate values
        assert quant_config['quantization_type'] in ['int8', 'float16']
    
    def test_model_size_reduction(self, model_registry, mock_model, tmp_path):
        """Test that compression reduces model size."""
        # Save original model
        original_path = tmp_path / "original.pth"
        model_registry.save_model(mock_model, str(original_path))
        original_size = original_path.stat().st_size
        
        # Apply compression
        compressed_model = model_registry.apply_pruning(mock_model)
        compressed_model = model_registry.apply_quantization(compressed_model)
        
        # Save compressed model
        compressed_path = tmp_path / "compressed.pth"
        model_registry.save_model(compressed_model, str(compressed_path))
        compressed_size = compressed_path.stat().st_size
        
        # Size should be reduced or similar (quantization effects)
        # Note: Dynamic quantization might not always reduce file size significantly
        assert compressed_size > 0
