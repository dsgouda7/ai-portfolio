"""Tests for Edge Deployment

Validates ONNX export, optimization, and edge device validation.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from src.edge import EdgeDeployer


class TestEdgeDeployer:
    """Test suite for EdgeDeployer."""
    
    def test_initialization(self, test_config):
        """Test EdgeDeployer initialization."""
        deployer = EdgeDeployer(test_config)
        
        assert deployer.config == test_config
        assert deployer.target_platform == test_config['edge_deployment']['target_platform']
    
    def test_export_to_onnx(self, edge_deployer, mock_model, tmp_path):
        """Test model export to ONNX."""
        output_path = tmp_path / "model.onnx"
        
        onnx_path = edge_deployer.export_to_onnx(
            mock_model,
            str(output_path),
            input_size=(640, 640)
        )
        
        assert Path(onnx_path).exists()
        assert Path(onnx_path).suffix == '.onnx'
        
        # Validate ONNX model
        from src.utils import validate_onnx_model
        is_valid = validate_onnx_model(onnx_path)
        assert is_valid
    
    def test_optimize_onnx(self, edge_deployer, mock_model, tmp_path):
        """Test ONNX model optimization."""
        # First export model
        onnx_path = tmp_path / "model.onnx"
        edge_deployer.export_to_onnx(mock_model, str(onnx_path))
        
        # Optimize
        optimized_path = edge_deployer.optimize_onnx(str(onnx_path))
        
        assert Path(optimized_path).exists()
        
        # Validate optimized model
        from src.utils import validate_onnx_model
        is_valid = validate_onnx_model(optimized_path)
        assert is_valid
    
    def test_validate_onnx_inference(self, edge_deployer, mock_model, tmp_path):
        """Test ONNX inference validation."""
        # Export model
        onnx_path = tmp_path / "model.onnx"
        edge_deployer.export_to_onnx(mock_model, str(onnx_path))
        
        # Validate inference
        is_valid = edge_deployer.validate_onnx_inference(
            str(onnx_path),
            pytorch_model=None,  # Skip PyTorch comparison
            input_size=(640, 640)
        )
        
        assert is_valid
    
    def test_benchmark_edge_device(self, edge_deployer, mock_model, tmp_path):
        """Test edge device benchmarking."""
        # Export model
        onnx_path = tmp_path / "model.onnx"
        edge_deployer.export_to_onnx(mock_model, str(onnx_path))
        
        # Benchmark
        metrics = edge_deployer.benchmark_edge_device(
            str(onnx_path),
            num_iterations=10
        )
        
        assert 'mean_latency_ms' in metrics
        assert 'median_latency_ms' in metrics
        assert 'throughput_fps' in metrics
        
        # Check values are reasonable
        assert metrics['mean_latency_ms'] > 0
        assert metrics['throughput_fps'] > 0
    
    def test_validate_jetson_nano(self, edge_deployer, mock_model, tmp_path):
        """Test Jetson Nano deployment validation."""
        # Export model
        onnx_path = tmp_path / "model.onnx"
        edge_deployer.export_to_onnx(mock_model, str(onnx_path))
        
        # Validate for Jetson Nano
        results = edge_deployer.validate_jetson_nano(str(onnx_path))
        
        assert 'model_size_mb' in results
        assert 'size_ok' in results
        assert 'onnx_valid' in results
        assert 'mean_latency_ms' in results
        assert 'latency_ok' in results
        assert 'deployment_ready' in results
        
        # Check types
        assert isinstance(results['size_ok'], bool)
        assert isinstance(results['onnx_valid'], bool)
        assert isinstance(results['deployment_ready'], bool)
    
    def test_deploy_full_pipeline(self, edge_deployer, mock_model, tmp_path):
        """Test full deployment pipeline."""
        # Update config to use tmp_path
        edge_deployer.onnx_dir = str(tmp_path)
        
        # Run deployment pipeline
        exported_models = edge_deployer.deploy_full_pipeline(
            mock_model,
            model_name="test_model"
        )
        
        assert 'onnx' in exported_models
        assert 'onnx_optimized' in exported_models
        
        # Check files exist
        assert Path(exported_models['onnx']).exists()
        assert Path(exported_models['onnx_optimized']).exists()


class TestONNXExport:
    """Test suite for ONNX export utilities."""
    
    def test_onnx_validation(self, mock_model, tmp_path):
        """Test ONNX model validation."""
        from src.utils import validate_onnx_model
        
        # Export model
        onnx_path = tmp_path / "model.onnx"
        
        dummy_input = torch.randn(1, 3, 640, 640)
        torch.onnx.export(
            mock_model,
            dummy_input,
            str(onnx_path),
            opset_version=11,
            input_names=['input'],
            output_names=['output']
        )
        
        # Validate
        is_valid = validate_onnx_model(str(onnx_path))
        assert is_valid
    
    def test_onnx_session_creation(self, mock_model, tmp_path):
        """Test ONNX Runtime session creation."""
        from src.utils import create_onnx_session
        
        # Export model
        onnx_path = tmp_path / "model.onnx"
        
        dummy_input = torch.randn(1, 3, 640, 640)
        torch.onnx.export(
            mock_model,
            dummy_input,
            str(onnx_path),
            opset_version=11,
            input_names=['input'],
            output_names=['output']
        )
        
        # Create session
        session = create_onnx_session(str(onnx_path))
        
        assert session is not None
        
        # Test inference
        input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
        outputs = session.run(None, {'input': input_data})
        
        assert outputs is not None
        assert len(outputs) > 0
    
    def test_model_size_check(self, mock_model, tmp_path):
        """Test model size calculation."""
        from src.utils import get_model_size_mb
        
        # Save model
        model_path = tmp_path / "model.pth"
        torch.save(mock_model.state_dict(), str(model_path))
        
        # Get size
        size_mb = get_model_size_mb(str(model_path))
        
        assert size_mb > 0
        assert isinstance(size_mb, float)
