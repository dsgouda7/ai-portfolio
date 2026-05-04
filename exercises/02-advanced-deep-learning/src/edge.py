"""Edge Deployment Utilities

Handles ONNX export, TensorRT optimization, and edge device validation.
"""

import os
import torch
import torch.onnx
import onnx
import onnxruntime as ort
from typing import Dict, Tuple, Optional
import numpy as np
from src.utils import setup_logger, timing_decorator, ensure_dir, validate_onnx_model


logger = setup_logger()


class EdgeDeployer:
    """
    Edge deployment toolkit for computer vision models.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize edge deployer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.edge_config = config['edge_deployment']
        self.onnx_dir = config['paths']['onnx_dir']
        self.target_platform = self.edge_config['target_platform']
        
        ensure_dir(self.onnx_dir)
        
        logger.info(f"Initialized EdgeDeployer for platform: {self.target_platform}")
    
    @timing_decorator
    def export_to_onnx(
        self,
        model: torch.nn.Module,
        output_path: str,
        input_size: Tuple[int, int] = (640, 640),
        opset_version: int = 11
    ) -> str:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            output_path: Output ONNX file path
            input_size: Input image size (height, width)
            opset_version: ONNX opset version
            
        Returns:
            Path to exported ONNX model
        """
        logger.info(f"Exporting model to ONNX (opset {opset_version})...")
        
        model.eval()
        device = next(model.parameters()).device
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, *input_size).to(device)
        
        # Export to ONNX
        ensure_dir(os.path.dirname(output_path))
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Validate exported model
        is_valid = validate_onnx_model(output_path)
        
        if is_valid:
            logger.info(f"Successfully exported ONNX model to {output_path}")
        else:
            logger.error(f"ONNX export validation failed!")
        
        return output_path
    
    @timing_decorator
    def optimize_onnx(
        self,
        onnx_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Optimize ONNX model for inference.
        
        Args:
            onnx_path: Input ONNX model path
            output_path: Output path for optimized model
            
        Returns:
            Path to optimized ONNX model
        """
        logger.info("Optimizing ONNX model...")
        
        if output_path is None:
            base, ext = os.path.splitext(onnx_path)
            output_path = f"{base}_optimized{ext}"
        
        # Load model
        model = onnx.load(onnx_path)
        
        # Apply optimizations
        from onnx import optimizer
        
        passes = [
            'eliminate_identity',
            'eliminate_nop_transpose',
            'eliminate_nop_pad',
            'fuse_consecutive_transposes',
            'fuse_transpose_into_gemm',
            'fuse_add_bias_into_conv',
            'fuse_bn_into_conv'
        ]
        
        optimized_model = optimizer.optimize(model, passes)
        
        # Save optimized model
        onnx.save(optimized_model, output_path)
        
        logger.info(f"Optimized ONNX model saved to {output_path}")
        
        return output_path
    
    def validate_onnx_inference(
        self,
        onnx_path: str,
        pytorch_model: Optional[torch.nn.Module] = None,
        input_size: Tuple[int, int] = (640, 640),
        rtol: float = 1e-3,
        atol: float = 1e-5
    ) -> bool:
        """
        Validate ONNX model inference against PyTorch model.
        
        Args:
            onnx_path: Path to ONNX model
            pytorch_model: Original PyTorch model for comparison
            input_size: Input image size
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            True if validation passes
        """
        logger.info("Validating ONNX inference...")
        
        # Create ONNX session
        session = ort.InferenceSession(onnx_path)
        
        # Create test input
        test_input = np.random.randn(1, 3, *input_size).astype(np.float32)
        
        # ONNX inference
        onnx_output = session.run(None, {'input': test_input})[0]
        
        # Compare with PyTorch if model provided
        if pytorch_model is not None:
            pytorch_model.eval()
            device = next(pytorch_model.parameters()).device
            
            with torch.no_grad():
                pytorch_input = torch.from_numpy(test_input).to(device)
                pytorch_output = pytorch_model(pytorch_input).cpu().numpy()
            
            # Check if outputs match
            if np.allclose(onnx_output, pytorch_output, rtol=rtol, atol=atol):
                logger.info("✓ ONNX inference validation passed!")
                return True
            else:
                max_diff = np.max(np.abs(onnx_output - pytorch_output))
                logger.error(f"✗ ONNX inference validation failed! Max diff: {max_diff}")
                return False
        else:
            logger.info("✓ ONNX inference runs successfully (no PyTorch comparison)")
            return True
    
    def export_to_tflite(
        self,
        onnx_path: str,
        output_path: str
    ) -> str:
        """
        Convert ONNX model to TensorFlow Lite format.
        
        Args:
            onnx_path: Input ONNX model path
            output_path: Output TFLite model path
            
        Returns:
            Path to TFLite model
        """
        logger.info("Converting ONNX to TensorFlow Lite...")
        
        try:
            import onnx_tf
            import tensorflow as tf
            
            # Convert ONNX to TensorFlow
            onnx_model = onnx.load(onnx_path)
            tf_rep = onnx_tf.backend.prepare(onnx_model)
            
            # Export to SavedModel format
            tf_model_dir = output_path.replace('.tflite', '_tf')
            tf_rep.export_graph(tf_model_dir)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            # Save TFLite model
            ensure_dir(os.path.dirname(output_path))
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"TFLite model saved to {output_path}")
            
            return output_path
            
        except ImportError as e:
            logger.error(f"TFLite conversion failed: {e}")
            logger.error("Install onnx-tf and tensorflow to enable TFLite export")
            return None
    
    def benchmark_edge_device(
        self,
        model_path: str,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark model on edge device.
        
        Args:
            model_path: Path to ONNX model
            num_iterations: Number of benchmark iterations
            
        Returns:
            Benchmark metrics
        """
        logger.info(f"Benchmarking on {self.target_platform}...")
        
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_input})
        
        # Benchmark
        import time
        latencies = []
        
        for _ in range(num_iterations):
            start = time.time()
            _ = session.run(None, {input_name: dummy_input})
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        metrics = {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'std_latency_ms': np.std(latencies),
            'throughput_fps': 1000 / np.mean(latencies)
        }
        
        logger.info(f"Edge device latency: {metrics['mean_latency_ms']:.2f}ms "
                   f"({metrics['throughput_fps']:.1f} FPS)")
        
        return metrics
    
    def validate_jetson_nano(
        self,
        onnx_path: str
    ) -> Dict[str, any]:
        """
        Validate model for Jetson Nano deployment.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Validation results
        """
        logger.info("Validating for Jetson Nano deployment...")
        
        from src.utils import get_model_size_mb
        
        # Check model size
        model_size_mb = get_model_size_mb(onnx_path)
        size_ok = model_size_mb <= self.config['model']['target_size_mb']
        
        # Validate ONNX
        is_valid = validate_onnx_model(onnx_path)
        
        # Benchmark
        benchmark_metrics = self.benchmark_edge_device(onnx_path)
        latency_ok = benchmark_metrics['mean_latency_ms'] <= self.config['model']['target_inference_ms']
        
        results = {
            'model_size_mb': model_size_mb,
            'size_ok': size_ok,
            'onnx_valid': is_valid,
            'mean_latency_ms': benchmark_metrics['mean_latency_ms'],
            'latency_ok': latency_ok,
            'deployment_ready': size_ok and is_valid and latency_ok
        }
        
        logger.info("=" * 60)
        logger.info("Jetson Nano Validation Results:")
        logger.info(f"  Model Size: {model_size_mb:.2f} MB {'✓' if size_ok else '✗'}")
        logger.info(f"  ONNX Valid: {'✓' if is_valid else '✗'}")
        logger.info(f"  Latency: {benchmark_metrics['mean_latency_ms']:.2f}ms {'✓' if latency_ok else '✗'}")
        logger.info(f"  Deployment Ready: {'✓' if results['deployment_ready'] else '✗'}")
        logger.info("=" * 60)
        
        return results
    
    def deploy_full_pipeline(
        self,
        pytorch_model: torch.nn.Module,
        model_name: str = "model"
    ) -> Dict[str, str]:
        """
        Full deployment pipeline: PyTorch → ONNX → Optimize → Validate.
        
        Args:
            pytorch_model: PyTorch model to deploy
            model_name: Base name for exported models
            
        Returns:
            Dictionary of exported model paths
        """
        logger.info("Starting full deployment pipeline...")
        
        exported_models = {}
        
        # 1. Export to ONNX
        onnx_path = os.path.join(self.onnx_dir, f"{model_name}.onnx")
        onnx_path = self.export_to_onnx(pytorch_model, onnx_path)
        exported_models['onnx'] = onnx_path
        
        # 2. Optimize ONNX
        optimized_path = self.optimize_onnx(onnx_path)
        exported_models['onnx_optimized'] = optimized_path
        
        # 3. Validate ONNX inference
        self.validate_onnx_inference(optimized_path, pytorch_model)
        
        # 4. Export to TFLite (if configured)
        if 'tflite' in self.edge_config['export_formats']:
            tflite_path = os.path.join(self.onnx_dir, f"{model_name}.tflite")
            tflite_path = self.export_to_tflite(optimized_path, tflite_path)
            if tflite_path:
                exported_models['tflite'] = tflite_path
        
        # 5. Validate for target platform
        if self.target_platform == 'jetson_nano':
            self.validate_jetson_nano(optimized_path)
        
        logger.info("Deployment pipeline completed!")
        logger.info(f"Exported models: {list(exported_models.keys())}")
        
        return exported_models


if __name__ == "__main__":
    import argparse
    from src.utils import load_config
    from src.models import ModelRegistry
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    deployer = EdgeDeployer(config)
    
    if args.export:
        # Load PyTorch model
        registry = ModelRegistry(config)
        model = registry.load_model(args.model_path)
        
        # Run full deployment pipeline
        exported = deployer.deploy_full_pipeline(model, "productioncv")
        
        print("\nExported models:")
        for format_name, path in exported.items():
            print(f"  {format_name}: {path}")
