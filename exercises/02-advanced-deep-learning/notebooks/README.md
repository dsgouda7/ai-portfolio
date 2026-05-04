# ProductionCV Notebooks

This directory contains exploratory Jupyter notebooks for:

## Available Notebooks

### 1. `01_data_exploration.ipynb`
- Load and visualize COCO dataset
- Explore augmentation effects
- Analyze label distributions
- Visualize bounding boxes and masks

### 2. `02_model_training.ipynb`
- Train baseline Faster R-CNN
- Experiment with different architectures
- Visualize training curves
- Compare model performance

### 3. `03_compression_analysis.ipynb`
- Apply compression techniques
- Analyze accuracy vs. size tradeoffs
- Benchmark inference latency
- Compare distillation strategies

### 4. `04_edge_deployment.ipynb`
- Export models to ONNX
- Test TFLite conversion
- Validate edge device performance
- Optimize for inference

### 5. `05_api_testing.ipynb`
- Test API endpoints
- Visualize detection results
- Benchmark API latency
- Stress test with multiple requests

## Usage

Start Jupyter Lab:
```bash
jupyter lab
```

Or use VS Code with Jupyter extension.

## Note

Notebooks are for exploration only. Production code should be in `src/` with proper tests.
