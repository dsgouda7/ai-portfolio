# Solution Guide: ProductionCV

> **Note**: Attempt the exercise independently before reading this guide.

---

## Implementation Strategy

### Phase 1: Baseline Model (Days 1-2)

**Goal**: Get a working object detection pipeline.

1. **Start with data loading**:
   ```python
   from src.data import COCODataLoader
   
   loader = COCODataLoader(config)
   train_ds, val_ds, test_ds = loader.load_and_split()
   ```

2. **Train baseline Faster R-CNN**:
   ```python
   from src.models import ModelRegistry
   
   registry = ModelRegistry(config)
   model = registry.train_faster_rcnn()
   ```

3. **Validate it works**:
   ```bash
   make test
   pytest tests/test_models.py::TestModelRegistry::test_create_faster_rcnn
   ```

**Common Issues**:
- CUDA out of memory → Reduce batch size
- Slow training → Use pretrained model
- Missing dependencies → Check requirements.txt

---

### Phase 2: Compression Pipeline (Days 3-4)

**Goal**: Reduce model size while maintaining accuracy.

#### Knowledge Distillation

**Key insight**: Teacher model outputs soft probability distributions that contain more information than hard labels.

```python
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    # Soft targets from teacher
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_predictions = F.log_softmax(student_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean')
    soft_loss *= (temperature ** 2)
    
    # Hard targets (ground truth)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Combine
    return alpha * hard_loss + (1 - alpha) * soft_loss
```

**Best practices**:
- Temperature: 3-5 works well for vision tasks
- Alpha: 0.5 balances hard and soft targets
- Teacher should be larger than student

#### Pruning

**Strategy**: Remove weights with smallest L1 norms.

```python
import torch.nn.utils.prune as prune

for module in model.modules():
    if isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)
```

**Tips**:
- Start with 20-30% pruning ratio
- Fine-tune after pruning
- Use structured pruning for better speedup

#### Quantization

**Options**:
1. **Dynamic quantization**: Weights quantized, activations computed in FP32
2. **Static quantization**: Both weights and activations quantized (requires calibration)
3. **Quantization-aware training**: Simulate quantization during training

```python
# Dynamic quantization (easiest)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Conv2d, nn.Linear},
    dtype=torch.qint8
)
```

**Expected results**:
- Size reduction: 4x (FP32 → INT8)
- Speedup: 2-4x on CPU
- Accuracy drop: <2%

---

### Phase 3: Edge Deployment (Day 5)

**Goal**: Export model for edge devices.

#### ONNX Export

```python
dummy_input = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)
```

**Validation checklist**:
- [ ] ONNX model loads without errors
- [ ] Inference produces same results as PyTorch
- [ ] Model size <100 MB
- [ ] Inference <50ms on target device

#### TensorRT Optimization (Optional)

For Jetson Nano:

```python
import tensorrt as trt

# Build TensorRT engine
with trt.Builder(TRT_LOGGER) as builder:
    with builder.create_network() as network:
        with trt.OnnxParser(network, TRT_LOGGER) as parser:
            parser.parse_from_file('model.onnx')
            
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        engine = builder.build_engine(network, config)
```

---

### Phase 4: Production API (Day 6)

**Goal**: REST API with monitoring.

#### Flask API Structure

```python
@app.route('/detect', methods=['POST'])
@monitor_api_request
def detect_objects():
    # 1. Validate input
    if 'image' not in request.files:
        return error_response("No image provided")
    
    # 2. Load and preprocess
    image = load_image(request.files['image'])
    image_tensor = preprocessor.preprocess(image)
    
    # 3. Inference
    with torch.no_grad():
        predictions = model([image_tensor])
    
    # 4. Post-process
    detections = format_detections(predictions[0])
    
    # 5. Return response
    return jsonify({
        'success': True,
        'detections': detections,
        'inference_time_ms': inference_time
    })
```

**Best practices**:
- Validate input size (reject >10MB)
- Set confidence threshold (default 0.5)
- Limit max detections (default 100)
- Add request timeout
- Log all requests
- Monitor latency

---

## Optimization Tips

### Memory Optimization

1. **Use gradient checkpointing**:
   ```python
   from torch.utils.checkpoint import checkpoint
   ```

2. **Mixed precision training**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   with autocast():
       loss = model(images, targets)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

3. **Reduce batch size**: 32 → 16 → 8

### Speed Optimization

1. **Use DataLoader workers**: `num_workers=4`
2. **Pin memory**: `pin_memory=True`
3. **Prefetch**: Use CUDA streams
4. **Batch inference**: Process multiple images together

### Accuracy Optimization

1. **Data augmentation**:
   - Horizontal flip
   - Random crop
   - Color jitter
   - Mixup/CutMix

2. **Longer training**: 50-100 epochs
3. **Learning rate schedule**: Cosine annealing
4. **Ensemble**: Combine multiple models

---

## Common Pitfalls

### 1. Overfitting on Small Dataset

**Problem**: Model memorizes 1000 training images.

**Solutions**:
- Heavy data augmentation
- Dropout (p=0.5)
- Weight decay (1e-4)
- Early stopping

### 2. Slow Inference

**Problem**: Model takes >100ms per image.

**Solutions**:
- Use smaller backbone (ResNet-18)
- Reduce input size (640 → 416)
- Enable quantization
- Optimize ONNX model

### 3. Low mAP

**Problem**: mAP <70%.

**Solutions**:
- Train longer (100+ epochs)
- Use stronger augmentation
- Adjust anchor boxes for YOLO
- Fine-tune on full COCO

### 4. ONNX Export Fails

**Problem**: Unsupported operations.

**Solutions**:
- Use opset_version=11 or higher
- Avoid dynamic control flow
- Use supported operations only
- Export to TorchScript first

---

## Evaluation Metrics

### Mean Average Precision (mAP)

**Calculation**:
1. For each class, compute AP
2. Average APs across all classes

**Code**:
```python
def calculate_ap(recalls, precisions):
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) > 0:
            ap += np.max(precisions[recalls >= t])
    return ap / 11

# Calculate mAP
aps = []
for class_id in range(num_classes):
    ap = calculate_ap_for_class(class_id, predictions, targets)
    aps.append(ap)

mAP = np.mean(aps)
```

### Inference Latency

**Measurement**:
```python
latencies = []

for _ in range(100):
    start = time.time()
    with torch.no_grad():
        _ = model([image])
    torch.cuda.synchronize()  # Important!
    latency = time.time() - start
    latencies.append(latency)

mean_latency = np.mean(latencies) * 1000  # ms
```

---

## Success Criteria Validation

Before submitting:

```bash
# 1. All tests pass
make test

# 2. Model meets size constraint
python -c "from src.utils import get_model_size_mb; print(get_model_size_mb('models/compressed_model.pth'))"
# Should output <100

# 3. mAP meets target
python -m src.evaluate --model-path models/compressed_model.pth --benchmark
# Should show mAP ≥0.85

# 4. Latency meets target
# (Included in benchmark above)
# Should show mean_latency_ms <50

# 5. API works
make serve
curl -X POST http://localhost:5000/detect -F "image=@test.jpg"
# Should return detections

# 6. Docker builds
make docker-build
make docker-run
# Should start successfully
```

---

## Further Improvements

### Advanced Compression

1. **Neural Architecture Search (NAS)**
2. **Learned quantization** (LSQ, QAT)
3. **Filter pruning** (remove entire channels)
4. **Lottery ticket hypothesis** (find sparse subnetworks)

### Advanced Detection

1. **Anchor-free detectors** (FCOS, CenterNet)
2. **Transformer-based** (DETR, Deformable DETR)
3. **Self-supervised pretraining** (DINO, MAE)

### Production Features

1. **A/B testing** (compare model versions)
2. **Model versioning** (MLflow)
3. **Continuous deployment** (CI/CD pipeline)
4. **Auto-scaling** (Kubernetes HPA)

---

## Resources

- **Papers**:
  - Distilling the Knowledge in a Neural Network (Hinton et al., 2015)
  - Learning Efficient Convolutional Networks through Network Slimming (Liu et al., 2017)
  - Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (Jacob et al., 2018)

- **Tutorials**:
  - [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
  - [ONNX Runtime Optimization](https://onnxruntime.ai/docs/performance/quantization.html)
  - [TensorRT Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)

- **Tools**:
  - [Netron](https://netron.app/) - Visualize models
  - [ONNX Simplifier](https://github.com/daquexian/onnx-simplifier)
  - [TensorRT](https://developer.nvidia.com/tensorrt)

---

## Time Estimate

- **Baseline implementation**: 2 days
- **Compression pipeline**: 2 days
- **Edge deployment**: 1 day
- **Production API**: 1 day
- **Testing & documentation**: 1 day

**Total**: ~7 days full-time

---

## Next Steps

After completing this exercise:

1. Deploy to actual Jetson Nano
2. Benchmark on Raspberry Pi 4
3. Add video object tracking
4. Implement instance segmentation
5. Deploy to Kubernetes cluster
