# TensorRT

NVIDIA's inference optimization and runtime engine.

## What It Does

TensorRT takes a trained model and produces an optimized engine:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Trained Model  │ ──► │    TensorRT     │ ──► │ Optimized Engine│
│ (PyTorch/ONNX)  │     │   (Optimizer)   │     │   (2-6x faster) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Optimizations Applied

### 1. Layer Fusion

```
Before:                          After:
┌──────┐   ┌──────┐   ┌──────┐   ┌─────────────────────┐
│ Conv │ → │  BN  │ → │ ReLU │   │ Conv+BN+ReLU (fused)│
└──────┘   └──────┘   └──────┘   └─────────────────────┘
3 kernel launches                 1 kernel launch
```

Common fusions:
- Conv + BatchNorm + Activation
- Linear + Bias + Activation
- Multiple element-wise ops

### 2. Precision Reduction

```
FP32 (training) → FP16/INT8 (inference)
- Half the memory bandwidth
- 2-4x tensor core speedup
- Calibration for INT8 quantization
```

### 3. Kernel Auto-Tuning

```
For each layer:
- Try multiple kernel implementations
- Benchmark on target GPU
- Select fastest
- Cache selection in engine file
```

### 4. Memory Optimization

```
- Reuse memory across layers
- Optimal tensor layouts (NHWC vs NCHW)
- Minimize GPU memory footprint
```

## Usage: PyTorch Path

### torch.compile (easiest, PyTorch 2.0+)

```python
import torch

model = MyModel().cuda().eval()

# Use TensorRT backend via torch.compile
compiled = torch.compile(model, backend="tensorrt")

# First call builds engine (slow), subsequent calls fast
output = compiled(input)
```

### Torch-TensorRT (more control)

```python
import torch_tensorrt

model = MyModel().cuda().eval()

# Compile with explicit settings
optimized = torch_tensorrt.compile(model,
    inputs=[torch_tensorrt.Input(
        min_shape=[1, 3, 224, 224],
        opt_shape=[8, 3, 224, 224],
        max_shape=[32, 3, 224, 224],
        dtype=torch.float16
    )],
    enabled_precisions={torch.float16},
    workspace_size=1 << 30,  # 1GB workspace
)

# Save for deployment
torch.jit.save(optimized, "model_trt.ts")
```

### Dynamic shapes

```python
# TensorRT supports dynamic batch size
inputs = [
    torch_tensorrt.Input(
        min_shape=[1, 3, 224, 224],    # Minimum batch
        opt_shape=[16, 3, 224, 224],   # Optimal (tuned for)
        max_shape=[64, 3, 224, 224],   # Maximum batch
    )
]
```

## Usage: ONNX Path

```python
# 1. Export to ONNX
torch.onnx.export(model, example_input, "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)

# 2. Build TensorRT engine (Python API)
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, logger)

with open("model.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

# Enable FP16
config.set_flag(trt.BuilderFlag.FP16)

# Build engine
engine = builder.build_serialized_network(network, config)

# Save engine
with open("model.engine", "wb") as f:
    f.write(engine)
```

```python
# 3. Run inference
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(open("model.engine", "rb").read())
context = engine.create_execution_context()

# Allocate buffers and run
# (Usually use higher-level wrapper)
```

## INT8 Quantization

For maximum performance, quantize to INT8:

```python
import torch_tensorrt

# Post-training quantization with calibration
def calibration_dataloader():
    for batch in representative_data:
        yield batch

optimized = torch_tensorrt.compile(model,
    inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224])],
    enabled_precisions={torch.int8, torch.float16},
    calibrator=torch_tensorrt.ptq.DataLoaderCalibrator(
        calibration_dataloader(),
        cache_file="calibration.cache",
        algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2
    )
)
```

INT8 provides:
- 4x memory bandwidth vs FP32
- Fastest tensor core operations
- ~1% accuracy loss (model dependent)

## Triton Inference Server

For serving TensorRT models at scale:

```python
# model_repository/
# └── my_model/
#     ├── config.pbtxt
#     └── 1/
#         └── model.plan  # TensorRT engine
```

```protobuf
# config.pbtxt
name: "my_model"
platform: "tensorrt_plan"
max_batch_size: 64
input [
  { name: "input", data_type: TYPE_FP16, dims: [3, 224, 224] }
]
output [
  { name: "output", data_type: TYPE_FP16, dims: [1000] }
]
instance_group [
  { count: 2, kind: KIND_GPU }
]
dynamic_batching {
  max_queue_delay_microseconds: 100
}
```

```bash
# Launch server
docker run --gpus all -p 8000:8000 -p 8001:8001 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

## When to Use TensorRT

| Scenario | Recommendation |
|----------|----------------|
| Development/debugging | PyTorch eager |
| Training | PyTorch (TensorRT is inference only) |
| Batch inference | TensorRT |
| Real-time inference | TensorRT + Triton |
| Edge deployment | TensorRT (Jetson) |
| Quick optimization | torch.compile |

## Performance Expectations

Typical speedups vs PyTorch eager (varies by model):

| Model Type | FP16 Speedup | INT8 Speedup |
|------------|--------------|--------------|
| ResNet-50 | 2-3x | 3-5x |
| BERT | 2-4x | 3-6x |
| Transformer (LLM) | 1.5-2x | 2-3x |
| U-Net | 2-3x | 3-4x |

Latency improvements are often larger than throughput due to kernel fusion.

## Gotchas

### Engine is GPU-specific
```python
# Engine built on A100 won't run on V100
# Rebuild for each target GPU architecture
```

### Build time can be long
```python
# Complex models: 10-60 minutes to build
# Cache the .engine file, don't rebuild in production
```

### Not all ops supported
```python
# Some custom ops need plugins or fall back to PyTorch
# Check supported ops: https://github.com/NVIDIA/TensorRT/tree/main/plugin
```

### Dynamic shapes have overhead
```python
# Fixed shapes are fastest
# If batch size varies, use opt_shape for common case
```

## Comparison with Alternatives

| Tool | Strengths | When to Use |
|------|-----------|-------------|
| **TensorRT** | Maximum NVIDIA GPU perf | Production inference on NVIDIA |
| **ONNX Runtime** | Cross-platform | Need CPU/AMD/multiple backends |
| **torch.compile** | Easy, good perf | Quick wins, PyTorch ecosystem |
| **OpenVINO** | Intel hardware | Intel CPUs/GPUs |
| **TFLite** | Mobile/edge | Android/iOS/embedded |

TensorRT is the right choice when you're deploying to NVIDIA GPUs and need maximum performance.
