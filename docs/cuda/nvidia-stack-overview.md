# NVIDIA Software Stack Overview

The full picture of NVIDIA's GPU computing ecosystem for ML.

## The Complete Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Applications                                 │
│              PyTorch, TensorFlow, JAX, etc.                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                     High-Level Libraries                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ TensorRT │ │  Triton  │ │  DALI    │ │  RAPIDS  │ │ DeepSpeed│  │
│  │(inference)│ │(serving) │ │(data)    │ │(analytics)│ │(training)│  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                     Core ML Libraries                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│  │  cuDNN   │ │  cuBLAS  │ │  NCCL    │ │ CUTLASS  │               │
│  │ (DL ops) │ │ (GEMM)   │ │(comms)   │ │(templates)│               │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                     Math Libraries                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  cuFFT   │ │ cuSPARSE │ │  cuRAND  │ │cuSOLVER  │ │  Thrust  │  │
│  │  (FFT)   │ │ (sparse) │ │ (random) │ │(solvers) │ │(parallel)│  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                     CUDA Runtime & Tools                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  CUDA    │ │   nvcc   │ │  Nsight  │ │   NVTX   │ │  NVML    │  │
│  │ Runtime  │ │(compiler)│ │(profiler)│ │(markers) │ │(mgmt)    │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                     CUDA Driver                                      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                     GPU Hardware                                     │
│            Tensor Cores, CUDA Cores, HBM, NVLink                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Libraries You'll Actually Use

### Tier 1: Critical for ML (you use these daily)

| Library | Purpose | When It's Called |
|---------|---------|------------------|
| **cuBLAS** | Matrix multiply | Every `@`, `torch.mm`, linear layer |
| **cuDNN** | Conv, attention, norms | Conv layers, transformers, batch norm |
| **NCCL** | Multi-GPU communication | `DistributedDataParallel`, FSDP |

### Tier 2: Important for Production

| Library | Purpose | When It's Called |
|---------|---------|------------------|
| **TensorRT** | Inference optimization | Deploying models to production |
| **Triton Inference Server** | Model serving | Serving models at scale |
| **DALI** | Data loading | GPU-accelerated preprocessing |

### Tier 3: Specialized Use Cases

| Library | Purpose | When It's Called |
|---------|---------|------------------|
| **CUTLASS** | Custom GEMM kernels | Building custom ops, understanding internals |
| **cuFFT** | Fourier transforms | Signal processing, some conv algorithms |
| **cuSPARSE** | Sparse matrices | Sparse models, graph neural networks |
| **cuRAND** | Random numbers | Dropout, initialization, sampling |
| **Thrust** | Parallel algorithms | Custom CUDA code, sorting, reductions |

### Tier 4: Development & Debugging

| Tool | Purpose | When Used |
|------|---------|-----------|
| **Nsight Compute** | Kernel profiling | Optimizing CUDA kernels |
| **Nsight Systems** | System profiling | Finding bottlenecks |
| **NVTX** | Profiling markers | Adding annotations to profiles |
| **NVML** | GPU management | Monitoring, multi-GPU setup |

## What Calls What

```
torch.nn.Linear(x)
    └── F.linear()
        └── torch._C._nn.linear()
            └── at::linear()
                └── cuBLAS cublasGemmEx()

torch.nn.Conv2d(x)
    └── F.conv2d()
        └── cudnn_convolution()
            └── cuDNN cudnnConvolutionForward()
                └── cuBLAS (for implicit GEMM algorithm)

DistributedDataParallel gradient sync
    └── torch.distributed
        └── ProcessGroupNCCL
            └── NCCL ncclAllReduce()
                └── NVLink / NVSwitch / InfiniBand

model.to('cuda')
    └── CUDA Runtime cudaMemcpy()
        └── CUDA Driver
            └── PCIe / NVLink transfer
```

## The Libraries in Detail

### NCCL (Multi-GPU Communication)

**N**VIDIA **C**ollective **C**ommunications **L**ibrary

```python
# You don't call NCCL directly, but it powers:
model = DistributedDataParallel(model)  # Uses NCCL
model = FSDP(model)                      # Uses NCCL

# NCCL operations:
# - AllReduce: Sum gradients across GPUs
# - AllGather: Collect tensors from all GPUs
# - Broadcast: Send from one GPU to all
# - ReduceScatter: Reduce and distribute
```

Critical for training on multiple GPUs - handles NVLink, NVSwitch, InfiniBand automatically.

### TensorRT (Inference Optimization)

Optimizes trained models for deployment:

```python
import torch_tensorrt

# Compile PyTorch model to TensorRT
optimized = torch_tensorrt.compile(model,
    inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224])],
    enabled_precisions={torch.float16}
)

# 2-6x faster inference
output = optimized(input)
```

What TensorRT does:
- Layer fusion (conv + bn + relu → single kernel)
- Precision calibration (FP32 → FP16/INT8)
- Kernel auto-tuning
- Memory optimization

### CUTLASS (Template Library)

Building blocks for custom GEMM kernels:

```cpp
// CUTLASS is what cuBLAS is built on
// Use it when you need custom matrix operations

#include <cutlass/gemm/device/gemm.h>

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,              // A type
    cutlass::layout::RowMajor,    // A layout
    cutlass::half_t,              // B type
    cutlass::layout::RowMajor,    // B layout
    cutlass::half_t,              // C type
    cutlass::layout::RowMajor,    // C layout
    float                         // Accumulator
>;

Gemm gemm_op;
gemm_op(args);
```

PyTorch uses CUTLASS for some operations. Flash Attention was originally built on CUTLASS.

### cuFFT (Fourier Transforms)

```python
# PyTorch wraps cuFFT
spectrum = torch.fft.fft2(image)

# Used internally by cuDNN for FFT-based convolution
# (one of several conv algorithms)
```

### cuSPARSE (Sparse Operations)

```python
# PyTorch sparse tensors use cuSPARSE
sparse = torch.sparse_csr_tensor(crow, col, values)
result = sparse @ dense  # cuSPARSE SpMM
```

Increasingly relevant for:
- Sparse attention patterns
- Pruned models
- Graph neural networks

### cuRAND (Random Numbers)

```python
# Powers random operations
torch.randn(1000, 1000, device='cuda')  # Uses cuRAND
F.dropout(x, p=0.1)                      # Uses cuRAND for mask
```

### DALI (Data Loading)

GPU-accelerated data pipeline:

```python
from nvidia.dali import pipeline_def, fn

@pipeline_def
def image_pipeline():
    images = fn.readers.file(file_root="data")
    images = fn.decoders.image(images, device="mixed")  # Decode on GPU
    images = fn.resize(images, size=(224, 224))
    images = fn.crop_mirror_normalize(images,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    return images

# Keeps GPU fed while training
```

### Nsight Tools (Profiling)

```bash
# System-level profiling (find bottlenecks)
nsys profile python train.py
nsys-ui report.nsys-rep

# Kernel-level profiling (optimize kernels)
ncu python train.py
ncu-ui report.ncu-rep
```

```python
# Add markers for profiling
import torch.cuda.nvtx as nvtx

with nvtx.range("forward_pass"):
    output = model(input)

with nvtx.range("backward_pass"):
    loss.backward()
```

## Version Compatibility

The dreaded version matrix:

```
CUDA Toolkit  ←→  Driver  ←→  cuDNN  ←→  PyTorch
     │              │           │           │
     └──────────────┴───────────┴───────────┘
              Must be compatible!
```

```bash
# Check versions
nvidia-smi                    # Driver version
nvcc --version               # CUDA toolkit version
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA
python -c "import torch; print(torch.backends.cudnn.version())"  # cuDNN
```

PyTorch ships with its own cuDNN/cuBLAS, but uses system CUDA driver.

## Summary: What You Need to Know

| If you're... | Focus on... |
|--------------|-------------|
| Training single GPU | cuBLAS, cuDNN (automatic) |
| Training multi-GPU | Above + NCCL |
| Deploying models | TensorRT, Triton Server |
| Optimizing performance | Nsight tools, NVTX |
| Building custom ops | CUTLASS, CUDA C++ |
| Data bottlenecked | DALI |
| Working with sparse | cuSPARSE |

Most of the time, PyTorch abstracts everything away. But knowing the stack helps when:
- Debugging performance issues
- Understanding error messages
- Choosing the right tools for deployment
- Writing custom CUDA extensions
