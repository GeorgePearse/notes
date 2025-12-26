# CUDA Extensions for PyTorch

Notes on tooling for integrating CUDA extensions into visdet (mmdetection fork).

## Build Systems

### Ninja
- What mmdetection uses
- Parallel build system, much faster than make for CUDA compilation
- Key advantage: only recompiles changed files
- Install: `pip install ninja` or system package manager
- PyTorch's cpp_extension auto-detects and uses ninja if available

### setuptools
- Standard Python packaging
- Works but slow for CUDA - recompiles everything each time
- Fine for distribution, painful for development

## Core Libraries

### torch.utils.cpp_extension
The standard approach for PyTorch CUDA extensions:

```python
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    ext_modules=[
        CUDAExtension(
            name='my_cuda_ops',
            sources=['csrc/my_op.cpp', 'csrc/my_op_cuda.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

Two modes:
- **AOT (Ahead of Time)** - compile during `pip install`, what mmdetection does
- **JIT (Just in Time)** - compile on first use via `load()`, good for prototyping

### nvcc
- NVIDIA's CUDA compiler
- Comes with CUDA toolkit
- cpp_extension wraps this for you
- Version compatibility matters - match PyTorch's CUDA version

### pybind11
- C++ to Python bindings
- PyTorch includes it, don't need separate install
- Used to expose CUDA kernels to Python

## Alternative Approaches

### CuPy
- NumPy-like API with CUDA backend
- Built-in JIT compilation for custom kernels
- Less boilerplate than cpp_extension
- Trade-off: less control, potentially less optimized

### Triton (OpenAI)
- Python-like syntax for GPU kernels
- JIT compiled to PTX
- Much easier than raw CUDA
- Good for custom ops, less for porting existing CUDA code

### CUDA Python (NVIDIA)
- Direct CUDA API access from Python
- Low-level, more control
- Probably overkill for detection ops

## How mmdetection / mmcv Split Works

### Historical Context (important for visdet)

**mmdetection v2.x** - HAD its own CUDA extensions:
- nms_ext, roi_align_ext, roi_pool_ext
- deform_conv_ext, deform_pool_ext
- sigmoid_focal_loss_ext, masked_conv2d_ext
- carafe_ext
- Built directly via setup.py with `make_cuda_ext()`

**mmdetection v3.x (current)** - pure Python now:
- All ops migrated to mmcv
- setup.py still has the infrastructure but `ext_modules=[]`
- Imports compiled ops from mmcv

**mmcv** = consolidated ops library:
- `mmcv/ops/csrc/` contains both C++ and CUDA implementations
- C++ for CPU fallbacks, CUDA for GPU
- 65+ ops now, expanded beyond original mmdetection set
- Uses ninja for fast parallel builds
- Pre-built wheels for common CUDA versions

### Implication for visdet

If forking, you have two reference points:
1. **mmdetection v2.x** - simpler, self-contained ops in one repo
2. **mmcv current** - more ops, more polished, but separate dependency

Key files to study:
- mmdetection v2 `setup.py` - how ops were built inline
- mmcv `setup.py` - current build configuration
- mmcv `mmcv/ops/csrc/` - C++ and CUDA source
- mmcv `mmcv/ops/*.py` - Python wrappers

## Decisions for visdet

### Option 1: Fork mmcv ops
- Pros: battle-tested, comprehensive ops
- Cons: brings in mmcv complexity, dependency hell

### Option 2: Use torchvision ops where possible
- `torchvision.ops` has roi_align, nms, deformable_conv
- Less custom code to maintain
- May be missing some specialized ops

### Option 3: Build minimal custom ops
- Only what's not in torchvision
- Cleaner, easier to maintain
- More work upfront

### Recommended: Hybrid
1. Use torchvision.ops as much as possible
2. Build minimal custom ops with cpp_extension + ninja
3. Structure similar to mmcv but simpler

## Practical Setup

```bash
# Requirements
pip install ninja
# CUDA toolkit must be installed and match PyTorch
nvcc --version  # verify

# Directory structure
visdet/
  ops/
    csrc/
      my_op.cpp       # CPU fallback + dispatch
      my_op_cuda.cu   # CUDA kernels
    __init__.py       # Python API
  setup.py            # with CUDAExtension
```

## Common Gotchas

- **CUDA/PyTorch version mismatch** - most common issue
- **Missing CUDA_HOME** - export CUDA_HOME=/usr/local/cuda
- **ABI compatibility** - compile with same compiler as PyTorch
- **Dynamic parallelism** - needs explicit nvcc flags if used
- **Shared memory limits** - device-specific, test on target hardware

## Resources

- PyTorch custom C++ and CUDA extensions tutorial
- mmcv source code
- torchvision ops implementations
- detectron2's approach (Facebook's take on same problem)
