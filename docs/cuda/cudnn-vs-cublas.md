# cuDNN vs cuBLAS

How NVIDIA's two main deep learning libraries relate.

## The Short Answer

```
cuBLAS = Linear algebra primitives (matmul, GEMM)
cuDNN  = Deep learning operations (conv, attention, normalization)
         └── calls cuBLAS internally for the matmul parts
```

cuDNN is a **higher-level** library that **uses cuBLAS** under the hood.

## The Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Your Code / PyTorch                   │
└─────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┴─────────────────┐
          ▼                                   ▼
┌───────────────────┐               ┌───────────────────┐
│      cuDNN        │               │      cuBLAS       │
│  - Convolution    │               │  - GEMM           │
│  - Attention      │               │  - Matrix ops     │
│  - Pooling        │               │  - BLAS 1/2/3     │
│  - Normalization  │               │                   │
│  - RNN/LSTM       │               │                   │
└───────────────────┘               └───────────────────┘
          │                                   │
          │         ┌─────────────┐           │
          └────────►│   cuBLAS    │◄──────────┘
                    └─────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    CUDA Runtime                          │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              GPU Hardware (Tensor Cores, etc.)           │
└─────────────────────────────────────────────────────────┘
```

## What Each Library Owns

### cuBLAS
Pure math operations with no ML semantics:

```cpp
// cuBLAS knows nothing about "layers" or "batches" in the ML sense
cublasSgemm()      // C = αAB + βC
cublasSgemv()      // y = αAx + βy
cublasGemmBatchedEx()  // Many matmuls
```

### cuDNN
ML-specific operations with semantic understanding:

```cpp
// cuDNN understands NCHW tensors, padding, strides, etc.
cudnnConvolutionForward()
cudnnBatchNormalizationForward()
cudnnMultiHeadAttnForward()
cudnnActivationForward()      // ReLU, sigmoid, etc.
cudnnPoolingForward()
cudnnRNNForward()
cudnnSoftmaxForward()
```

## How cuDNN Uses cuBLAS

### Convolution (im2col + GEMM)

The classic conv implementation:

```
Input: [N, C_in, H, W]
Kernel: [C_out, C_in, kH, kW]

Step 1: im2col - unfold input patches into columns
        [N, C_in, H, W] → [N * H_out * W_out, C_in * kH * kW]

Step 2: GEMM via cuBLAS
        Output = Kernel × im2col(Input)
        [C_out, C_in*kH*kW] × [C_in*kH*kW, N*H_out*W_out]

Step 3: Reshape to [N, C_out, H_out, W_out]
```

```cpp
// Simplified: cuDNN conv internally does something like
void cudnn_conv_implicit_gemm(...) {
    // Transform input to matrix form
    im2col(input, col_buffer);

    // Call cuBLAS for the heavy lifting
    cublasGemmEx(handle, ..., kernel, col_buffer, output, ...);

    // Handle bias, activation fusion
}
```

### Attention (FMHA)

```
Q, K, V: [batch, heads, seq, dim]

Internally:
1. Q × K^T → cuBLAS batched GEMM
2. Softmax → cuDNN softmax kernel
3. scores × V → cuBLAS batched GEMM

(Or fused into single FMHA kernel that still uses tensor core matmul primitives)
```

### Linear Layers

`nn.Linear` typically bypasses cuDNN entirely:

```python
# PyTorch Linear forward
def forward(self, x):
    # Goes directly to cuBLAS, no cuDNN involved
    return F.linear(x, self.weight, self.bias)
    # → cublasGemmEx() under the hood
```

## When Each Gets Called (PyTorch)

| Operation | Library |
|-----------|---------|
| `torch.mm`, `@`, `torch.matmul` | cuBLAS |
| `F.linear` | cuBLAS |
| `F.conv2d` | cuDNN |
| `F.batch_norm` | cuDNN |
| `F.layer_norm` | cuDNN (or custom kernel) |
| `F.scaled_dot_product_attention` | cuDNN (FMHA) or Flash Attention |
| `F.relu`, `F.gelu` | cuDNN or custom |
| `F.softmax` | cuDNN |
| `F.max_pool2d` | cuDNN |
| `nn.LSTM` | cuDNN |

## Why Separate Libraries?

### cuBLAS (1990s BLAS heritage)
- Generic math, not ML-specific
- Stable API, decades of optimization
- Used by everything: scientific computing, graphics, ML, etc.
- Focus: make GEMM as fast as possible

### cuDNN (2014, deep learning boom)
- ML-specific optimizations (fused ops, tensor formats)
- Algorithm selection (Winograd, FFT, implicit GEMM for conv)
- Workspace management for temporary buffers
- Fused operations (conv + bias + activation in one kernel)
- Focus: make neural nets as fast as possible

## Practical Implications

### Debugging

```python
# Disable cuDNN to isolate issues
torch.backends.cudnn.enabled = False  # Falls back to slower but more stable

# cuBLAS has no equivalent toggle - it's always there
```

### Determinism

```python
# cuDNN has non-deterministic algorithms by default
torch.backends.cudnn.deterministic = True  # Slower but reproducible

# cuBLAS is generally deterministic
# (except for some reduction operations)
```

### Benchmarking

```python
# cuDNN can benchmark algorithms at runtime
torch.backends.cudnn.benchmark = True

# First conv with each shape: try multiple algorithms, cache winner
# Later convs: use cached best algorithm

# No equivalent for cuBLAS - algorithm is shape-dependent but auto-selected
```

## Performance: Who's the Bottleneck?

For most models:

```
Transformer: 90%+ time in cuBLAS (matmuls dominate)
             └── attention QKV projections, FFN layers

CNN:         60-80% time in cuDNN (convolutions)
             └── some in cuBLAS for FC layers

RNN/LSTM:    Mixed - cuDNN has fused RNN kernels
             └── matrix ops inside are still GEMM
```

```python
# Profile to see the split
with torch.profiler.profile() as prof:
    model(input)

# Look for:
# - ampere_sgemm_*, volta_sgemm_* → cuBLAS
# - cudnn_*, winograd_*, implicit_gemm_* → cuDNN
```

## Summary

| Aspect | cuBLAS | cuDNN |
|--------|--------|-------|
| Level | Low (primitives) | High (operations) |
| Domain | Linear algebra | Deep learning |
| Key ops | GEMM, BLAS 1/2/3 | Conv, attention, norm |
| Calls other? | No | Yes (uses cuBLAS) |
| Determinism | Mostly yes | Configurable |
| Algorithm selection | Automatic | Benchmarkable |

Think of cuBLAS as the **engine** and cuDNN as the **car** - cuDNN handles the ML semantics and calls cuBLAS when it needs raw matrix math.
