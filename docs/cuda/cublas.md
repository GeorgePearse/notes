# cuBLAS

NVIDIA's GPU-accelerated Basic Linear Algebra Subroutines library.

## What It Is

cuBLAS = **CUDA Basic Linear Algebra Subroutines**

It's the GPU equivalent of CPU BLAS libraries (OpenBLAS, MKL, ATLAS). When you do `torch.mm()` or `@ `in PyTorch, cuBLAS is what actually runs on the GPU.

## The BLAS Hierarchy

BLAS operations are organized into three levels:

| Level | Operations | Memory Complexity | Compute Complexity |
|-------|------------|-------------------|-------------------|
| 1 | Vector-vector (dot, axpy, scale) | O(n) | O(n) |
| 2 | Matrix-vector (gemv, trsv) | O(n²) | O(n²) |
| 3 | Matrix-matrix (gemm, trsm) | O(n²) | O(n³) |

Level 3 ops (especially GEMM) are where GPUs shine - high arithmetic intensity means compute-bound, not memory-bound.

## GEMM: The Core Operation

**G**eneral **M**atrix **M**ultiply:

```
C = α(op(A) × op(B)) + βC
```

Where `op()` can be normal, transpose, or conjugate transpose.

```cpp
cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,  // CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C
    cublasOperation_t transb,
    int m, int n, int k,       // Output is m×n, inner dimension is k
    const float *alpha,
    const float *A, int lda,   // A is m×k (or k×m if transposed)
    const float *B, int ldb,   // B is k×n (or n×k if transposed)
    const float *beta,
    float *C, int ldc          // C is m×n
);
```

## Column-Major Gotcha

cuBLAS uses **column-major** ordering (Fortran style), not row-major (C style):

```
Row-major (C/C++/Python):     Column-major (Fortran/cuBLAS):
┌─────────────┐               ┌─────────────┐
│ 1  2  3  4  │               │ 1  5  9  13 │
│ 5  6  7  8  │      →        │ 2  6  10 14 │
│ 9  10 11 12 │               │ 3  7  11 15 │
│ 13 14 15 16 │               │ 4  8  12 16 │
└─────────────┘               └─────────────┘
Memory: [1,2,3,4,5,6,...]     Memory: [1,5,9,13,2,6,...]
```

**The trick:** For row-major matrices, compute `B × A` instead of `A × B`:

```cpp
// You want: C = A × B (row-major)
// Call:     C^T = B^T × A^T (column-major)
// Which is the same memory layout!

// So swap A and B, swap m and n:
cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, m, k,        // Note: n and m swapped
    &alpha,
    B, n,           // B first
    A, k,           // A second
    &beta,
    C, n);
```

## cuBLAS API Flavors

### Legacy API (cuBLAS v1)
```cpp
// Global state, implicit handle - don't use this
cublasSgemm('N', 'N', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
```

### Modern API (cuBLAS v2)
```cpp
// Explicit handle, pointer scalars, better error handling
cublasHandle_t handle;
cublasCreate(&handle);
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
            &alpha, A, lda, B, ldb, &beta, C, ldc);
cublasDestroy(handle);
```

### cublasEx API (flexible types)
```cpp
// Mixed precision, tensor cores
cublasGemmEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    A, CUDA_R_16F, lda,  // FP16 input
    B, CUDA_R_16F, ldb,  // FP16 input
    &beta,
    C, CUDA_R_32F, ldc,  // FP32 output
    CUBLAS_COMPUTE_32F,  // Compute in FP32
    CUBLAS_GEMM_DEFAULT_TENSOR_OP  // Use tensor cores
);
```

### cublasLt API (maximum control)
```cpp
// Layout descriptors, algorithm selection, workspace management
cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
cublasLtMatmulPreferenceCreate(&preference);
cublasLtMatmulAlgoGetHeuristic(lightHandle, matmulDesc, ...);
cublasLtMatmul(lightHandle, matmulDesc, ...);
```

## Tensor Core Usage

Tensor cores are used automatically when:

1. **Dimensions** are multiples of 8 (16 for best performance)
2. **Types** are FP16, BF16, TF32, INT8, or FP8
3. **Algorithm** allows it (CUBLAS_GEMM_DEFAULT_TENSOR_OP)

```cpp
// Force tensor cores with cublasGemmEx
cublasGemmEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    A, CUDA_R_16F, lda,
    B, CUDA_R_16F, ldb,
    &beta,
    C, CUDA_R_16F, ldc,
    CUBLAS_COMPUTE_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP  // Key flag
);

// Check math mode
cublasMath_t mode;
cublasGetMathMode(handle, &mode);
// CUBLAS_DEFAULT_MATH - no tensor cores
// CUBLAS_TENSOR_OP_MATH - tensor cores enabled
// CUBLAS_TF32_TENSOR_OP_MATH - TF32 on Ampere+
```

## TF32: Ampere's Secret Sauce

On Ampere+, FP32 GEMMs can use TF32 (19-bit float) automatically:

```cpp
// Enable TF32 (default on Ampere)
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

// Now regular Sgemm uses tensor cores internally
cublasSgemm(handle, ...);  // 8x faster than pure FP32!
```

```python
# PyTorch control
torch.backends.cuda.matmul.allow_tf32 = True   # Default on Ampere
torch.backends.cudnn.allow_tf32 = True
```

TF32 has FP32 range but reduced precision (10-bit mantissa vs 23-bit). Usually fine for training, occasionally matters for inference.

## Batched Operations

For many small matmuls (like attention heads):

```cpp
// Strided batched - matrices evenly spaced in memory
cublasGemmStridedBatchedEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    A, CUDA_R_16F, lda, strideA,  // strideA = m*k
    B, CUDA_R_16F, ldb, strideB,  // strideB = k*n
    &beta,
    C, CUDA_R_16F, ldc, strideC,  // strideC = m*n
    batchCount,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT
);

// Pointer array batched - arbitrary matrix locations
const void *Aarray[batchCount];
const void *Barray[batchCount];
void *Carray[batchCount];
cublasGemmBatchedEx(handle, ..., Aarray, ..., Barray, ..., Carray, ...);
```

## Common Functions

### Level 1 (Vector)
```cpp
cublasSasum(handle, n, x, incx, result);           // Sum of absolute values
cublasSaxpy(handle, n, &alpha, x, incx, y, incy);  // y = αx + y
cublasSdot(handle, n, x, incx, y, incy, result);   // Dot product
cublasSnrm2(handle, n, x, incx, result);           // L2 norm
cublasSscal(handle, n, &alpha, x, incx);           // x = αx
```

### Level 2 (Matrix-Vector)
```cpp
cublasSgemv(handle, trans, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
// y = αAx + βy
```

### Level 3 (Matrix-Matrix)
```cpp
cublasSgemm(handle, ...);   // C = αAB + βC
cublasStrsm(handle, ...);   // Triangular solve: op(A)X = αB
cublasSsyrk(handle, ...);   // Symmetric rank-k: C = αAA^T + βC
```

## Error Handling

```cpp
cublasStatus_t status = cublasSgemm(handle, ...);
if (status != CUBLAS_STATUS_SUCCESS) {
    switch (status) {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            // Handle not created
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            // Bad parameter
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            // GPU doesn't support operation
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            // Kernel launch failed
            break;
    }
}
```

## PyTorch Integration

```python
import torch

# All of these use cuBLAS:
C = torch.mm(A, B)                    # Matrix multiply
C = A @ B                             # Same thing
C = torch.bmm(A, B)                   # Batched matmul
y = torch.mv(A, x)                    # Matrix-vector
out = torch.nn.functional.linear(x, W, b)  # Wx + b

# Check what's happening under the hood:
with torch.profiler.profile() as prof:
    C = A @ B
print(prof.key_averages().table())
# Shows: ampere_sgemm_128x64_tn (or similar cuBLAS kernel)
```

## Performance Tips

1. **Align dimensions to 8 or 16** for tensor cores
2. **Use FP16/BF16** when precision allows - 2x memory bandwidth, tensor cores
3. **Batch small matmuls** with strided batched API
4. **Reuse handles** - creation has overhead
5. **Use cublasLt** for repeated shapes - algorithm selection is cached
6. **Enable TF32** for FP32 workloads on Ampere+

```python
# Padding for tensor core alignment
def pad_to_multiple(tensor, multiple=8):
    *batch, m, n = tensor.shape
    pad_m = (multiple - m % multiple) % multiple
    pad_n = (multiple - n % multiple) % multiple
    return F.pad(tensor, (0, pad_n, 0, pad_m))
```
