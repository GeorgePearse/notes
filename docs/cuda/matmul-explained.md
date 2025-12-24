# Matrix Multiplication in CUDA

How matmul works from naive implementations to tensor cores.

## Level 0: Naive CPU (Baseline)

```cpp
// O(N³) - what we're trying to accelerate
void matmul_cpu(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
```

## Level 1: Naive CUDA (One Thread Per Output)

```cpp
// Each thread computes one element of C
// Problem: Tons of redundant global memory reads
__global__ void matmul_naive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // Every thread reads the same A row and B column
            // from slow global memory (HBM) - ~1TB/s bandwidth
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Launch:
// dim3 block(16, 16);
// dim3 grid((N + 15) / 16, (M + 15) / 16);
// matmul_naive<<<grid, block>>>(A, B, C, M, N, K);
```

**Why it's slow:** For a 1024×1024 matmul, each element of A and B gets read ~1024 times from global memory.

## Level 2: Tiled with Shared Memory

```cpp
// The key insight: Load tiles into fast shared memory (SRAM, ~19TB/s)
// then reuse them across threads in the block

#define TILE_SIZE 16

__global__ void matmul_tiled(float* A, float* B, float* C, int M, int N, int K) {
    // Shared memory - visible to all threads in block
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Slide tile across K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Cooperative load: each thread loads one element
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();  // Wait for all threads to finish loading

        // Compute partial dot product from shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();  // Wait before loading next tile
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**Why it's faster:** Each element loaded from global memory once per tile, reused TILE_SIZE times. ~16x fewer global memory accesses.

## Level 3: cuBLAS (What You Should Actually Use)

```cpp
#include <cublas_v2.h>

void matmul_cublas(float* A, float* B, float* C, int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Note: cuBLAS uses column-major, so we compute B^T × A^T = (AB)^T
    // Or just pass your matrices "backwards"
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,           // Dimensions (swapped for row-major)
        &alpha,
        B, N,              // B matrix, leading dimension
        A, K,              // A matrix, leading dimension
        &beta,
        C, N);             // C matrix, leading dimension

    cublasDestroy(handle);
}
```

**Why use it:**
- Auto-selects optimal tile sizes, unroll factors, memory access patterns
- Uses tensor cores automatically when available
- Handles edge cases, alignment, all GPU architectures
- Years of NVIDIA engineer optimization

## Level 4: Tensor Cores with WMMA

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// Tensor cores operate on matrix fragments
// Ampere: 16×16×16 for FP16, meaning:
//   A is 16×16, B is 16×16, accumulator C is 16×16

__global__ void matmul_wmma(half* A, half* B, float* C, int M, int N, int K) {
    // Declare matrix fragments
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    // Initialize accumulator to zero
    fill_fragment(c_frag, 0.0f);

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * 16;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * 16;

    // Loop over K dimension
    for (int k = 0; k < K; k += 16) {
        // Load 16×16 tiles into fragments
        load_matrix_sync(a_frag, A + warpM * K + k, K);
        load_matrix_sync(b_frag, B + k * N + warpN, N);

        // Tensor core matrix multiply-accumulate
        // This single instruction does 16×16×16 = 4096 FMAs
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    store_matrix_sync(C + warpM * N + warpN, c_frag, N, mem_row_major);
}
```

**Key insight:** `mma_sync` executes on tensor cores - specialized hardware that does a 16×16×16 matmul in one operation. A single Ampere SM can do 256 tensor core ops per cycle.

## Level 5: What cuBLAS Actually Does

Real high-performance GEMM combines everything:

```
┌─────────────────────────────────────────────────────┐
│                    Grid Level                        │
│  - Tile output matrix into large blocks             │
│  - Each thread block handles one output tile        │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│                Thread Block Level                    │
│  - Multiple warps cooperate                         │
│  - Double-buffered shared memory                    │
│  - Async global→shared copies (cp.async on Ampere) │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│                   Warp Level                         │
│  - Each warp owns a piece of the output tile        │
│  - WMMA/MMA instructions for tensor cores           │
│  - Register-level tiling for data reuse             │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│               Tensor Core Level                      │
│  - mma.sync.aligned.m16n8k16.f32.f16.f16.f32       │
│  - Hardware does 16×8×16 in one instruction         │
└─────────────────────────────────────────────────────┘
```

## Performance Comparison (Rough)

| Implementation | TFLOPS (A100) | % of Peak |
|----------------|---------------|-----------|
| Naive CUDA     | ~0.5          | 0.3%      |
| Tiled          | ~5            | 3%        |
| cuBLAS FP32    | ~19           | 12%       |
| cuBLAS TF32    | ~150          | 50%       |
| cuBLAS FP16    | ~300          | 95%       |

The jump from tiled to cuBLAS comes from tensor cores + years of micro-optimization.

## PyTorch Connection

```python
# All of these eventually call cuBLAS GEMM:
torch.mm(A, B)
torch.matmul(A, B)
A @ B
F.linear(x, weight)

# Check if tensor cores are being used:
# - Dimensions divisible by 8 (ideal: 16)
# - FP16 or BF16 dtype
# - Compute capability ≥ 7.0
```
