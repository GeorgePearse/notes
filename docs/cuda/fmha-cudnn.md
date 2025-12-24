# FMHA in cuDNN

FMHA = **Fused Multi-Head Attention**

## What It Is

cuDNN's FMHA is a highly optimized implementation of the transformer attention mechanism that fuses multiple operations into a single GPU kernel. Instead of executing separate kernels for Q×K^T, softmax, and ×V, FMHA does it all in one pass.

## Why It Matters

Standard attention is memory-bound:

```
# Naive attention (3 separate kernels, 2 large intermediate tensors)
scores = Q @ K.T              # Write NxN to HBM
weights = softmax(scores)     # Read NxN, write NxN to HBM
output = weights @ V          # Read NxN from HBM
```

FMHA eliminates the intermediate memory traffic by:

1. **Tiling** - Process attention in blocks that fit in SRAM
2. **Kernel fusion** - Never materialize the full N×N attention matrix
3. **Online softmax** - Compute softmax incrementally without a second pass

Memory complexity drops from O(N²) to O(N) for sequence length N.

## cuDNN's Implementation

cuDNN 8.9+ includes `cudnnMultiHeadAttnForward()` with:

- Flash Attention-style tiling
- Support for causal masks, padding masks, dropout
- FP16, BF16, FP8 (Hopper)
- Automatic selection of optimal tile sizes per GPU architecture

## How to Use It

### Via PyTorch (easiest)

```python
# PyTorch 2.0+ automatically dispatches to cuDNN FMHA when available
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(q, k, v)
```

### Via cuDNN directly

```cpp
cudnnAttnDescriptor_t attnDesc;
cudnnCreateAttnDescriptor(&attnDesc);
cudnnSetAttnDescriptor(attnDesc,
    CUDNN_ATTN_QUERYMAP_ALL_TO_ONE,
    nHeads, smScaler,
    CUDNN_DATA_HALF, CUDNN_DATA_HALF,
    mathType, NULL, NULL);

cudnnMultiHeadAttnForward(handle, attnDesc, ...);
```

## When cuDNN FMHA Gets Used

PyTorch's `scaled_dot_product_attention` tries backends in order:

1. **Flash Attention** - If head_dim ≤ 256, no custom mask
2. **cuDNN FMHA** - If available and constraints met
3. **Memory-efficient attention** - xFormers-style chunked
4. **Math fallback** - Naive implementation

Check what's being used:

```python
from torch.backends.cuda import sdp_kernel, SDPBackend

with sdp_kernel(SDPBackend.CUDNN_ATTENTION):
    # Force cuDNN path
    output = F.scaled_dot_product_attention(q, k, v)
```

## Performance Characteristics

Typical speedups over naive attention:

| Sequence Length | Speedup |
|-----------------|---------|
| 512             | 2-3x    |
| 2048            | 4-6x    |
| 8192            | 8-12x   |

Memory savings scale with sequence length squared - critical for long-context models.

## Limitations

- Head dimensions must be 64, 128, or 256 (varies by cuDNN version)
- Some mask patterns not supported
- Requires compute capability 8.0+ (Ampere) for best performance
- Backward pass has different constraints than forward
