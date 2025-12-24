# BF16 vs FP16

Two 16-bit floating point formats with different tradeoffs.

## Bit Layout

```
FP32 (reference):
┌─────────┬──────────────┬───────────────────────────────┐
│ 1 sign  │  8 exponent  │        23 mantissa            │
└─────────┴──────────────┴───────────────────────────────┘

FP16 (IEEE 754 half precision):
┌─────────┬──────────────┬─────────────┐
│ 1 sign  │  5 exponent  │ 10 mantissa │
└─────────┴──────────────┴─────────────┘

BF16 (bfloat16 - "brain float"):
┌─────────┬──────────────┬─────────┐
│ 1 sign  │  8 exponent  │ 7 mant  │
└─────────┴──────────────┴─────────┘
```

## The Key Difference

|          | Exponent Bits | Mantissa Bits | Range          | Precision       |
|----------|---------------|---------------|----------------|-----------------|
| FP32     | 8             | 23            | ±3.4 × 10³⁸    | ~7 decimal digits |
| FP16     | 5             | 10            | ±65,504        | ~3 decimal digits |
| BF16     | 8             | 7             | ±3.4 × 10³⁸    | ~2 decimal digits |

- **FP16**: More precision, less range
- **BF16**: Less precision, same range as FP32

## Why BF16 Exists

FP16's small range (max ~65K) causes problems in training:

```python
# FP16 overflow example
import torch

x = torch.tensor([60000.0], dtype=torch.float16)
y = x * 2  # Overflow! Returns inf

# BF16 handles it fine
x = torch.tensor([60000.0], dtype=torch.bfloat16)
y = x * 2  # Returns 120000.0
```

Gradient values during backprop can easily exceed 65K, causing:
- `inf` values that poison the entire model
- Need for careful loss scaling (AMP's GradScaler)

BF16 was designed at Google Brain specifically for ML training - same dynamic range as FP32 means gradients don't overflow.

## Precision vs Range Visualized

```
                    FP16 representable range
                    ├────────────────────────┤
    ──────┼─────────┼────────────────────────┼─────────┼──────▶
       -65504      -1                        1      65504

                    BF16 representable range
├───────────────────────────────────────────────────────────────┤
────┼───────────────┼────────────────────────┼───────────────┼──▶
  -3.4×10³⁸        -1                        1          3.4×10³⁸


FP16 precision near 1.0:  1.0, 1.001, 1.002, 1.003, ...  (1024 steps to 2.0)
BF16 precision near 1.0:  1.0, 1.008, 1.016, 1.023, ...  (128 steps to 2.0)
```

## When to Use Each

### Use BF16 when:
- **Training** - gradient magnitudes vary wildly
- You want to avoid loss scaling complexity
- Hardware supports it (Ampere+, TPUs)
- Numerical stability matters more than precision

### Use FP16 when:
- **Inference** - values are bounded, no gradients
- Older hardware (Volta, Turing) - no BF16 tensor cores
- Memory bandwidth is the bottleneck
- You need slightly better precision

### Use FP8 when (Hopper+):
- Inference at scale
- You've validated accuracy is acceptable
- Maximum throughput needed

## Hardware Support

| GPU Generation | FP16 Tensor Cores | BF16 Tensor Cores |
|----------------|-------------------|-------------------|
| Volta (V100)   | ✓                 | ✗                 |
| Turing (T4)    | ✓                 | ✗                 |
| Ampere (A100)  | ✓                 | ✓                 |
| Hopper (H100)  | ✓                 | ✓                 |
| TPU (all)      | ✓                 | ✓ (native)        |

## PyTorch Usage

```python
import torch

# Create tensors
fp16_tensor = torch.randn(1000, 1000, dtype=torch.float16, device='cuda')
bf16_tensor = torch.randn(1000, 1000, dtype=torch.bfloat16, device='cuda')

# Automatic Mixed Precision - BF16 (recommended for Ampere+)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)
# No GradScaler needed with BF16!
loss.backward()
optimizer.step()

# Automatic Mixed Precision - FP16 (needs scaler)
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Conversion

```python
# FP32 → BF16: Just truncate mantissa (fast!)
# This is why BF16 was designed with 8 exponent bits
fp32_bits = 0x41200000  # 10.0 in FP32
bf16_bits = fp32_bits >> 16  # Just take upper 16 bits!

# FP32 → FP16: Requires exponent remapping (slower)
# Can overflow/underflow if value outside FP16 range
```

```cpp
// CUDA conversion
__device__ __nv_bfloat16 float_to_bf16(float f) {
    return __float2bfloat16(f);  // Hardware instruction on Ampere+
}

__device__ __half float_to_fp16(float f) {
    return __float2half(f);
}
```

## Memory and Bandwidth

Both are 16-bit, so same memory footprint:

```python
# Both use 2 bytes per element
fp16 = torch.randn(1000, 1000, dtype=torch.float16)  # 2 MB
bf16 = torch.randn(1000, 1000, dtype=torch.bfloat16) # 2 MB
fp32 = torch.randn(1000, 1000, dtype=torch.float32)  # 4 MB
```

For memory-bound ops (most of inference), FP16 and BF16 give the same 2x speedup over FP32.

## Numerical Gotchas

```python
# BF16 precision loss in accumulation
total_bf16 = torch.tensor(0.0, dtype=torch.bfloat16)
for _ in range(1000):
    total_bf16 += 0.001  # Many small values lost to rounding
print(total_bf16)  # Much less than 1.0!

# Solution: Accumulate in FP32, store in BF16
# (This is what tensor cores do automatically)
```

## TL;DR

| Scenario | Recommendation |
|----------|----------------|
| Training on Ampere+/TPU | BF16 |
| Training on Volta/Turing | FP16 + GradScaler |
| Inference (general) | FP16 |
| Inference (maximum throughput, Hopper) | FP8 |
| When in doubt | BF16 (safer) |

BF16 trades precision for range. For neural nets, range matters more - a gradient of 100,000 that overflows to `inf` is catastrophic, but a weight of 0.123 becoming 0.125 is fine.
