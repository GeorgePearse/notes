# Nsight Profiling Tools

NVIDIA's profiling tools for understanding GPU performance.

## The Two Main Tools

| Tool | Level | Question It Answers |
|------|-------|---------------------|
| **Nsight Systems** | System-wide | Where is time spent? What's the bottleneck? |
| **Nsight Compute** | Kernel-level | Why is this specific kernel slow? |

Start with Systems, drill down with Compute.

## Nsight Systems

### Basic Usage

```bash
# Profile a training script
nsys profile -o report python train.py

# With more options
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    -o training_profile \
    python train.py

# Open in GUI
nsys-ui report.nsys-rep
```

### What You See

```
Timeline view:
├── CPU threads
│   ├── Python main thread
│   ├── Data loader workers
│   └── CUDA runtime calls
├── CUDA API
│   ├── cudaLaunchKernel
│   ├── cudaMemcpy
│   └── cudaStreamSynchronize
├── GPU Kernels
│   ├── ampere_sgemm_128x64_tn
│   ├── cudnn_convolution_forward
│   └── elementwise_kernel
└── GPU Memory
    ├── Allocations
    └── Transfers
```

### Common Patterns to Look For

**GPU Idle (gaps in timeline)**
```
CPU: ████████░░░░░░░░████████░░░░░░░░████████
GPU: ░░░░████████░░░░░░░░████████░░░░░░░░████
     ^          ^
     Gaps = GPU waiting for CPU
```
Fix: Overlap data loading, use CUDA graphs, reduce Python overhead

**Memory Transfer Bottleneck**
```
GPU: █H2D██kernel██D2H█H2D██kernel██D2H█
     ^^^^               ^^^^
     Host-to-device transfers dominating
```
Fix: Pin memory, prefetch, keep data on GPU

**Small Kernels**
```
GPU: █░█░█░█░█░█░█░█░█░█░  (tiny kernels with gaps)
```
Fix: Kernel fusion, CUDA graphs, torch.compile

### Adding Custom Markers (NVTX)

```python
import torch
import torch.cuda.nvtx as nvtx

# Simple range
with nvtx.range("forward"):
    output = model(input)

# Nested ranges
with nvtx.range("training_step"):
    with nvtx.range("forward"):
        output = model(input)
        loss = criterion(output, target)

    with nvtx.range("backward"):
        loss.backward()

    with nvtx.range("optimizer"):
        optimizer.step()

# Or use decorator
@nvtx.annotate("my_function", color="blue")
def my_function():
    pass
```

These appear as labeled regions in the timeline.

### PyTorch Profiler Integration

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        output = model(batch)
        loss.backward()
        optimizer.step()
        prof.step()

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Nsight Compute

For deep-diving into individual kernels.

### Basic Usage

```bash
# Profile specific kernel
ncu --target-processes all python train.py

# Focus on specific kernel
ncu --kernel-name "ampere_sgemm" python train.py

# Full metrics (slow but complete)
ncu --set full -o kernel_analysis python train.py

# Open report
ncu-ui kernel_analysis.ncu-rep
```

### Key Metrics

**Compute Throughput**
```
SM Throughput: 85%     # Good - compute bound
Memory Throughput: 95% # High - memory bound

Target: One of these near 100%
If both low: latency bound (occupancy, dependencies)
```

**Occupancy**
```
Achieved Occupancy: 45%
Theoretical Occupancy: 100%

Low occupancy reasons:
- Too many registers per thread
- Too much shared memory
- Block size not optimal
```

**Memory Analysis**
```
Global Memory Throughput: 1200 GB/s (vs 2039 GB/s peak on A100)
L2 Cache Hit Rate: 35%
Shared Memory Efficiency: 90%

Low throughput: Uncoalesced access, strided patterns
Low cache hit: Poor data locality
```

### Roofline Analysis

```
                    Compute
                    Bound
              ___________
             /
            /  Your kernel ●
           /
          /   ← Ridge point
         /
Memory  /
Bound  /

X-axis: Arithmetic Intensity (FLOPs / Bytes)
Y-axis: Performance (FLOPS)
```

If kernel is:
- Below the roof: Room for optimization
- On compute roof: Compute bound (optimize math)
- On memory roof: Memory bound (optimize access)

## Quick Profiling Commands

```bash
# Quick timeline
nsys profile -o quick python script.py

# CUDA API trace only
nsys profile --trace=cuda python script.py

# With Python backtrace (find which code launches which kernel)
nsys profile --trace=cuda,nvtx --python-sampling=true python script.py

# Memory profiling
nsys profile --cuda-memory-usage=true python script.py

# Multi-GPU
nsys profile --gpu-metrics-device=all python script.py
```

## Interpreting Common Issues

### Issue: Low GPU Utilization

**Symptoms in Nsight Systems:**
```
GPU utilization: 30%
Timeline shows gaps between kernels
```

**Causes & Fixes:**
1. **Data loading** - Use more workers, pin_memory=True
2. **Python overhead** - Use torch.compile, CUDA graphs
3. **Sync points** - Remove unnecessary .item(), print(tensor)
4. **Small batches** - Increase batch size

### Issue: Memory Bound Kernels

**Symptoms in Nsight Compute:**
```
Memory throughput: 95% of peak
Compute throughput: 40%
```

**Causes & Fixes:**
1. **Low arithmetic intensity** - Fuse operations
2. **Uncoalesced access** - Ensure contiguous tensors
3. **Redundant loads** - Use shared memory tiling

### Issue: Slow Convolutions

**Symptoms:**
```
cudnn_convolution taking 60% of time
```

**Debug:**
```python
# Check which algorithm cuDNN selected
torch.backends.cudnn.benchmark = True  # Enable autotuning

# Profile to see algorithm choice
# Look for: implicit_gemm, winograd, fft, etc.
```

## Environment Variables for Debug

```bash
# CUDA launch blocking (synchronous, for debugging)
export CUDA_LAUNCH_BLOCKING=1

# Detailed cuDNN logging
export CUDNN_LOGINFO_DBG=1
export CUDNN_LOGDEST_DBG=stdout

# NCCL debug (for distributed)
export NCCL_DEBUG=INFO
```

## Profiling Checklist

1. **Start with Nsight Systems**
   - Where is time spent? (kernels, memory, CPU?)
   - Are there gaps in GPU timeline?
   - Is data loading overlapped?

2. **Identify hotspot kernels**
   - Which kernels take most time?
   - Are they expected to be slow?

3. **Deep dive with Nsight Compute**
   - Is kernel compute or memory bound?
   - What's limiting occupancy?
   - Are memory accesses efficient?

4. **Iterate**
   - Make one change at a time
   - Re-profile to verify improvement
   - Compare before/after timelines
