# CUDA Auto-tuning

Tools and approaches for auto-tuning CUDA kernels.

## NVIDIA-provided

### cuBLAS/cuDNN heuristics

These libraries do internal autotuning. For cuDNN specifically, you can enable benchmarking mode (`torch.backends.cudnn.benchmark = True`) which tests different algorithm implementations and caches the fastest.

### CUTLASS Profiler

CUTLASS (NVIDIA's template library for matrix ops) has a profiler that searches over tile sizes, warp arrangements, and pipeline stages. Good for GEMM-heavy workloads.

## General-purpose Autotuners

### OpenTuner

MIT's framework-agnostic autotuner. You define the parameter space (block sizes, unroll factors, etc.) and it searches using ensembles of techniques (genetic algorithms, differential evolution, bandit methods). Works but requires you to wire up the CUDA compilation yourself.

### Kernel Tuner

Python library specifically for GPU kernel tuning. Nice API:

```python
from kernel_tuner import tune_kernel

tune_params = {
    "BLOCK_SIZE_X": [16, 32, 64],
    "BLOCK_SIZE_Y": [1, 2, 4, 8],
    "UNROLL": [1, 2, 4]
}

results = tune_kernel("my_kernel", "my_kernel.cu",
                      problem_size, args, tune_params)
```

Supports CUDA, OpenCL, and HIP. Probably the most accessible option.

### ATF (Auto-Tuning Framework)

From TU Dresden. More research-oriented but handles complex search spaces well.

## ML-compilation Approaches

### TVM/AutoTVM

If you're willing to express your kernel in TVM's tensor expression language, AutoTVM does learned cost model tuning. Significant investment to adopt but powerful for repeated deployment.

### Ansor (TVM)

Successor to AutoTVM, does hierarchical search over program sketches. Better for novel operators.

## Template-based

### CUTLASS

You select from pre-defined tile configurations. Not search-based but the templates are highly optimized.

### Triton's autotune

Really just grid search over the params you specify. Simple but effective.

## Pragmatic Approach

For most cases: parameterize your kernel with preprocessor macros for block sizes and unroll factors, then use Kernel Tuner or a simple Python script that compiles and benchmarks each configuration. The search space for a single kernel is usually small enough that exhaustive search is tractable.
