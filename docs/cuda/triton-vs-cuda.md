# Triton vs CUDA

Comparison of Triton and CUDA for writing GPU kernels.

## Triton Pros

- **Much faster development** - Python syntax, no header files, no manual memory management. A kernel that takes 200 lines of CUDA might be 40 lines of Triton.
- **Automatic tiling and memory coalescing** - You specify block sizes and Triton handles a lot of the optimization that you'd hand-tune in CUDA.
- **No CUDA boilerplate** - No kernel launch configs, no explicit shared memory declarations, no `__syncthreads()` placement debugging.
- **Easier autotuning** - Built-in `@triton.autotune` decorator lets you search over block sizes and other params declaratively.
- **Fuses with torch.compile** - Triton kernels integrate cleanly into PyTorch's compilation stack.
- **Portable-ish** - Triton has AMD GPU support now (via ROCm), whereas CUDA locks you to NVIDIA.

## Triton Cons

- **Less control** - You can't drop down to PTX, can't do warp-level primitives directly, limited control over register usage.
- **Performance ceiling** - For the last 5-10% of performance, hand-tuned CUDA still wins. Triton's compiler makes reasonable choices, not optimal ones.
- **Debugging is harder** - When something goes wrong, the abstraction layers make it trickier to understand what's happening at the hardware level.
- **Smaller ecosystem** - Fewer examples, less Stack Overflow coverage, fewer battle-tested patterns.
- **Some operations are awkward** - Anything involving irregular memory access patterns or complex synchronization is harder to express.

## CUDA Pros

- **Full control** - Warp shuffles, tensor cores, async copies, explicit shared memory banking - everything is accessible.
- **Mature tooling** - Nsight Compute, cuda-memcheck, nvprof all work perfectly.
- **Maximum performance** - When you need it, nothing else gets as fast.
- **Huge ecosystem** - Every optimization trick has been documented somewhere.

## CUDA Cons

- **Slow iteration** - Compile times, verbose code, easy to introduce subtle bugs.
- **Expertise required** - You need to understand memory hierarchies, occupancy, bank conflicts, etc. to write good CUDA.
- **Vendor lock-in** - NVIDIA only.

## Rule of Thumb

Start with Triton. If profiling shows you're leaving significant performance on the table and the kernel is hot enough to matter, consider rewriting in CUDA. For most ML workloads, Triton gets you 90%+ of theoretical peak, which is good enough.
