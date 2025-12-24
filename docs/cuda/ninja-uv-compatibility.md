# Ninja + uv Compatibility

Ninja works fine with uv - they operate at different layers. uv handles Python package management while Ninja is the underlying build system that gets invoked during compilation.

## Installation

```bash
uv pip install ninja
# or system-level: apt install ninja-build / brew install ninja
```

PyTorch automatically detects Ninja and uses it for JIT-compiled CUDA extensions via `torch.utils.cpp_extension.load()` or `load_inline()`. The speedup comes from parallel compilation of multiple source files.

## Alternative Approaches

### Triton

If you're writing custom kernels, Triton lets you write GPU code in Python syntax. Much nicer development experience than raw CUDA, and it's what PyTorch 2.0's `torch.compile` uses under the hood. Compilation is handled automatically.

### Pre-compilation via setuptools

Build your extension as a proper package with `setup.py` using `CUDAExtension`. This way you compile once at install time rather than JIT:

```python
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[CUDAExtension('my_ext', ['my_ext.cu'])],
    cmdclass={'build_ext': BuildExtension}
)
```

### CuPy

For operations that map well to NumPy semantics, CuPy can be simpler than writing raw CUDA. Also has `@cupy.fuse` for kernel fusion.

### torch.compile

If your goal is performance rather than custom kernels specifically, PyTorch 2.x's compiler can often eliminate the need for hand-written CUDA entirely.
