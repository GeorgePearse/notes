# Build Backends for CUDA Extensions

Modern Python build backends for CUDA extension compilation.

## The Challenge

PyTorch's `BuildExtension` is a setuptools command class. `uv build` is just a frontend - it invokes whatever build backend you specify in `pyproject.toml`.

## Hatchling

Hatchling doesn't natively support setuptools command classes, so you can't directly use `torch.utils.cpp_extension.BuildExtension`. You'd need to use a build hook, which gets messy. Not the smoothest path.

## scikit-build-core (Recommended)

CMake-based, first-class CUDA support, works cleanly with uv:

```toml
[build-system]
requires = ["scikit-build-core", "torch"]
build-backend = "scikit_build_core.build"
```

Then write a `CMakeLists.txt` that finds CUDA and torch:

```cmake
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_library(my_ext MODULE my_ext.cu)
target_link_libraries(my_ext PRIVATE torch CUDA::cudart)
```

scikit-build-core handles cross-platform CUDA detection better than the PyTorch helpers.

## meson-python

Meson also has solid CUDA support and is faster than CMake. Slightly less common in the PyTorch ecosystem though.

## Setuptools with pyproject.toml

If you want minimal friction, you can still use setuptools as your backend but configure it via `pyproject.toml`. This lets you keep using `BuildExtension` while staying modern:

```toml
[build-system]
requires = ["setuptools", "torch", "ninja"]
build-backend = "setuptools.build_meta"
```
