# Build Instructions

Complete guide for building the HIP Demand Texture Loader on Windows and Linux.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Windows Build](#windows-build-visual-studio-2022)
- [Linux Build](#linux-build)
- [HIP Module API](#hip-module-api)
- [GPU Architecture](#gpu-architecture-configuration)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### All Platforms

- **ROCm/HIP**: Version 6.0+ (tested with 6.4)
- **CMake**: Version 3.21+
- **C++17 Compiler**:
  - Windows: Visual Studio 2022
  - Linux: GCC 9+ or Clang 10+
- **stb_image**: Header-only library

### Supported GPUs

- RDNA2 (gfx1030): RX 6000 series
- RDNA3 (gfx1100): RX 7000 series
- CDNA2 (gfx90a): MI200 series
- Vega (gfx900/906): RX Vega, Radeon VII

## Windows Build (Visual Studio 2022)

### Quick Start

```powershell
# 1. Set HIP_PATH
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\6.4"

# 2. Download dependencies
New-Item -ItemType Directory -Force -Path external\stb
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" `
                  -OutFile "external\stb\stb_image.h"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h" `
                  -OutFile "external\stb\stb_image_write.h"

# 3. Build
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

# 4. Run
.\build\Release\texture_loader_example.exe
```

### Custom HIP Path

If using a custom ROCm installation:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DHIP_PATH="E:\Custom\Path\To\hip\win64"
```

## Linux Build

```bash
# 1. Install ROCm
sudo apt install rocm-hip-sdk

# 2. Download dependencies
mkdir -p external/stb
wget -O external/stb/stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
wget -O external/stb/stb_image_write.h https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

# 3. Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j$(nproc)

# 4. Run
./build/texture_loader_example
```

## HIP Module API

### Why Module API?

Visual Studio 2022 doesn't support HIP language in CMake. We use Module API to:
- Compile device code separately with `hipcc`
- Load compiled kernels at runtime
- Support any CMake generator

### Build Process

```
.hip source → hipcc → .co (code object) → Runtime loading
```

### Kernel Requirements

Kernels must use `extern "C"`:

```cpp
// Correct
extern "C" __global__ void myKernel(int* data) { }

// Wrong - name mangling
__global__ void myKernel(int* data) { }
```

### Runtime Loading

```cpp
hipModule_t module;
hipModuleLoad(&module, "kernel.co");
hipFunction_t kernel;
hipModuleGetFunction(&kernel, module, "myKernel");
hipModuleLaunchKernel(kernel, ...);
```

## GPU Architecture Configuration

### Find Your Architecture

```bash
# Linux
rocminfo | grep gfx

# Windows
& "$env:HIP_PATH\bin\rocminfo.exe" | Select-String "gfx"
```

### Architecture Table

| GPU | Architecture | CMake Value |
|-----|-------------|-------------|
| RX 6000 series | RDNA2 | gfx1030 |
| RX 7000 series | RDNA3 | gfx1100 |
| RX Vega | Vega | gfx900 |
| Radeon VII | Vega 20 | gfx906 |
| MI100 | CDNA | gfx908 |
| MI200 | CDNA2 | gfx90a |

### Set Architecture

In `CMakeLists.txt`:

```cmake
hip_add_executable(
    TARGET render_kernel
    SOURCES examples/simple_render_kernel.hip
    ARCHITECTURES gfx1030 gfx1100  # Your GPU(s)
    OPTIONS -O3 --std=c++17
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include
)
```

## Troubleshooting

### "HIP not found"

**Solution**:
```powershell
# Windows - set HIP_PATH
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\6.4"

# Linux - check installation
ls /opt/rocm
```

### "amd_comgr not found"

**Solution**: FindHIP.cmake automatically tries multiple library names (amd_comgr_2, amd_comgr0604, amd_comgr). If still failing:

```powershell
# Check what's available
ls "$env:HIP_PATH\lib" | Select-String comgr
```

### "stb_image.h not found"

**Solution**:
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" `
                  -OutFile "external\stb\stb_image.h"
```

### "Failed to load HIP module"

**Solution**: Check .co file exists:
```powershell
ls build\Release\render_kernel.co
```

### "Failed to get kernel function"

**Solution**:
1. Verify kernel has `extern "C"`
2. Check name matches exactly (case-sensitive)
3. List symbols: `llvm-nm render_kernel.co`

### "Request buffer overflow"

**Solution**: Increase buffer size:
```cpp
options.maxRequestsPerLaunch = renderWidth * renderHeight;
```

### Slow Loading

**Optimize**:
```cpp
desc.generateMipmaps = false;  // Disable if not needed
desc.maxMipLevel = 4;          // Limit mip levels
```

## Build Options

### CMake Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HIP_PATH` | Auto | Path to HIP installation |
| `STB_INCLUDE_DIR` | `external/stb` | Path to stb headers |
| `CMAKE_BUILD_TYPE` | Release | Build configuration |

### Compiler Flags

```cmake
# Optimize for speed
OPTIONS -O3 --std=c++17 -ffast-math

# Debug with symbols
OPTIONS -O0 -g --std=c++17
```

### Clean Build

```powershell
# Windows
Remove-Item -Recurse -Force build
cmake -S . -B build -G "Visual Studio 17 2022" -A x64

# Linux
rm -rf build && cmake -S . -B build
```

## Advanced

### Multiple Architectures

```cmake
ARCHITECTURES gfx900 gfx906 gfx1030 gfx1100 gfx90a
```

Creates "fat binary" supporting multiple GPUs.

### Verbose Build

```powershell
cmake -S . -B build --trace-expand
cmake --build build --config Release --verbose
```

### Verify Installation

```powershell
& "$env:HIP_PATH\bin\hipcc.bat" --version
& "$env:HIP_PATH\bin\rocminfo.exe"
```

## Getting Help

Include in bug reports:
- ROCm version
- GPU model (from `rocminfo`)
- CMake version
- Full error message

See [README.md](README.md) for API reference and usage examples.
