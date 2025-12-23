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

### Building with Examples

Examples are disabled by default. To enable:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DBUILD_EXAMPLES=ON
```

### OpenImageIO Support (Optional)

For advanced image format support (EXR, HDR, TIFF 16/32-bit, etc.):

**Quick Setup (vcpkg - Recommended)**:
```powershell
# Install OpenImageIO and dependencies via vcpkg
vcpkg install openimageio:x64-windows

# Configure with vcpkg toolchain
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DUSE_OIIO=ON `
      -DCMAKE_TOOLCHAIN_FILE="C:\vcpkg\scripts\buildsystems\vcpkg.cmake"

cmake --build build --config Release
```

**Custom OIIO Build**:

If you have OpenImageIO built from source or installed elsewhere:

1. Copy the example configuration script:
   ```powershell
   Copy-Item cmake_configure_vs17_oiio.cmd.example cmake_configure_vs17.cmd
   ```

2. Edit `cmake_configure_vs17.cmd` and update all `<PATH_TO_*>` placeholders with your actual installation paths

3. Run the configuration script:
   ```powershell
   .\cmake_configure_vs17.cmd
   cmake --build build --config Release
   ```

The example script includes:
- Detailed path configuration for OpenImageIO and all dependencies
- Multiple installation examples (vcpkg, custom builds, Conan)
- Complete troubleshooting guide
- Dependency overview

**Required Dependencies** (when building OIIO from source):
- Imath 3.1+
- OpenEXR 3.1+
- libtiff 4.0+
- libpng 1.6+
- libjpeg or libjpeg-turbo
- zlib 1.2+

## Linux Build

```bash
# 1. Install ROCm
sudo apt install rocm-hip-sdk

# 2. Download dependencies
mkdir -p external/stb
wget -O external/stb/stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
wget -O external/stb/stb_image_write.h https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

# 3. Build (library only)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j$(nproc)

# 4. Build with examples
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
./build/texture_loader_example

# 5. Build with OpenImageIO (optional)
sudo apt install libopenimageio-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm -DUSE_OIIO=ON
cmake --build build -j$(nproc)
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
