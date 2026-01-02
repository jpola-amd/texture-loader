# HIP Demand Texture Loader

**Efficient on-demand texture streaming for AMD GPUs using HIP/ROCm.** Inspired by NVIDIA OptiX Demand Loading but simplified for whole-texture loading. Features multi-pass rendering, LRU eviction, automatic mipmap generation, and Visual Studio 2022 support via HIP Module API.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ROCm](https://img.shields.io/badge/ROCm-6.4+-green.svg)](https://rocmdocs.amd.com/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)](https://en.cppreference.com/w/cpp/17)

## Features

- ✅ **On-Demand Loading**: Stream textures only when needed by shaders
- ✅ **Memory Management**: LRU eviction keeps memory usage under control
- ✅ **Mipmap Generation**: Automatic mipmap creation with box filter
- ✅ **Multi-Pass Rendering**: Automatically handles missing textures across passes
- ✅ **Thread-Safe**: Concurrent texture loading and rendering
- ✅ **Visual Studio 2022**: Full support via HIP Module API
- ✅ **Error Handling**: Comprehensive HIP error checking on all API calls
- ✅ **Request Overflow Detection**: Monitors and reports request buffer status
- ✅ **Flexible Image Loading**: Built-in stb_image support with optional OpenImageIO for advanced formats (EXR, HDR, etc.)
- ✅ **Unit Tests**: Comprehensive test suite using Google Test

## Quick Start

### Windows (Visual Studio 2022)

**Using vcpkg (Recommended)**:
```powershell
# 1. Install vcpkg if you haven't already
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# 2. Install OpenImageIO (provides all dependencies)
cd C:\vcpkg
.\vcpkg install openimageio:x64-windows

# 3. Build with vcpkg toolchain
cd <your-project-dir>
cmake -B build -S . -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake `
      -DBUILD_EXAMPLES=ON -DUSE_OIIO=ON
cmake --build build --config Release
.\build\Release\texture_loader_example.exe
```

**Basic Build (without OpenImageIO)**:
```powershell
# Set HIP_PATH
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\6.4"

# Build (library only)
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

# Build with examples
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DBUILD_EXAMPLES=ON
cmake --build build --config Release
.\build\Release\texture_loader_example.exe
```

**Advanced: Custom OpenImageIO Build**:
```powershell
# For manually built OIIO (copy and edit example script)
Copy-Item cmake_configure_vs17_oiio.cmd.example cmake_configure_vs17.cmd
# Edit cmake_configure_vs17.cmd with your paths
.\cmake_configure_vs17.cmd
```

### Linux

```bash
# Library only
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# With examples
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
cmake --build build
./build/texture_loader_example

# With unit tests
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure

# With OpenImageIO
sudo apt install libopenimageio-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OIIO=ON -DBUILD_EXAMPLES=ON
cmake --build build
```

See [BUILD.md](BUILD.md) for detailed build instructions.

## Architecture

### Core Components

1. **DemandTextureLoader**: Main API for creating and managing demand-loaded textures
2. **DeviceContext**: GPU-side data structure passed to kernels for texture access
3. **TextureSampling.h**: Device-side sampling functions that check residency and record requests
4. **Request Buffer**: Tracks which textures were requested during kernel execution

### How It Works

```
┌─────────────────┐
│  Create Textures│  (Not loaded yet)
│  (metadata only)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Launch Prepare  │  Update device context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Render Kernel  │  Sample textures
│                 │  → Record missing IDs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Process Requests │  Load requested textures
│                 │  → Create HIP texture objects
└────────┬────────┘
         │
         ▼
    (Repeat until all textures resident)
```

## Building

See [BUILD.md](BUILD.md) for comprehensive build instructions including:
- Prerequisites and dependencies
- Windows Visual Studio 2022 setup
- Linux build instructions  
- HIP Module API configuration
- Troubleshooting guide

## Usage

### Basic Example

```cpp
#include "DemandLoading/DemandTextureLoader.h"
#include "DemandLoading/TextureSampling.h"

// Create loader
hip_demand::LoaderOptions options;
options.maxTextureMemory = 2ULL * 1024 * 1024 * 1024;  // 2 GB
hip_demand::DemandTextureLoader loader(options);

// Create textures (not loaded yet)
auto tex1 = loader.createTexture("texture1.png");
auto tex2 = loader.createTexture("texture2.png");

// Multi-pass rendering loop
while (true) {
    // Prepare for launch
    loader.launchPrepare();
    auto ctx = loader.getDeviceContext();
    
    // Launch your kernel with device context
    myRenderKernel<<<grid, block>>>(ctx, tex1.id, tex2.id, ...);
    hipDeviceSynchronize();
    
    // Process missing texture requests
    size_t loaded = loader.processRequests();
    
    // Done when no new textures loaded
    if (loaded == 0) break;
}
```

### Device-Side Sampling

```cpp
__global__ void myKernel(hip_demand::DeviceContext ctx, uint32_t texId) {
    float u = ..., v = ...;
    float4 color;
    
    // Sample texture (automatically handles missing textures)
    bool resident = hip_demand::tex2D(ctx, texId, u, v, color);
    
    if (!resident) {
        // Texture not loaded, color is set to fallback (magenta)
        // Will be loaded and re-rendered in next pass
    }
}
```

### OpenImageIO Format Example

When built with `-DUSE_OIIO=ON`, an additional example demonstrates advanced format support:

```bash
# Test various image formats
.\build\Release\oiio_formats_example.exe texture.exr texture.hdr texture.tif

# Example output:
# ✓ OpenImageIO support ENABLED
# Testing: texture.exr
# Resolution:    2048 x 1024
# Channels:      4
# Mip Levels:    11
# Pixel Format:  FLOAT32 (float)
# Base Size:     8192 KB
# ✓ Successfully read base mip level
# Center pixel:  RGBA(128, 64, 32, 255)
```

See [examples/oiio_formats.cpp](examples/oiio_formats.cpp) for:
- Loading EXR, HDR, TIFF 16/32-bit files
- Format detection and statistics
- Mip level access
- Direct ImageSource API usage
- DemandTextureLoader integration

## API Reference

### Core Classes

#### DemandTextureLoader

Main interface for texture management:

```cpp
// Create loader
hip_demand::LoaderOptions options;
options.maxTextureMemory = 512 * 1024 * 1024;  // 512 MB
options.maxRequestsPerLaunch = 1920 * 1080;    // Per-pixel requests
hip_demand::DemandTextureLoader loader(options);

// Create textures
auto tex = loader.createTexture("texture.png");
auto mem = loader.createTextureFromMemory(data, width, height, 4);

// Rendering loop
loader.launchPrepare(stream);
auto ctx = loader.getDeviceContext();
myKernel<<<grid, block>>>(ctx, tex.id);
size_t loaded = loader.processRequests(stream);
```

#### Key Methods

| Method | Description |
|--------|-------------|
| `createTexture(filename, desc)` | Create texture from file |
| `createTextureFromMemory(data, w, h, c, desc)` | Create from memory |
| `launchPrepare(stream)` | Update device context before kernel |
| `getDeviceContext()` | Get context to pass to kernel |
| `processRequests(stream)` | Load requested textures after kernel |
| `getResidentTextureCount()` | Number of loaded textures |
| `getTotalTextureMemory()` | GPU memory usage |
| `hadRequestOverflow()` | Check if buffer overflowed |

### Configuration

```cpp
struct LoaderOptions {
    size_t maxTextureMemory = 2ULL * 1024 * 1024 * 1024;  // 2 GB
    size_t maxTextures = 4096;
    size_t maxRequestsPerLaunch = 1024;  // Set to width×height for best results
    bool enableEviction = true;
};

struct TextureDesc {
    hipTextureAddressMode addressMode = hipAddressModeWrap;
    hipTextureFilterMode filterMode = hipFilterModeLinear;
    bool normalizedCoords = true;
    bool sRGB = false;
    bool generateMipmaps = true;
    unsigned int maxMipLevel = 0;  // 0 = auto
};
```

### Device-Side API

```cpp
#include "DemandLoading/TextureSampling.h"

__global__ void myKernel(hip_demand::DeviceContext ctx, uint32_t texId) {
    float u = ..., v = ...;
    float4 color;
    
    // Sample with automatic request recording
    bool resident = hip_demand::tex2D(ctx, texId, u, v, color);
    
    if (!resident) {
        // color = magenta (1,0,1,1) - texture not loaded
        // Will be loaded in next pass
    }
}
```

### Error Handling

All HIP API calls are checked. Query errors:

```cpp
if (loader.getLastError() != hip_demand::LoaderError::Success) {
    std::cerr << "Error: " << hip_demand::getErrorString(loader.getLastError()) << "\n";
}
```

Error codes: `Success`, `InvalidTextureId`, `MaxTexturesExceeded`, `FileNotFound`, `ImageLoadFailed`, `OutOfMemory`, `InvalidParameter`, `HipError`

## Performance Considerations

### Request Buffer Sizing

The `maxRequestsPerLaunch` parameter is critical for performance:

```cpp
// Worst case: every pixel requests a texture
options.maxRequestsPerLaunch = renderWidth * renderHeight;

// Example for 1920×1080:
options.maxRequestsPerLaunch = 1920 * 1080;  // ~8 MB GPU memory
```

**Too small** → Buffer overflow → Multiple passes to discover all textures  
**Optimal** → width×height → Single pass texture discovery  
**Too large** → Wastes GPU memory

### Memory Management

- Set `maxTextureMemory` to 50-70% of GPU memory
- Enable `enableEviction` for large texture sets
- Disable eviction if working set fits in memory (faster)
- Monitor with `getTotalTextureMemory()` and `getResidentTextureCount()`

### Mipmap Strategy

```cpp
desc.generateMipmaps = true;   // Better quality, more memory
desc.generateMipmaps = false;  // Less memory, aliasing artifacts
desc.maxMipLevel = 4;          // Limit mip levels to save memory
```

Mipmaps increase memory by ~33% but significantly improve rendering quality.

## Implementation Notes

### Architecture Comparison

| Feature | OptiX Demand Loading | HIP Demand Loader |
|---------|---------------------|-------------------|
| Granularity | 64KB tiles | Whole textures |
| Hardware | CUDA sparse textures | Standard HIP textures |
| Memory Efficiency | Excellent (sub-texture) | Good (whole texture) |
| Complexity | High | Low |
| Best For | Huge textures (8K+) | Many medium textures |
| AMD GPU Support | ❌ No | ✅ Yes |

### Key Differences from OptiX

**OptiX Approach**:
- Tile-based sparse textures (requires hardware support)
- 64KB tiles loaded on demand
- Virtual texture addressing
- CUDA-only

**Our Approach**:
- Whole-texture loading (simpler, more compatible)
- LRU eviction manages memory
- Standard HIP texture objects
- Works on all AMD GPUs with HIP support

### Image Loading Backends

The loader supports two image loading backends:

**Built-in (stb_image)**:
- Default backend, always available
- Supports: PNG, JPG, BMP, TGA, PSD, GIF
- Lightweight, no external dependencies
- UINT8 format only

**Optional (OpenImageIO)**:
- Enable with `-DUSE_OIIO=ON` during CMake configuration
- Supports: EXR, HDR, TIFF (16/32-bit), and 100+ formats
- Production-grade format handling
- Automatic format detection and conversion
- Thread-safe with statistics tracking
- Used automatically when available, falls back to stb_image

```bash
# Build with OpenImageIO support
cmake -S . -B build -DUSE_OIIO=ON
cmake --build build
```

**When to use OpenImageIO**:
- HDR/EXR workflows (VFX, film production)
- 16/32-bit per channel textures
- Advanced format requirements
- Professional production pipelines

The loader automatically tries OIIO first (if enabled), then falls back to stb_image if OIIO fails or isn't available.

### Visual Studio 2022 Support

This project uses the **HIP Module API** for Visual Studio compatibility:

- Device code compiled separately with `hipcc` to `.co` files
- Runtime loading via `hipModuleLoad` and `hipModuleLaunchKernel`
- Works with any CMake generator (VS, Ninja, Make)
- Custom `cmake/FindHIP.cmake` module handles compilation

See [BUILD.md](BUILD.md) for details on the Module API approach.

### Thread Safety

- `processRequests()`: Fully thread-safe with mutex protection
- Multiple loaders can coexist (use separate streams)
- Texture loading gathers metadata under lock, loads outside lock
- No race conditions in request processing

### Error Handling

All HIP API calls are checked with progressive cleanup:

```cpp
hipError_t err = hipMallocArray(&array, ...);
if (err != hipSuccess) {
    // Clean up partial allocations
    if (texObj) hipDestroyTextureObject(texObj);
    return false;
}
```

No memory leaks even on error paths.

## Project Status

### Completed Features

- ✅ Core demand loading system
- ✅ Multi-pass rendering with request recording
- ✅ LRU eviction with accurate memory tracking
- ✅ Automatic mipmap generation (box filter)
- ✅ Thread-safe texture loading
- ✅ Comprehensive HIP error checking
- ✅ Request buffer overflow detection
- ✅ Visual Studio 2022 support (Module API)
- ✅ File and memory-based texture creation
- ✅ Flexible addressing and filtering modes

### Future Enhancements

- [ ] Parallel texture loading (multi-threaded CPU)
- [ ] Texture compression (BC/DXT formats)
- [ ] OpenEXR/HDR texture support
- [ ] Texture atlasing for small textures
- [ ] Advanced eviction (frequency-based, priority-based)
- [ ] Async texture creation (overlap with rendering)
- [ ] UDIM texture support for production rendering
- [ ] Preloading hints API

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Credits

Inspired by **NVIDIA OptiX Toolkit Demand Loading**:  
https://github.com/NVIDIA/optix-toolkit/tree/master/DemandLoading

Reference implementation studied for architectural patterns, adapted for whole-texture loading on HIP/ROCm.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

For bugs or feature requests, open an issue on GitHub.

## Contact

For questions or support:
- Open an issue on GitHub
- Check [BUILD.md](BUILD.md) for troubleshooting

---

**Built with ❤️ for the AMD GPU community**
