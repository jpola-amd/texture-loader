# ImageSource Abstraction Layer

## Overview

The ImageSource abstraction layer provides a flexible interface for loading images from various sources and formats. It supports both built-in (stb_image) and optional advanced backends (OpenImageIO).

## Architecture

```
┌─────────────────────────┐
│   DemandTextureLoader   │
│  (High-level texture    │
│   management)           │
└────────────┬────────────┘
             │ uses
             ▼
┌─────────────────────────┐
│     ImageSource         │  ← Abstract interface
│   (Pure virtual class)  │
└────────────┬────────────┘
             │ implements
      ┌──────┴──────┐
      ▼             ▼
┌──────────┐  ┌──────────────┐
│ stb_image│  │ OIIOReader   │
│ (built-in)  │ (optional)   │
└──────────┘  └──────────────┘
```

## Components

### 1. ImageSource (Abstract Interface)

**File**: `include/ImageSource/ImageSource.h`

Pure virtual interface for image loading implementations:

```cpp
class ImageSource {
public:
    virtual ~ImageSource() = default;
    
    // Core operations
    virtual bool open(const std::string& filename) = 0;
    virtual void close() = 0;
    virtual bool isOpen() const = 0;
    
    // Image metadata
    virtual const TextureInfo& getInfo() const = 0;
    
    // Data access
    virtual bool readMipLevel(void* dest, int mipLevel, size_t bufferSize) = 0;
    virtual bool readBaseColor(float* color) = 0;
    
    // Statistics
    virtual uint64_t getBytesRead() const = 0;
    virtual double getReadTime() const = 0;
};
```

**Thread Safety**: All implementations must be thread-safe.

### 2. TextureInfo Structure

**File**: `include/ImageSource/TextureInfo.h`

Image metadata and format information:

```cpp
enum class PixelFormat {
    UINT8,      // 8-bit unsigned integer (most common)
    UINT16,     // 16-bit unsigned integer
    FLOAT16,    // 16-bit float (half precision)
    FLOAT32     // 32-bit float (full precision)
};

struct TextureInfo {
    int width = 0;
    int height = 0;
    int numChannels = 0;
    int numMipLevels = 0;
    PixelFormat format = PixelFormat::UINT8;
    bool isValid = false;
    
    // Helper methods
    size_t getBytesPerChannel() const;
    size_t getTextureSizeInBytes(int mipLevel = 0) const;
};
```

### 3. OIIOReader Implementation

**Files**: 
- `include/ImageSource/OIIOReader.h`
- `src/ImageSource/OIIOReader.cpp`

OpenImageIO-based implementation with production-grade features:

**Features**:
- Automatic format detection for 100+ image formats
- Thread-safe with internal mutex protection
- Caches all mip levels in memory for fast access
- Automatic UINT8 conversion from any source format
- Box filter mipmap generation
- Statistics tracking (bytes read, read time)

**Supported Formats**:
- HDR formats: EXR, HDR (Radiance)
- Standard formats: PNG, JPEG, TIFF, BMP, TGA
- Professional formats: DPX, Cineon, SGI, etc.
- And 100+ more via OIIO plugins

**Usage**:

```cpp
#include "ImageSource/ImageSource.h"
#include "ImageSource/OIIOReader.h"

// Create reader via factory function
std::unique_ptr<ImageSource> img = createImageSource("texture.exr");

// Open and read
if (img && img->open("texture.exr")) {
    const auto& info = img->getInfo();
    
    // Allocate buffer
    size_t bufferSize = info.getTextureSizeInBytes(0);
    std::vector<uint8_t> buffer(bufferSize);
    
    // Read base mip level
    img->readMipLevel(buffer.data(), 0, bufferSize);
    
    // Statistics
    printf("Bytes read: %llu\n", img->getBytesRead());
    printf("Read time: %.3f ms\n", img->getReadTime() * 1000.0);
    
    img->close();
}
```

## Integration with DemandTextureLoader

The loader uses a **fallback strategy**:

1. **OIIO Enabled** (`-DUSE_OIIO=ON`):
   - Try OpenImageIO first
   - Fall back to stb_image if OIIO fails
   - Transparent to user

2. **OIIO Disabled** (default):
   - Use stb_image only
   - Supports PNG, JPG, BMP, TGA, PSD, GIF

**Code Flow**:

```cpp
// In createTexture() and loadTexture()
#ifdef USE_OIIO
    try {
        std::unique_ptr<ImageSource> imgSrc = createImageSource(filename);
        if (imgSrc && imgSrc->open(filename)) {
            // Use OIIO
            const auto& texInfo = imgSrc->getInfo();
            // ... load data ...
        } else {
            // Fall back to stb_image
        }
    } catch (...) {
        // Fall back to stb_image on any exception
    }
#else
    // Use stb_image
#endif
```

## Building with OpenImageIO

### Windows (vcpkg)

```powershell
# Install OpenImageIO
vcpkg install openimageio

# Configure
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DUSE_OIIO=ON `
      -DCMAKE_TOOLCHAIN_FILE="C:\vcpkg\scripts\buildsystems\vcpkg.cmake"

# Build
cmake --build build --config Release
```

### Linux

```bash
# Install OIIO
sudo apt install libopenimageio-dev

# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OIIO=ON

# Build
cmake --build build -j$(nproc)
```

## When to Use OpenImageIO

**Use OIIO when**:
- Working with HDR/EXR workflows (VFX, film)
- Need 16/32-bit per channel support
- Require wide format compatibility
- Production pipeline requirements
- Color space management needed

**Use stb_image when**:
- Simple PNG/JPG textures sufficient
- Minimal dependencies preferred
- Target platform has no OIIO
- Memory footprint is critical
- Rapid prototyping

## Performance Characteristics

### Memory Usage

**OIIOReader**:
- Caches all mip levels in RAM
- Higher memory usage for large textures
- Faster repeated access (no disk I/O)
- ~1.33× base texture size (with mipmaps)

**stb_image**:
- Load on demand, no caching
- Minimal memory overhead
- Slower repeated loads (disk I/O)
- Only base level loaded initially

### Load Time

**First Load**:
- OIIO: Slower (loads all mips, format conversion)
- stb_image: Faster (single level, simple formats)

**Subsequent Access**:
- OIIO: Instant (cached in memory)
- stb_image: Same as first load (no caching)

## Extending ImageSource

To add a new image loading backend:

1. **Create Implementation**:
   ```cpp
   class MyReader : public ImageSource {
   public:
       bool open(const std::string& filename) override;
       // ... implement all virtual methods ...
   };
   ```

2. **Add Factory Function** (optional):
   ```cpp
   std::unique_ptr<ImageSource> createMyReader(const std::string& filename) {
       return std::make_unique<MyReader>();
   }
   ```

3. **Update CMakeLists.txt**:
   ```cmake
   option(USE_MY_READER "Use MyReader" OFF)
   if(USE_MY_READER)
       target_sources(hip_demand_texture PRIVATE
           src/ImageSource/MyReader.cpp
       )
       target_compile_definitions(hip_demand_texture PRIVATE USE_MY_READER)
   endif()
   ```

## Future Extensions

Possible ImageSource implementations:

- **EXRReader**: Dedicated OpenEXR with deep images
- **ProceduralSource**: Generate textures programmatically
- **NetworkSource**: Load from HTTP/cloud storage
- **DatabaseSource**: Load from database BLOBs
- **VideoSource**: Extract frames from video files
- **RawReader**: Support RAW camera formats

## API Stability

**Stable**:
- `ImageSource` interface (virtual methods)
- `TextureInfo` structure
- `PixelFormat` enum

**Experimental**:
- Specific implementations (OIIOReader)
- Factory function signature
- Statistics API

The abstract interface is stable; implementations may change between versions.
