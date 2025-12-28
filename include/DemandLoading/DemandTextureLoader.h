#pragma once

#include "DemandLoading/DeviceContext.h"
#include "DemandLoading/Ticket.h"
#include <hip/hip_runtime.h>
#include <string>
#include <memory>
#include <vector>
#include <cstdint>

namespace hip_demand {

// Error codes
enum class LoaderError {
    Success = 0,
    InvalidTextureId,
    MaxTexturesExceeded,
    FileNotFound,
    ImageLoadFailed,
    OutOfMemory,
    InvalidParameter,
    HipError
};

const char* getErrorString(LoaderError error);

// Forward declarations
class TextureRegistry;
class RequestBuffer;
class ImageReader;

// Configuration options
struct LoaderOptions {
    size_t maxTextureMemory = 2ULL * 1024 * 1024 * 1024;  // 2 GB default
    size_t maxTextures = 4096;
    size_t maxRequestsPerLaunch = 1024;
    bool enableEviction = true;
    unsigned int maxThreads = 0;  // 0 = auto
};

// Texture descriptor
struct TextureDesc {
    hipTextureAddressMode addressMode[2] = {hipAddressModeWrap, hipAddressModeWrap};
    hipTextureFilterMode filterMode = hipFilterModeLinear;
    hipTextureFilterMode mipmapFilterMode = hipFilterModeLinear;
    bool normalizedCoords = true;
    bool sRGB = false;
    bool generateMipmaps = true;  // Generate mipmaps for better quality
    unsigned int maxMipLevel = 0;  // 0 = auto-generate all levels
};

inline bool operator==(const TextureDesc& a, const TextureDesc& b) {
    return (a.addressMode[0] == b.addressMode[0] &&
            a.addressMode[1] == b.addressMode[1] &&
            a.filterMode == b.filterMode &&
            a.mipmapFilterMode == b.mipmapFilterMode &&
            a.normalizedCoords == b.normalizedCoords &&
            a.sRGB == b.sRGB &&
            a.generateMipmaps == b.generateMipmaps &&
            a.maxMipLevel == b.maxMipLevel);
}

// Texture information returned after creation
struct TextureHandle {
    uint32_t id = 0;
    bool valid = false;
    int width = 0;
    int height = 0;
    int channels = 0;
    LoaderError error = LoaderError::Success;
};

class DemandTextureLoader {
public:
    explicit DemandTextureLoader(const LoaderOptions& options = LoaderOptions());
    ~DemandTextureLoader();

    // Disable copy
    DemandTextureLoader(const DemandTextureLoader&) = delete;
    DemandTextureLoader& operator=(const DemandTextureLoader&) = delete;

    // Create a texture from file (not loaded until requested)
    TextureHandle createTexture(const std::string& filename, 
                                const TextureDesc& desc = TextureDesc());
    
    // Create a texture from memory
    TextureHandle createTextureFromMemory(const void* data, 
                                         int width, int height, int channels,
                                         const TextureDesc& desc = TextureDesc());

    // Prepare for launch (updates device context)
    void launchPrepare(hipStream_t stream = 0);

    // Get device context to pass to kernel
    DeviceContext getDeviceContext() const;

    // Process texture requests after kernel launch
    // Returns number of textures loaded
    size_t processRequests(hipStream_t stream = 0);

    // Asynchronously process texture requests on a background thread and return a Ticket that can
    // be waited on. Uses the provided stream for any device copies performed during processing.
    Ticket processRequestsAsync(hipStream_t stream = 0);

    // Statistics
    size_t getResidentTextureCount() const;
    size_t getTotalTextureMemory() const;
    size_t getRequestCount() const;
    bool hadRequestOverflow() const;
    LoaderError getLastError() const;

    // Eviction control
    void enableEviction(bool enable);
    void setMaxTextureMemory(size_t bytes);
    size_t getMaxTextureMemory() const;

    // Utility
    void unloadTexture(uint32_t textureId);
    void unloadAll();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hip_demand
