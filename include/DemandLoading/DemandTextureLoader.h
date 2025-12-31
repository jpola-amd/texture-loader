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

// Eviction priority for textures
enum class EvictionPriority {
    Normal = 0,    // Default - standard LRU eviction
    Low = 1,       // Evict first (temporary/preview textures)
    High = 2,      // Evict last (important textures)
    KeepResident = 3  // Never evict (UI, hero textures)
};

// Configuration options
struct LoaderOptions {
    size_t maxTextureMemory = 2ULL * 1024 * 1024 * 1024;  // 2 GB default
    size_t maxTextures = 4096;
    size_t maxRequestsPerLaunch = 1024;
    bool enableEviction = true;
    unsigned int maxThreads = 0;  // 0 = auto
    unsigned int minResidentFrames = 3;  // Thrashing prevention: don't evict textures younger than this
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
    EvictionPriority evictionPriority = EvictionPriority::Normal;  // Eviction priority hint
};

inline bool operator==(const TextureDesc& a, const TextureDesc& b) {
    return (a.addressMode[0] == b.addressMode[0] &&
            a.addressMode[1] == b.addressMode[1] &&
            a.filterMode == b.filterMode &&
            a.mipmapFilterMode == b.mipmapFilterMode &&
            a.normalizedCoords == b.normalizedCoords &&
            a.sRGB == b.sRGB &&
            a.generateMipmaps == b.generateMipmaps &&
            a.maxMipLevel == b.maxMipLevel &&
            a.evictionPriority == b.evictionPriority);
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

    // Process texture requests after kernel launch using the provided device context
    // Returns number of textures loaded
    size_t processRequests(hipStream_t stream, const DeviceContext& deviceContext);

    // Asynchronously process texture requests on a background thread using the provided device context and stream.
    // Returns a Ticket that can be waited on.
    Ticket processRequestsAsync(hipStream_t stream, const DeviceContext& deviceContext);

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
    
    /// Update the eviction priority for a texture dynamically.
    /// Use this to adjust priorities based on camera distance, LOD importance, etc.
    void updateEvictionPriority(uint32_t textureId, EvictionPriority priority);

    // Utility
    void unloadTexture(uint32_t textureId);
    void unloadAll();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hip_demand
