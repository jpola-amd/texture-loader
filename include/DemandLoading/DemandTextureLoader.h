#pragma once

/// @file DemandTextureLoader.h
/// @brief Demand texture loading API for HIP applications.
/// @note This header requires HIP types. Include <hip/hip_runtime.h> before this header.

#include "DemandLoading/DeviceContext.h"
#include "DemandLoading/Ticket.h"
#include <string>
#include <memory>
#include <vector>
#include <cstdint>

namespace hip_demand {

class ImageSource;  // Forward declaration

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
    size_t maxTextures = 4096;  // 1..UINT32_MAX; UINT32_MAX is never a texture ID
    size_t maxRequestsPerLaunch = 1024;  // 1..UINT32_MAX
    bool enableEviction = true;
    unsigned int maxThreads = 0;  // Legacy pool size (0 = auto); decode/upload currently serialized
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
inline constexpr uint32_t InvalidTextureId = UINT32_MAX;

struct TextureHandle {
    uint32_t id = InvalidTextureId;
    bool valid = false;
    int width = 0;
    int height = 0;
    int channels = 0;
    LoaderError error = LoaderError::InvalidTextureId;
};

class DemandTextureLoader {
public:
    explicit DemandTextureLoader(const LoaderOptions& options = LoaderOptions());
    ~DemandTextureLoader();

    // Disable copy
    DemandTextureLoader(const DemandTextureLoader&) = delete;
    DemandTextureLoader& operator=(const DemandTextureLoader&) = delete;

    // Register a descriptor-specific sampler (pixels remain demand-loaded).
    // Equal complete descriptors for the same filename/source identity reuse
    // an ID, including at capacity. Every distinct descriptor consumes one ID.
    // Compatible variants share image backing; sRGB and mip-generation/limit
    // policies conservatively split storage. All descriptor fields participate
    // in sampler identity, including eviction priority.
    // These legacy handles are not owning leases: unload does not release
    // registration capacity, and IDs are never recycled.
    TextureHandle createTexture(const std::string& filename, 
                                const TextureDesc& desc = TextureDesc());

    // Create a texture from an ImageSource (pixels not loaded until requested).
    // Invalid metadata or an exception while opening/hashing fails registration.
    // Content hashes use the ImageSource identity contract plus validated native
    // dimensions/format/channels/mip metadata. Filename and content identities
    // occupy separate namespaces. Storage retains its selected source through
    // registration and active reads; an equivalent incoming alias need not be
    // retained. Source content/metadata must remain stable while registered.
    TextureHandle createTexture(std::shared_ptr<ImageSource> imageSource,
                                const TextureDesc& desc = TextureDesc());
    
    // Copy byte pixels into a fresh registration/storage identity on every call.
    TextureHandle createTextureFromMemory(const void* data, 
                                         int width, int height, int channels,
                                         const TextureDesc& desc = TextureDesc());

    // Prepare for launch (updates device context). This implementation uses a
    // serialized, device-quiescent publication/retirement baseline; table copies
    // complete before return. Callers must not concurrently submit consumers
    // while preparing or unloading, or overlap reuse of the request context.
    void launchPrepare(hipStream_t stream = 0);

    // Get device context to pass to kernel
    DeviceContext getDeviceContext() const;

    // Process texture requests after kernel launch using the provided device context
    // Returns the number of sampler IDs newly made resident, not backing uploads.
    size_t processRequests(hipStream_t stream, const DeviceContext& deviceContext);

    // Completes request readback before handing decode/upload to a background
    // worker. Runtime operations are serialized; a Ticket reports completion,
    // not per-sampler success.
    Ticket processRequestsAsync(hipStream_t stream, const DeviceContext& deviceContext);

    // Statistics
    size_t getResidentTextureCount() const;
    // Array texel payload, counted once per shared allocation, including failed
    // cleanup until freed. Excludes opaque HIP/sampler overhead and host caches;
    // this is neither a physical-VRAM measurement nor a hard budget guarantee.
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
    /// Existing IDs never merge. Lookup of equal keys chooses the oldest ID.
    /// Storage uses the highest sibling priority; any registered KeepResident
    /// variant protects its backing even before that sampler is requested.
    void updateEvictionPriority(uint32_t textureId, EvictionPriority priority);

    // Unload only this sampler; live or failed-to-destroy sibling objects keep
    // their backing alive. Cancels older requests, waits for active operations
    // and GPU consumers, and invalidates device mappings before resource free.
    // Whole-storage eviction invalidates every sibling. Cleanup errors retain
    // resources and charges for a later retry. Unload is not registration release.
    // Invalid/unregistered IDs set InvalidTextureId without mutation.
    void unloadTexture(uint32_t textureId);
    void unloadAll();

    /// Abort all pending operations and halt the loader gracefully.
    /// After calling abort(), no new requests will be processed.
    /// Safe to call from any thread. Blocks until all in-flight operations complete.
    void abort();

    /// Check if the loader has been aborted.
    bool isAborted() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hip_demand
