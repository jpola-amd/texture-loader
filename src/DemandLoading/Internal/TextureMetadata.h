// SPDX-License-Identifier: MIT
// Internal header for texture metadata storage

#pragma once

#include <hip/hip_runtime.h>
#include <DemandLoading/DemandTextureLoader.h>
#include <ImageSource/ImageSource.h>

#include <atomic>
#include <memory>
#include <string>

 // Verify binary compatibility between TextureObject and hipTextureObject_t
static_assert(sizeof(hip_demand::TextureObject) == sizeof(hipTextureObject_t),
              "TextureObject must be binary-compatible with hipTextureObject_t");

namespace hip_demand {
namespace internal {

/// Per-texture metadata stored on the host.
/// Contains GPU resources, loading state, and texture properties.
struct TextureMetadata {
    std::string filename;
    std::shared_ptr<ImageSource> imageSource;  // Optional: user-provided image source
    TextureDesc desc{};
    
    // GPU resources
    hipTextureObject_t texObj = 0;
    hipArray_t array = nullptr;
    hipMipmappedArray_t mipmapArray = nullptr;
    
    // Texture properties
    int width = 0;
    int height = 0;
    int channels = 0;
    bool hasMipmaps = false;
    int numMipLevels = 0;
    size_t memoryUsage = 0;
    uint32_t lastUsedFrame = 0;
    uint32_t loadedFrame = 0;  // Frame when texture was loaded (for thrashing prevention)
    
    // Loading state - atomic for thread-safe double-checked locking
    std::atomic<bool> resident{false};
    std::atomic<bool> loading{false};
    
    LoaderError lastError = LoaderError::Success;
    
    // Cached data for memory-based textures
    std::unique_ptr<uint8_t[]> cachedData;
    
    // Default constructor
    TextureMetadata() = default;
    
    // Move constructor - handles atomic members properly
    TextureMetadata(TextureMetadata&& other) noexcept
        : filename(std::move(other.filename))
        , imageSource(std::move(other.imageSource))
        , desc(other.desc)
        , texObj(other.texObj)
        , array(other.array)
        , mipmapArray(other.mipmapArray)
        , width(other.width)
        , height(other.height)
        , channels(other.channels)
        , hasMipmaps(other.hasMipmaps)
        , numMipLevels(other.numMipLevels)
        , memoryUsage(other.memoryUsage)
        , lastUsedFrame(other.lastUsedFrame)
        , loadedFrame(other.loadedFrame)
        , resident(other.resident.load(std::memory_order_relaxed))
        , loading(other.loading.load(std::memory_order_relaxed))
        , lastError(other.lastError)
        , cachedData(std::move(other.cachedData))
    {}
    
    // Move assignment - handles atomic members properly
    TextureMetadata& operator=(TextureMetadata&& other) noexcept {
        if (this != &other) {
            filename = std::move(other.filename);
            imageSource = std::move(other.imageSource);
            desc = other.desc;
            texObj = other.texObj;
            array = other.array;
            mipmapArray = other.mipmapArray;
            width = other.width;
            height = other.height;
            channels = other.channels;
            hasMipmaps = other.hasMipmaps;
            numMipLevels = other.numMipLevels;
            memoryUsage = other.memoryUsage;
            lastUsedFrame = other.lastUsedFrame;
            loadedFrame = other.loadedFrame;
            resident.store(other.resident.load(std::memory_order_relaxed), std::memory_order_relaxed);
            loading.store(other.loading.load(std::memory_order_relaxed), std::memory_order_relaxed);
            lastError = other.lastError;
            cachedData = std::move(other.cachedData);
        }
        return *this;
    }
    
    // Delete copy operations - atomic<bool> is not copyable
    TextureMetadata(const TextureMetadata&) = delete;
    TextureMetadata& operator=(const TextureMetadata&) = delete;
};

/// Request statistics copied back from device
struct RequestStats {
    uint32_t count;
    uint32_t overflow;
};

} // namespace internal
} // namespace hip_demand
