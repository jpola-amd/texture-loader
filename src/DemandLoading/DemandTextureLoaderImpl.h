// SPDX-License-Identifier: MIT
// Internal implementation header for DemandTextureLoader

#pragma once

#include <DemandLoading/DemandTextureLoader.h>
#include <DemandLoading/DeviceContext.h>
#include <DemandLoading/Ticket.h>
#include "Internal/TextureMetadata.h"
#include "Internal/HipEventPool.h"
#include "Internal/PinnedMemoryPool.h"
#include "Internal/ThreadPool.h"
#include "Internal/Utils.h"

#include <hip/hip_runtime.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace hip_demand {

/// Implementation class for DemandTextureLoader (PIMPL pattern)
class DemandTextureLoader::Impl {
public:
    explicit Impl(const LoaderOptions& options);
    ~Impl();

    // Non-copyable
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    // Public API implementations
    TextureHandle createTexture(const std::string& filename, const TextureDesc& desc);
    TextureHandle createTextureFromMemory(const void* data, int width, int height,
                                         int channels, const TextureDesc& desc);
    void launchPrepare(hipStream_t stream);
    DeviceContext getDeviceContext() const;
    size_t processRequests(hipStream_t stream, const DeviceContext& deviceContext);
    Ticket processRequestsAsync(hipStream_t stream, const DeviceContext& deviceContext);
    size_t getResidentTextureCount() const;
    size_t getTotalTextureMemory() const;
    size_t getRequestCount() const;
    bool hadRequestOverflow() const;
    LoaderError getLastError() const;
    void enableEviction(bool enable);
    void setMaxTextureMemory(size_t bytes);
    size_t getMaxTextureMemory() const;
    void updateEvictionPriority(uint32_t texId, EvictionPriority priority);
    void unloadTexture(uint32_t texId);
    void unloadAll();

private:
    // RAII guard for async operations
    struct AsyncGuard {
        DemandTextureLoader::Impl* self;
        bool committed = false;
        ~AsyncGuard();
    };

    // Dirty tracking helpers (require mutex_ held)
    void markAllDirty();
    void clearDirtyLocked();
    void markTextureDirtyLocked(uint32_t texId);
    void markResidentWordDirtyLocked(uint32_t wordIdx);

    // Core loading/unloading
    bool loadTexture(uint32_t texId);
    bool loadTextureThreadSafe(uint32_t texId);
    void destroyTexture(uint32_t texId);
    void evictIfNeeded(size_t requiredMemory);

    // Request processing
    size_t processRequestsHost(uint32_t requestCount, const uint32_t* requests);

    // Mipmap generation
    bool generateMipLevels(hipMipmappedArray_t mipmapArray, unsigned char* baseData,
                          int baseWidth, int baseHeight, int numLevels);

    // Configuration
    LoaderOptions options_;
    int device_;
    mutable std::mutex mutex_;

    // Device context with all device pointers
    DeviceContext deviceContext_{};
    internal::RequestStats* d_requestStats_ = nullptr;

    hipStream_t requestCopyStream_ = nullptr;

    // Host pinned buffers
    uint32_t* h_residentFlags_ = nullptr;
    hipTextureObject_t* h_textures_ = nullptr;
    uint32_t* h_requests_ = nullptr;
    internal::RequestStats* h_requestStats_ = nullptr;
    size_t flagWordCount_ = 0;

    // Dirty tracking for deviceContext_ updates (requires mutex_)
    bool residentFlagsDirty_ = false;
    bool texturesDirty_ = false;
    size_t dirtyResidentWordBegin_ = std::numeric_limits<size_t>::max();
    size_t dirtyResidentWordEnd_ = 0;
    size_t dirtyTextureBegin_ = std::numeric_limits<size_t>::max();
    size_t dirtyTextureEnd_ = 0;

    // Texture storage
    std::vector<internal::TextureMetadata> textures_;
    uint32_t nextTextureId_ = 0;
    uint32_t currentFrame_ = 0;
    size_t totalMemoryUsage_ = 0;

    // Statistics
    std::atomic<size_t> lastRequestCount_{0};
    std::atomic<bool> lastRequestOverflow_{false};

    // Async operation coordination
    std::atomic<int> inFlightAsync_{0};
    std::atomic<bool> destroying_{false};
    mutable std::mutex asyncMutex_;
    mutable std::condition_variable asyncCv_;

    // Thread pool for parallel texture loading
    std::unique_ptr<internal::ThreadPool> threadPool_;

    // Pinned memory pool for async request processing
    std::unique_ptr<internal::PinnedMemoryPool> pinnedMemoryPool_;

    // HIP event pool for async operations
    std::unique_ptr<internal::HipEventPool> hipEventPool_;

    LoaderError lastError_ = LoaderError::Success;
};

} // namespace hip_demand
