// SPDX-License-Identifier: MIT
// Internal implementation header for DemandTextureLoader

#pragma once

#include <DemandLoading/DemandTextureLoader.h>
#include <DemandLoading/DeviceContext.h>
#include <DemandLoading/Ticket.h>
#include <ImageSource/ImageSource.h>
#include "Internal/TextureMetadata.h"
#include "Internal/TextureIdentity.h"
#include "Internal/HipEventPool.h"
#include "Internal/PinnedMemoryPool.h"
#include "Internal/ThreadPool.h"
#include "Internal/Utils.h"
#include "Internal/ImageData.h"
#include "Internal/HipCalls.h"

#include <hip/hip_runtime.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
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
    TextureHandle createTexture(const std::string& filename, const TextureDesc& desc,
        const capability_v1::Policy& policy = {{capability_v1::Version, sizeof(capability_v1::Policy)},
                                              capability_v1::MipPolicy::LegacyCompatibility},
        const anisotropy_v1::Request& request = anisotropy_v1::Request::legacy());
    TextureHandle createTexture(std::shared_ptr<ImageSource> imageSource, const TextureDesc& desc,
        const capability_v1::Policy& policy = {{capability_v1::Version, sizeof(capability_v1::Policy)},
                                              capability_v1::MipPolicy::LegacyCompatibility},
        const anisotropy_v1::Request& request = anisotropy_v1::Request::legacy());
    TextureHandle createTextureFromMemory(const void* data, int width, int height,
        int channels, const TextureDesc& desc,
        const capability_v1::Policy& policy = {{capability_v1::Version, sizeof(capability_v1::Policy)},
                                              capability_v1::MipPolicy::LegacyCompatibility},
        const anisotropy_v1::Request& request = anisotropy_v1::Request::legacy());
    contract_v1::Outcome getTextureStatusV1(uint32_t id, capability_v1::Status& status) const;
    contract_v1::Outcome getTextureAnisotropyStatusV1(uint32_t id, anisotropy_v1::Status& status) const;
    contract_v1::RegistrationResult enableCubicV1(uint32_t id);
    cubic_v1::DeviceContext getCubicContextV1() const;
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
    void abort();
    bool isAborted() const;

private:
    LoaderError initialize();
    TextureHandle registrationFailure(LoaderError error);
    TextureHandle commitRegistration(internal::TextureMetadata&& info);
    TextureHandle registeredHandle(uint32_t id) const;
    uint32_t findSampler(const internal::ImageStorage& storage, const TextureDesc& desc,
                         capability_v1::MipPolicy policy, const anisotropy_v1::Request& request) const;
    void refreshStorageStatusLocked(internal::ImageStorage& storage);
    bool cleanupTextureResources(internal::TextureMetadata& info);
    bool cleanupStorageResources(internal::ImageStorage& storage);
    bool selectDevice();
    bool publishMappingsLocked(bool retiring = false);
    bool quiesce(bool retiring = false);
    void cancelTextureLocked(uint32_t texId);
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
    struct LoadRequest {
        uint32_t id;
        uint64_t epoch;
        std::shared_ptr<internal::ImageStorage> storage;
    };
    enum class LoadOutcome { NotLoaded, StorageFailed, Loaded };
    LoadOutcome loadTexture(const LoadRequest& request);
    bool loadStorage(internal::ImageStorage& storage, const TextureDesc& desc, capability_v1::MipPolicy policy,
                     capability_v1::Support& support, uint32_t maxAnisotropy);
    void destroyTexture(uint32_t texId);
    void evictIfNeeded(size_t requiredMemory,
                       const std::unordered_set<internal::ImageStorage*>& requestedStorage);

    // Request processing
    std::vector<LoadRequest> readRequests(hipStream_t stream, const DeviceContext& deviceContext);
    size_t processRequestsHost(const std::vector<LoadRequest>& requests);

    // Mipmap generation
    bool generateMipLevels(hipMipmappedArray_t mipmapArray, const internal::ImageData& baseImage,
                          int numLevels, ImageSource* source, bool sourceSRGB, hipError_t& error,
                          capability_v1::Operation& operation);

    // Configuration
    LoaderOptions options_;
    internal::HipCalls hipCalls_;
    LoaderError initializationError_ = LoaderError::HipError;
    int device_;
    bool deviceKnown_ = false;
    hipCtx_t ownerContext_ = nullptr;
    std::atomic<hipError_t> lastDeviceError_{hipSuccess};
    capability_v1::Status identityStatus_{};
    mutable std::mutex mutex_;
    // Runtime operations are serialized independently of registration/metadata.
    // Never acquire this mutex while holding mutex_: a decoder can need mutex_.
    std::mutex operationMutex_;

    // Device context with all device pointers
    DeviceContext deviceContext_{};
    internal::RequestStats* d_requestStats_ = nullptr;
    std::vector<cubic_v1::Entry> cubicEntries_;
    cubic_v1::Entry* d_cubicEntries_ = nullptr;
    uint64_t cubicIncarnation_ = 0;

    hipStream_t requestCopyStream_ = nullptr;

    // Host pinned buffers
    uint32_t* h_residentFlags_ = nullptr;
    TextureObject* h_textures_ = nullptr;  // Binary-compatible with hipTextureObject_t
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

    // Distinct identity namespaces and complete equality resolve hash collisions.
    std::unordered_map<internal::StorageKey, std::shared_ptr<internal::ImageStorage>,
                       internal::StorageKeyHash> storages_;

    // Statistics
    std::atomic<size_t> lastRequestCount_{0};
    std::atomic<bool> lastRequestOverflow_{false};

    // Async operation coordination
    std::atomic<int> inFlightAsync_{0};
    std::atomic<bool> destroying_{false};
    std::atomic<bool> aborted_{false};
    mutable std::mutex asyncMutex_;
    mutable std::condition_variable asyncCv_;

    // Thread pool for parallel texture loading
    std::unique_ptr<internal::ThreadPool> threadPool_;

    // Pinned memory pool for async request processing
    std::unique_ptr<internal::PinnedMemoryPool> pinnedMemoryPool_;

    // HIP event pool for async operations
    std::unique_ptr<internal::HipEventPool> hipEventPool_;

    std::atomic<LoaderError> lastError_{LoaderError::Success};

    // Probe cache and retained rollback resources are scoped to this loader's
    // owning primary context lifetime; operationMutex_ serializes access.
    struct Probe {
        TextureDesc desc{};
        bool floatPixels = false;
        uint32_t maxAnisotropy = 0;
        capability_v1::Support support = capability_v1::Support::Unknown;
        capability_v1::Failure primary{}, cleanup{};
        hipMipmappedArray_t array = nullptr;
        hipTextureObject_t sampler = 0;
        size_t bytes = 0;
    };
    std::vector<Probe> probes_;
    Probe& probeMipmaps(const TextureDesc& desc, bool floatPixels, uint32_t maxAnisotropy);
    bool cleanupProbe(Probe& probe);
};

} // namespace hip_demand
