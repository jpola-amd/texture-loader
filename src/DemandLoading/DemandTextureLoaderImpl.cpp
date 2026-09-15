// SPDX-License-Identifier: MIT
// DemandTextureLoader implementation
#include <hip/hip_runtime.h>
#include "DemandTextureLoaderImpl.h"
#include "Internal/HipCheck.h"
#include "Internal/TextureMetadata.h"
#include "Internal/Utils.h"
#include "Internal/ImageData.h"

#include <DemandLoading/Logging.h>
#include <DemandLoading/Ticket.h>
#include <ImageSource/ImageSource.h>
#include <ImageSource/TextureInfo.h>

#ifdef USE_OIIO
#include <ImageSource/OIIOReader.h>
#endif

#include "stb_image.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <tuple>
#include <unordered_set>

namespace hip_demand {

using internal::TextureMetadata;
using internal::ImageStorage;
using internal::StorageKey;
using internal::StorageIdentity;
using internal::RequestStats;
using internal::calculateMipLevels;
using internal::HipOperation;

namespace {
bool validDescriptor(const TextureDesc& desc) {
    const auto validAddress = [](hipTextureAddressMode mode) {
        return mode == hipAddressModeWrap || mode == hipAddressModeClamp ||
               mode == hipAddressModeMirror || mode == hipAddressModeBorder;
    };
    const auto validFilter = [](hipTextureFilterMode mode) {
        return mode == hipFilterModePoint || mode == hipFilterModeLinear;
    };
    return validAddress(desc.addressMode[0]) && validAddress(desc.addressMode[1]) &&
           validFilter(desc.filterMode) && validFilter(desc.mipmapFilterMode) &&
           (desc.evictionPriority == EvictionPriority::Normal || desc.evictionPriority == EvictionPriority::Low ||
            desc.evictionPriority == EvictionPriority::High || desc.evictionPriority == EvictionPriority::KeepResident);
}

LoaderError hipLoaderError(hipError_t error) {
    return error == hipErrorOutOfMemory ? LoaderError::OutOfMemory : LoaderError::HipError;
}

void validateRegistrationImage(const TextureInfo& info) {
    if (!info.isValid)
        throw std::invalid_argument("Invalid source metadata");
    const size_t sourceBytes = internal::imageByteSize(info.width, info.height, info.numChannels,
                                                       getBytesPerChannel(info.format));
    const size_t uploadBytes = internal::imageByteSize(info.width, info.height, 4,
        info.format == HIP_AD_FORMAT_UNSIGNED_INT8 ? 1 : sizeof(float));
    if (std::max(sourceBytes, uploadBytes) > static_cast<size_t>(PTRDIFF_MAX))
        throw std::overflow_error("Image exceeds addressable buffer representation");
}

StorageKey storageKey(const ImageStorage& storage, const TextureDesc& desc, StorageIdentity identity) {
    StorageKey key;
    key.identity = identity;
    if (identity == StorageIdentity::Filename) {
        key.filename = storage.filename;
    } else {
        key.metadata = storage.sourceInfo;
        key.contentHash = storage.contentHash;
        if (identity == StorageIdentity::Source)
            key.source = storage.imageSource.get();
    }
    key.sRGB = desc.sRGB;
    key.generateMipmaps = desc.generateMipmaps;
    key.maxMipLevel = desc.maxMipLevel;
    return key;
}
}

// -----------------------------------------------------------------------------
// Constructor / Destructor
// -----------------------------------------------------------------------------

DemandTextureLoader::Impl::Impl(const LoaderOptions& options)
    : options_(options), device_(0)
{
    try {
        initializationError_ = initialize();
    } catch (const std::bad_alloc&) {
        initializationError_ = LoaderError::OutOfMemory;
    }
    lastError_ = initializationError_;
    if (initializationError_ != LoaderError::Success)
        logMessage(LogLevel::Error, "Loader initialization failed: %s", getErrorString(initializationError_));
}

LoaderError DemandTextureLoader::Impl::initialize() {
    if (options_.maxTextures == 0 || options_.maxTextures > UINT32_MAX ||
        options_.maxRequestsPerLaunch == 0 || options_.maxRequestsPerLaunch > UINT32_MAX ||
        options_.maxTextures > SIZE_MAX / sizeof(TextureObject) ||
        options_.maxRequestsPerLaunch > SIZE_MAX / sizeof(uint32_t) ||
        options_.maxTextures > textures_.max_size()) {
        return LoaderError::InvalidParameter;
    }

    hipError_t err = hipCalls_.call(HipOperation::GetDevice, [&] { return hipGetDevice(&device_); });
    if (err != hipSuccess) return hipLoaderError(err);
    deviceKnown_ = true;
    err = hipStreamCreateWithFlags(&requestCopyStream_, hipStreamNonBlocking);
    if (err != hipSuccess) return hipLoaderError(err);

    err = hipCalls_.call(HipOperation::DeviceAllocation, [&] {
        return hipMalloc(&deviceContext_.requests, options_.maxRequestsPerLaunch * sizeof(uint32_t));
    });
    if (err != hipSuccess) return hipLoaderError(err);
    err = hipCalls_.call(HipOperation::DeviceAllocation, [&] {
        return hipMalloc(&deviceContext_.textures, options_.maxTextures * sizeof(TextureObject));
    });
    if (err != hipSuccess) return hipLoaderError(err);

    const size_t flagWords = options_.maxTextures / 32 + (options_.maxTextures % 32 != 0);
    err = hipCalls_.call(HipOperation::DeviceAllocation, [&] {
        return hipMalloc(&deviceContext_.residentFlags, flagWords * sizeof(uint32_t));
    });
    if (err != hipSuccess) return hipLoaderError(err);

    err = hipCalls_.call(HipOperation::DeviceAllocation, [&] {
        return hipMalloc(&d_requestStats_, sizeof(RequestStats));
    });
    if (err != hipSuccess) return hipLoaderError(err);
    deviceContext_.requestCount = reinterpret_cast<uint32_t*>(d_requestStats_);
    deviceContext_.requestOverflow = deviceContext_.requestCount + 1;

    err = hipCalls_.call(HipOperation::Initialize, [&] {
        return hipMemset(deviceContext_.residentFlags, 0, flagWords * sizeof(uint32_t));
    });
    if (err != hipSuccess) return hipLoaderError(err);
    err = hipCalls_.call(HipOperation::Initialize, [&] {
        return hipMemset(deviceContext_.textures, 0, options_.maxTextures * sizeof(TextureObject));
    });
    if (err != hipSuccess) return hipLoaderError(err);
    err = hipCalls_.call(HipOperation::Initialize, [&] {
        return hipMemset(d_requestStats_, 0, sizeof(RequestStats));
    });
    if (err != hipSuccess) return hipLoaderError(err);

    // Allocate host pinned buffers for async copies
    flagWordCount_ = flagWords;
    err = hipCalls_.call(HipOperation::HostAllocation, [&] {
        return hipHostMalloc(reinterpret_cast<void**>(&h_residentFlags_), flagWords * sizeof(uint32_t));
    });
    if (err != hipSuccess) return hipLoaderError(err);
    err = hipCalls_.call(HipOperation::HostAllocation, [&] {
        return hipHostMalloc(reinterpret_cast<void**>(&h_textures_), options_.maxTextures * sizeof(TextureObject));
    });
    if (err != hipSuccess) return hipLoaderError(err);
    err = hipCalls_.call(HipOperation::HostAllocation, [&] {
        return hipHostMalloc(reinterpret_cast<void**>(&h_requests_), options_.maxRequestsPerLaunch * sizeof(uint32_t));
    });
    if (err != hipSuccess) return hipLoaderError(err);
    err = hipCalls_.call(HipOperation::HostAllocation, [&] {
        return hipHostMalloc(reinterpret_cast<void**>(&h_requestStats_), sizeof(RequestStats));
    });
    if (err != hipSuccess) return hipLoaderError(err);

    std::fill_n(h_residentFlags_, flagWords, 0u);
    std::fill_n(h_textures_, options_.maxTextures, static_cast<TextureObject>(0));
    std::fill_n(h_requests_, options_.maxRequestsPerLaunch, 0u);
    h_requestStats_->count = 0;
    h_requestStats_->overflow = 0;

    // First launchPrepare must upload the entire state.
    markAllDirty();

    textures_.resize(options_.maxTextures);

    // Create thread pool for parallel texture loading
    unsigned int numThreads = options_.maxThreads;
    if (numThreads == 0) {
        numThreads = std::max(1u, std::thread::hardware_concurrency() / 2);
    }
    threadPool_ = std::make_unique<internal::ThreadPool>(numThreads);
    logMessage(LogLevel::Debug, "Impl: created thread pool with %u threads", threadPool_->size());

    // Create pinned memory pool for async request processing
    pinnedMemoryPool_ = std::make_unique<internal::PinnedMemoryPool>(4);

    // Create HIP event pool for async operations (pre-allocate 4 events)
    hipEventPool_ = std::make_unique<internal::HipEventPool>(4);
    deviceContext_.maxTextures = static_cast<uint32_t>(options_.maxTextures);
    deviceContext_.maxRequests = static_cast<uint32_t>(options_.maxRequestsPerLaunch);
    return LoaderError::Success;
}

DemandTextureLoader::Impl::~Impl() {
    if (deviceKnown_) {
        const hipError_t err = hipSetDevice(device_);
        if (err != hipSuccess)
            logMessage(LogLevel::Error, "Loader teardown: device selection failed: %s", hipGetErrorString(err));
    }
    // Ensure any async request-processing tasks complete before we start tearing down
    // resources they might touch (e.g., mutex_, textures_, options_, logging).
    // Use seq_cst to establish a total order with the check in processRequestsAsync.
    destroying_.store(true, std::memory_order_seq_cst);
    {
        std::unique_lock<std::mutex> lock(asyncMutex_);
        asyncCv_.wait(lock, [&] { return inFlightAsync_.load(std::memory_order_acquire) == 0; });
    }

    // Destroy thread pool first - ensures all loading tasks complete
    threadPool_.reset();

    // Destroy pinned memory pool
    pinnedMemoryPool_.reset();

    // Destroy HIP event pool
    hipEventPool_.reset();

    if (requestCopyStream_) {
        HIP_CHECK(hipStreamDestroy(requestCopyStream_));
        requestCopyStream_ = nullptr;
    }

    unloadAll();
    if (totalMemoryUsage_ != 0) {
        logMessage(LogLevel::Warn, "Loader teardown: retrying retained resource cleanup");
        unloadAll();
        if (totalMemoryUsage_ != 0)
            logMessage(LogLevel::Error, "Loader teardown: %zu payload bytes could not be freed", totalMemoryUsage_);
    }

    if (h_residentFlags_) HIP_CHECK(hipHostFree(h_residentFlags_));
    if (h_textures_) HIP_CHECK(hipHostFree(h_textures_));
    if (h_requests_) HIP_CHECK(hipHostFree(h_requests_));
    if (h_requestStats_) HIP_CHECK(hipHostFree(h_requestStats_));

    if (deviceContext_.residentFlags) HIP_CHECK(hipFree(deviceContext_.residentFlags));
    if (deviceContext_.textures) HIP_CHECK(hipFree(deviceContext_.textures));
    if (deviceContext_.requests) HIP_CHECK(hipFree(deviceContext_.requests));
    if (d_requestStats_) HIP_CHECK(hipFree(d_requestStats_));
}

// AsyncGuard destructor
DemandTextureLoader::Impl::AsyncGuard::~AsyncGuard() {
    if (!committed) {
        std::lock_guard<std::mutex> lock(self->asyncMutex_);
        self->inFlightAsync_.fetch_sub(1, std::memory_order_acq_rel);
        self->asyncCv_.notify_all();
    }
}

// -----------------------------------------------------------------------------
// Dirty Tracking Helpers
// -----------------------------------------------------------------------------

void DemandTextureLoader::Impl::markAllDirty() {
    residentFlagsDirty_ = true;
    texturesDirty_ = true;
    dirtyResidentWordBegin_ = 0;
    dirtyResidentWordEnd_ = flagWordCount_ ? (flagWordCount_ - 1) : 0;
    dirtyTextureBegin_ = 0;
    dirtyTextureEnd_ = options_.maxTextures ? (options_.maxTextures - 1) : 0;
}

void DemandTextureLoader::Impl::clearDirtyLocked() {
    residentFlagsDirty_ = false;
    texturesDirty_ = false;
    dirtyResidentWordBegin_ = std::numeric_limits<size_t>::max();
    dirtyResidentWordEnd_ = 0;
    dirtyTextureBegin_ = std::numeric_limits<size_t>::max();
    dirtyTextureEnd_ = 0;
}

void DemandTextureLoader::Impl::markTextureDirtyLocked(uint32_t texId) {
    texturesDirty_ = true;
    dirtyTextureBegin_ = std::min(dirtyTextureBegin_, static_cast<size_t>(texId));
    dirtyTextureEnd_ = std::max(dirtyTextureEnd_, static_cast<size_t>(texId));
}

void DemandTextureLoader::Impl::markResidentWordDirtyLocked(uint32_t wordIdx) {
    residentFlagsDirty_ = true;
    dirtyResidentWordBegin_ = std::min(dirtyResidentWordBegin_, static_cast<size_t>(wordIdx));
    dirtyResidentWordEnd_ = std::max(dirtyResidentWordEnd_, static_cast<size_t>(wordIdx));
}

// -----------------------------------------------------------------------------
// Texture Creation
// -----------------------------------------------------------------------------

TextureHandle DemandTextureLoader::Impl::createTexture(const std::string& filename, const TextureDesc& desc) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initializationError_ != LoaderError::Success)
        return registrationFailure(initializationError_);
    if (filename.empty() || filename.find('\0') != std::string::npos || !validDescriptor(desc))
        return registrationFailure(LoaderError::InvalidParameter);

    try {
        TextureMetadata info;
        info.desc = desc;
        ImageStorage identity;
        identity.filename = filename;
        auto it = storages_.find(storageKey(identity, desc, StorageIdentity::Filename));
        if (it != storages_.end()) {
            const uint32_t id = findSampler(*it->second, desc);
            if (id != InvalidTextureId)
                return registeredHandle(id);
            if (nextTextureId_ >= options_.maxTextures)
                return registrationFailure(LoaderError::MaxTexturesExceeded);
            info.storage = it->second;
            return commitRegistration(std::move(info));
        }
        if (nextTextureId_ >= options_.maxTextures)
            return registrationFailure(LoaderError::MaxTexturesExceeded);
        info.storage = std::make_shared<ImageStorage>();
        auto& storage = *info.storage;
        storage.filename = filename;
        storage.uploadBytesPerPixel = stbi_is_hdr(filename.c_str()) || stbi_is_16_bit(filename.c_str()) ? 16 : 4;
#ifdef USE_OIIO
        try {
            auto source = createImageSource(filename);
            if (source) {
                TextureInfo texInfo;
                source->open(&texInfo);
                if (source->isOpen()) {
                    validateRegistrationImage(texInfo);
                    storage.sourceInfo = texInfo;
                    storage.width = static_cast<int>(texInfo.width);
                    storage.height = static_cast<int>(texInfo.height);
                    storage.channels = static_cast<int>(texInfo.numChannels);
                    storage.uploadBytesPerPixel = texInfo.format == HIP_AD_FORMAT_UNSIGNED_INT8 ? 4 : 16;
                    storage.imageSource = std::move(source);
                }
            }
        } catch (const std::bad_alloc&) {
            throw;
        } catch (const std::exception& e) {
            // Filename registration remains lazy even when metadata probing fails.
            logMessage(LogLevel::Warn, "createTexture: metadata probe for '%s' failed: %s", filename.c_str(), e.what());
        }
#endif
        if (!storage.imageSource) {
            int w, h, c;
            if (stbi_info(filename.c_str(), &w, &h, &c)) {
                storage.width = w;
                storage.height = h;
                storage.channels = c;
            } else {
                info.lastError = LoaderError::FileNotFound;
                logMessage(LogLevel::Warn, "createTexture: deferred file read for '%s'", filename.c_str());
            }
        }
        return commitRegistration(std::move(info));
    } catch (const std::bad_alloc&) {
        return registrationFailure(LoaderError::OutOfMemory);
    }
}

TextureHandle DemandTextureLoader::Impl::createTexture(std::shared_ptr<ImageSource> imageSource, const TextureDesc& desc) {
    // A retained source can feed multiple incompatible storages. Serialize its
    // registration-time open/hash calls with all source reads as well.
    std::lock_guard<std::mutex> operation(operationMutex_);
    std::lock_guard<std::mutex> lock(mutex_);
    if (initializationError_ != LoaderError::Success)
        return registrationFailure(initializationError_);
    if (!imageSource || !validDescriptor(desc))
        return registrationFailure(LoaderError::InvalidParameter);
    if (!selectDevice())
        return registrationFailure(lastError_.load(std::memory_order_relaxed));

    unsigned long long contentHash;
    TextureInfo texInfo;
    try {
        contentHash = imageSource->getHash(0);
        imageSource->open(&texInfo);
        if (!imageSource->isOpen())
            return registrationFailure(LoaderError::ImageLoadFailed);
    } catch (const std::bad_alloc&) {
        return registrationFailure(LoaderError::OutOfMemory);
    } catch (const std::exception& e) {
        logMessage(LogLevel::Error, "createTexture: ImageSource exception: %s", e.what());
        return registrationFailure(LoaderError::ImageLoadFailed);
    }
    try {
        validateRegistrationImage(texInfo);
        if (texInfo.numMipLevels == 0 ||
            texInfo.numMipLevels > static_cast<unsigned int>(calculateMipLevels(texInfo.width, texInfo.height)))
            return registrationFailure(LoaderError::InvalidParameter);
    } catch (const std::invalid_argument&) {
        return registrationFailure(LoaderError::InvalidParameter);
    } catch (const std::overflow_error&) {
        return registrationFailure(LoaderError::InvalidParameter);
    }
    try {
        ImageStorage identity;
        identity.imageSource = imageSource;
        identity.sourceInfo = texInfo;
        identity.contentHash = contentHash;
        auto it = storages_.find(storageKey(identity, desc, StorageIdentity::Source));
        if (it == storages_.end() && contentHash)
            it = storages_.find(storageKey(identity, desc, StorageIdentity::Content));
        TextureMetadata info;
        info.desc = desc;
        if (it != storages_.end()) {
            const uint32_t id = findSampler(*it->second, desc);
            if (id != InvalidTextureId)
                return registeredHandle(id);
            info.storage = it->second;
            // The incoming alias is deliberately not cached or retained.
        }
        if (nextTextureId_ >= options_.maxTextures)
            return registrationFailure(LoaderError::MaxTexturesExceeded);
        if (!info.storage) {
            info.storage = std::make_shared<ImageStorage>();
            auto& storage = *info.storage;
            storage.imageSource = std::move(imageSource);
            storage.sourceInfo = texInfo;
            storage.contentHash = contentHash;
            storage.width = static_cast<int>(texInfo.width);
            storage.height = static_cast<int>(texInfo.height);
            storage.channels = static_cast<int>(texInfo.numChannels);
            storage.uploadBytesPerPixel = texInfo.format == HIP_AD_FORMAT_UNSIGNED_INT8 ? 4 : 16;
        }
        return commitRegistration(std::move(info));
    } catch (const std::bad_alloc&) {
        return registrationFailure(LoaderError::OutOfMemory);
    }
}

TextureHandle DemandTextureLoader::Impl::createTextureFromMemory(const void* data, int width, int height,
                                                                  int channels, const TextureDesc& desc) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initializationError_ != LoaderError::Success)
        return registrationFailure(initializationError_);

    if (!data || width <= 0 || height <= 0 || channels <= 0 || channels > 4 || !validDescriptor(desc)) {
        return registrationFailure(LoaderError::InvalidParameter);
    }

    size_t dataSize;
    try {
        TextureInfo sourceInfo;
        sourceInfo.width = width;
        sourceInfo.height = height;
        sourceInfo.numChannels = channels;
        sourceInfo.isValid = true;
        validateRegistrationImage(sourceInfo);
        dataSize = internal::imageByteSize(width, height, channels, 1);
    } catch (const std::invalid_argument&) {
        return registrationFailure(LoaderError::InvalidParameter);
    } catch (const std::overflow_error&) {
        return registrationFailure(LoaderError::InvalidParameter);
    }

    if (nextTextureId_ >= options_.maxTextures)
        return registrationFailure(LoaderError::MaxTexturesExceeded);
    try {
        TextureMetadata info;
        info.desc = desc;
        info.storage = std::make_shared<ImageStorage>();
        auto& storage = *info.storage;
        storage.width = width;
        storage.height = height;
        storage.channels = channels;
        hipCalls_.registrationCheckpoint(HipOperation::CacheData);
        storage.cachedData = std::make_unique<uint8_t[]>(dataSize);
        std::memcpy(storage.cachedData.get(), data, dataSize);
        return commitRegistration(std::move(info));
    } catch (const std::bad_alloc&) {
        return registrationFailure(LoaderError::OutOfMemory);
    }
}

TextureHandle DemandTextureLoader::Impl::registrationFailure(LoaderError error) {
    lastError_ = error;
    logMessage(LogLevel::Error, "Texture registration failed: %s", getErrorString(error));
    return {InvalidTextureId, false, 0, 0, 0, error};
}

TextureHandle DemandTextureLoader::Impl::registeredHandle(uint32_t id) const {
    const auto& storage = *textures_[id].storage;
    return {id, true, storage.width, storage.height, storage.channels, LoaderError::Success};
}

uint32_t DemandTextureLoader::Impl::findSampler(const ImageStorage& storage, const TextureDesc& desc) const {
    const size_t hash = internal::TextureDescHash{}(desc);
    // Creation order makes lookup deterministic even when priority mutation
    // leaves multiple live registrations with the same complete descriptor.
    for (uint32_t id : storage.samplers)
        if (textures_[id].descriptorHash == hash && textures_[id].desc == desc)
            return id;
    return InvalidTextureId;
}

TextureHandle DemandTextureLoader::Impl::commitRegistration(TextureMetadata&& info) {
    const uint32_t id = nextTextureId_;
    auto& storage = *info.storage;
    const StorageKey filenameKey = storageKey(storage, info.desc, StorageIdentity::Filename);
    const StorageKey sourceKey = storageKey(storage, info.desc, StorageIdentity::Source);
    const StorageKey contentKey = storageKey(storage, info.desc, StorageIdentity::Content);
    struct Rollback {
        Impl& self;
        ImageStorage& storage;
        const StorageKey& filenameKey;
        const StorageKey& sourceKey;
        const StorageKey& contentKey;
        bool filename = false, source = false, content = false, sampler = false, committed = false;
        ~Rollback() {
            if (committed) return;
            if (filename) self.storages_.erase(filenameKey);
            if (source) self.storages_.erase(sourceKey);
            if (content) self.storages_.erase(contentKey);
            if (sampler) storage.samplers.pop_back();
        }
    } rollback{*this, storage, filenameKey, sourceKey, contentKey};
    try {
        storage.samplers.push_back(id);
        rollback.sampler = true;
        if (!storage.filename.empty()) {
            hipCalls_.registrationCheckpoint(HipOperation::FilenameMap);
            rollback.filename = storages_.emplace(filenameKey, info.storage).second;
        }
        if (storage.imageSource) {
            hipCalls_.registrationCheckpoint(HipOperation::SourceMap);
            rollback.source = storages_.emplace(sourceKey, info.storage).second;
        }
        if (storage.contentHash) {
            hipCalls_.registrationCheckpoint(HipOperation::ContentMap);
            rollback.content = storages_.emplace(contentKey, info.storage).second;
        }
    } catch (const std::bad_alloc&) {
        return registrationFailure(LoaderError::OutOfMemory);
    }
    rollback.committed = true;
    info.descriptorHash = internal::TextureDescHash{}(info.desc);
    textures_[id] = std::move(info);
    ++nextTextureId_;
    lastError_ = LoaderError::Success;
    return registeredHandle(id);
}

// -----------------------------------------------------------------------------
// Launch Prepare
// -----------------------------------------------------------------------------

void DemandTextureLoader::Impl::launchPrepare(hipStream_t stream) {
    std::lock_guard<std::mutex> operation(operationMutex_);
    if (!quiesce())
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (initializationError_ != LoaderError::Success) {
        lastError_ = initializationError_;
        logMessage(LogLevel::Error, "launchPrepare: loader initialization failed");
        return;
    }

    if (!publishMappingsLocked())
        return;

    // Reset request counter and overflow flag
    hipError_t err = hipMemsetAsync(d_requestStats_, 0, sizeof(RequestStats), stream);
    const hipError_t syncError = hipStreamSynchronize(stream);
    if (err == hipSuccess)
        err = syncError;
    if (err != hipSuccess) {
        lastError_ = LoaderError::HipError;
        logMessage(LogLevel::Error, "launchPrepare: hipMemsetAsync(requestStats) failed: %s", hipGetErrorString(err));
        return;
    }

    currentFrame_++;
    logMessage(LogLevel::Debug, "launchPrepare: frame=%u", currentFrame_);
}

DeviceContext DemandTextureLoader::Impl::getDeviceContext() const {
    return initializationError_ == LoaderError::Success ? deviceContext_ : DeviceContext{};
}

// -----------------------------------------------------------------------------
// Request Processing
// -----------------------------------------------------------------------------

std::vector<DemandTextureLoader::Impl::LoadRequest>
DemandTextureLoader::Impl::readRequests(hipStream_t stream, const DeviceContext& deviceContext) {
    if (initializationError_ != LoaderError::Success) {
        lastError_ = initializationError_;
        logMessage(LogLevel::Error, "processRequests: loader initialization failed");
        return {};
    }
    // Early exit if aborted
    if (aborted_.load(std::memory_order_acquire)) {
        return {};
    }
    if (!selectDevice())
        return {};
    if (deviceContext.requests != deviceContext_.requests ||
        deviceContext.requestCount != deviceContext_.requestCount ||
        deviceContext.requestOverflow != deviceContext_.requestOverflow ||
        deviceContext.maxRequests != deviceContext_.maxRequests) {
        lastError_ = LoaderError::InvalidParameter;
        logMessage(LogLevel::Error, "processRequests: context does not belong to this loader");
        return {};
    }

    uint32_t requestCount = 0;
    uint32_t overflow = 0;

    const uint32_t copyCount = std::min<uint32_t>(static_cast<uint32_t>(options_.maxRequestsPerLaunch), deviceContext.maxRequests);

    // Readback is completed before returning or handing work to the ticket
    // worker. No pool buffer or caller-owned launch context outlives this call.
    hipError_t err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        lastError_ = hipLoaderError(err);
        return {};
    }
    err = hipMemcpy(&requestCount, deviceContext.requestCount, sizeof(uint32_t),
                    hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        lastError_ = LoaderError::HipError;
        return {};
    }

    err = hipMemcpy(&overflow, deviceContext.requestOverflow, sizeof(uint32_t),
                    hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        lastError_ = LoaderError::HipError;
        return {};
    }

    // Copy the full request list up-front so we only need one stream sync.
    err = hipMemcpy(h_requests_, deviceContext.requests,
                         copyCount * sizeof(uint32_t),
                         hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        lastError_ = LoaderError::HipError;
        return {};
    }

    err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        lastError_ = LoaderError::HipError;
        return {};
    }

    lastRequestOverflow_.store(overflow != 0, std::memory_order_release);
    lastRequestCount_.store(static_cast<size_t>(requestCount), std::memory_order_release);
    if (overflow) {
        logMessage(LogLevel::Warn, "processRequests: overflow flagged (count=%u, cap=%zu)", requestCount, static_cast<size_t>(options_.maxRequestsPerLaunch));
    }
    logMessage(LogLevel::Debug, "processRequests: requestCount=%u", requestCount);

    if (requestCount == 0) {
        return {};
    }

    requestCount = std::min(requestCount, copyCount);
    std::vector<LoadRequest> result;
    result.reserve(requestCount);
    std::lock_guard<std::mutex> lock(mutex_);
    for (uint32_t i = 0; i < requestCount; ++i) {
        const uint32_t id = h_requests_[i];
        if (id >= nextTextureId_) {
            lastError_ = LoaderError::InvalidTextureId;
            logMessage(LogLevel::Error, "processRequests: invalid texture ID %u", id);
            continue;
        }
        result.push_back({id, textures_[id].cancellationEpoch, textures_[id].storage});
    }
    return result;
}

size_t DemandTextureLoader::Impl::processRequests(hipStream_t stream, const DeviceContext& deviceContext) {
    std::lock_guard<std::mutex> operation(operationMutex_);
    try {
        return processRequestsHost(readRequests(stream, deviceContext));
    } catch (const std::bad_alloc&) {
        lastError_ = LoaderError::OutOfMemory;
        logMessage(LogLevel::Error, "processRequests: host allocation failed");
        return 0;
    }
}

Ticket DemandTextureLoader::Impl::processRequestsAsync(hipStream_t stream, const DeviceContext& deviceContext) {
    inFlightAsync_.fetch_add(1, std::memory_order_seq_cst);
    AsyncGuard asyncGuard{this};
    if (destroying_.load(std::memory_order_seq_cst) || aborted_.load(std::memory_order_acquire))
        return Ticket{};
    try {
        std::vector<LoadRequest> requests;
        {
            std::lock_guard<std::mutex> operation(operationMutex_);
            requests = readRequests(stream, deviceContext);
        }
        if (requests.empty())
            return Ticket{};
        auto task = [this, requests = std::move(requests)]() mutable {
            AsyncGuard guard{this};
            {
                std::lock_guard<std::mutex> operation(operationMutex_);
                if (!destroying_.load(std::memory_order_acquire) &&
                    !aborted_.load(std::memory_order_acquire) && selectDevice()) {
                    try {
                        processRequestsHost(requests);
                    } catch (const std::bad_alloc&) {
                        lastError_ = LoaderError::OutOfMemory;
                        logMessage(LogLevel::Error, "processRequestsAsync: host allocation failed");
                    } catch (const std::exception& error) {
                        lastError_ = LoaderError::ImageLoadFailed;
                        logMessage(LogLevel::Error, "processRequestsAsync: %s", error.what());
                    }
                }
                // Release storage captures before announcing completion. A
                // retained Ticket closure must not outlive its resource owner.
                requests.clear();
            }
        };
        auto impl = createTicketImpl(std::move(task), stream);
        asyncGuard.committed = true;
        return Ticket(std::move(impl));
    } catch (const std::bad_alloc&) {
        lastError_ = LoaderError::OutOfMemory;
        logMessage(LogLevel::Error, "processRequestsAsync: request capture allocation failed");
        return Ticket{};
    }
}

size_t DemandTextureLoader::Impl::processRequestsHost(const std::vector<LoadRequest>& requests) {
    std::unordered_set<uint32_t> uniqueRequests;
    std::unordered_set<ImageStorage*> uniqueStorage;
    std::unordered_set<ImageStorage*> requestedStorage;
    std::vector<LoadRequest> toLoad;
    size_t estimatedMemoryNeeded = 0;

    {
        std::lock_guard<std::mutex> lock(mutex_);

        for (const auto& request : requests) {
            const uint32_t texId = request.id;
            if (request.epoch != textures_[texId].cancellationEpoch || request.epoch == UINT64_MAX)
                continue;
            requestedStorage.insert(request.storage.get());
            if (!textures_[texId].resident.load(std::memory_order_relaxed)) {
                if (uniqueRequests.insert(texId).second) {
                    toLoad.push_back(request);
                    const TextureMetadata& info = textures_[texId];
                    const auto& storage = *request.storage;
                    if (storage.array || storage.mipmapArray || !uniqueStorage.insert(request.storage.get()).second)
                        continue;
                    int w = storage.width;
                    int h = storage.height;
                    if (w > 0 && h > 0) {
                        int levels = info.desc.generateMipmaps ? calculateMipLevels(w, h) : 1;
                        if (info.desc.maxMipLevel > 0)
                            levels = static_cast<int>(std::min(static_cast<unsigned int>(levels), info.desc.maxMipLevel));
                        try {
                            const size_t mipMemory = internal::mipImageByteSize(w, h,
                                storage.uploadBytesPerPixel / 4, levels);
                            if (mipMemory > std::numeric_limits<size_t>::max() - estimatedMemoryNeeded)
                                estimatedMemoryNeeded = std::numeric_limits<size_t>::max();
                            else
                                estimatedMemoryNeeded += mipMemory;
                        } catch (const std::overflow_error& error) {
                            lastError_ = LoaderError::OutOfMemory;
                            logMessage(LogLevel::Error, "processRequests: mip size overflow: %s", error.what());
                            return 0;
                        } catch (const std::invalid_argument& error) {
                            lastError_ = LoaderError::InvalidParameter;
                            logMessage(LogLevel::Error, "processRequests: invalid mip size: %s", error.what());
                            return 0;
                        }
                    }
                }
            }
        }
        logMessage(LogLevel::Debug, "processRequests: unique-to-load=%zu estMem=%.2f MB", toLoad.size(), static_cast<double>(estimatedMemoryNeeded) / (1024.0 * 1024.0));

        // Check if we need eviction (with actual size estimates)
        if (options_.enableEviction && options_.maxTextureMemory > 0 && estimatedMemoryNeeded > 0) {
            evictIfNeeded(estimatedMemoryNeeded, requestedStorage);
        }
    }

    // Serialized storage operations coalesce successful uploads and storage
    // failures within this batch; sampler construction still completes per ID.
    size_t loaded = 0;
    std::unordered_set<ImageStorage*> failedStorage;
    for (const auto& request : toLoad) {
        if (failedStorage.count(request.storage.get())) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& info = textures_[request.id];
            info.lastError = request.storage->lastError;
            info.primaryHipError = request.storage->primaryHipError;
            continue;
        }
        const auto outcome = loadTexture(request);
        if (outcome == LoadOutcome::Loaded)
            ++loaded;
        else if (outcome == LoadOutcome::StorageFailed)
            failedStorage.insert(request.storage.get());
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < toLoad.size(); ++i) {
            const auto& request = toLoad[i];
            if (std::any_of(toLoad.begin(), toLoad.begin() + i, [&](const LoadRequest& previous) {
                    return previous.storage == request.storage;
                }))
                continue;
            auto& storage = *request.storage;
            // Storage failures already attempted rollback; retain failed
            // cleanup until a later explicit request or unload retries it.
            if (storage.uploaded)
                cleanupStorageResources(storage);
        }
    }
    return loaded;
}

// -----------------------------------------------------------------------------
// Texture Loading
// -----------------------------------------------------------------------------

DemandTextureLoader::Impl::LoadOutcome DemandTextureLoader::Impl::loadTexture(const LoadRequest& request) {
    if (aborted_.load(std::memory_order_acquire) || !selectDevice())
        return LoadOutcome::NotLoaded;
    TextureDesc desc;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& info = textures_[request.id];
        if (info.resident.load(std::memory_order_relaxed) ||
            request.epoch != info.cancellationEpoch || request.epoch == UINT64_MAX)
            return LoadOutcome::NotLoaded;
        if (info.texObj) {
            destroyTexture(request.id);
            if (info.texObj)
                return LoadOutcome::NotLoaded;
        }
        info.primaryHipError = hipSuccess;
        info.cleanupHipError = hipSuccess;
        desc = info.desc;
    }
    auto& storage = *request.storage;
    if (!storage.uploaded && !loadStorage(storage, desc)) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& info = textures_[request.id];
        info.lastError = storage.lastError;
        info.primaryHipError = storage.primaryHipError;
        info.cleanupHipError = storage.cleanupHipError;
        return LoadOutcome::StorageFailed;
    }

    hipResourceDesc resource{};
    if (storage.hasMipmaps) {
        resource.resType = hipResourceTypeMipmappedArray;
        resource.res.mipmap.mipmap = storage.mipmapArray;
    } else {
        resource.resType = hipResourceTypeArray;
        resource.res.array.array = storage.array;
    }
    hipTextureDesc sampler{};
    sampler.addressMode[0] = desc.addressMode[0];
    sampler.addressMode[1] = desc.addressMode[1];
    sampler.filterMode = desc.filterMode;
    sampler.readMode = storage.floatPixels ? hipReadModeElementType : hipReadModeNormalizedFloat;
    sampler.normalizedCoords = desc.normalizedCoords ? 1 : 0;
    sampler.sRGB = desc.sRGB && !storage.floatPixels ? 1 : 0;
    if (storage.hasMipmaps) {
        sampler.mipmapFilterMode = desc.mipmapFilterMode;
        sampler.minMipmapLevelClamp = 0;
        sampler.maxMipmapLevelClamp = static_cast<float>(storage.numMipLevels - 1);
    }
    hipTextureObject_t object = 0;
    const hipError_t error = hipCalls_.call(HipOperation::CreateSampler, [&] {
        return hipCreateTextureObject(&object, &resource, &sampler, nullptr);
    });

    std::lock_guard<std::mutex> lock(mutex_);
    auto& info = textures_[request.id];
    info.texObj = object;
    if (error != hipSuccess) {
        info.primaryHipError = error;
        info.lastError = hipLoaderError(error);
        cleanupTextureResources(info);
        lastError_ = info.lastError;
        return LoadOutcome::NotLoaded;
    }
    if (request.epoch != info.cancellationEpoch || aborted_.load(std::memory_order_acquire) ||
        destroying_.load(std::memory_order_acquire)) {
        cleanupTextureResources(info);
        return LoadOutcome::NotLoaded;
    }
    info.lastError = LoaderError::Success;
    h_textures_[request.id] = (TextureObject)info.texObj;
    h_residentFlags_[request.id / 32] |= 1u << (request.id % 32);
    markTextureDirtyLocked(request.id);
    markResidentWordDirtyLocked(request.id / 32);
    info.resident.store(true, std::memory_order_release);
    storage.lastUsedFrame = currentFrame_;
    return LoadOutcome::Loaded;
}

bool DemandTextureLoader::Impl::loadStorage(ImageStorage& info, const TextureDesc& desc) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cleanupStorageResources(info) || info.array || info.mipmapArray)
        return false;
    info.primaryHipError = hipSuccess;
    info.cleanupHipError = hipSuccess;
    info.lastError = LoaderError::Success;
    std::string filename = info.filename;
    std::shared_ptr<ImageSource> imageSource = info.imageSource;
    int initWidth = info.width;
    int initHeight = info.height;
    int initChannels = info.channels;
    const unsigned char* cachedPtr = info.cachedData.get();
    bool hasCached = (cachedPtr != nullptr);
    lock.unlock();

    // Read each ImageSource using its native pixel width before expanding to
    // the upload representation. FLOAT RGBA sources require 16 bytes per pixel.
    internal::ImageData image;
    TextureInfo loadedSourceInfo{};
    try {
        bool loaded = false;
        if (imageSource) {
            loaded = internal::readImageSource(*imageSource, image);
        } else if (!filename.empty()) {
#ifdef USE_OIIO
            try {
                std::unique_ptr<ImageSource> source = createImageSource(filename);
                if (source) {
                    loaded = internal::readImageSource(*source, image);
                    if (loaded)
                        imageSource = std::move(source);
                }
            } catch (const std::bad_alloc&) {
                throw;
            } catch (const std::exception& error) {
                logMessage(LogLevel::Warn, "loadStorage: OIIO read failed, trying stb: %s", error.what());
                loaded = false;
            }
#endif
            if (!loaded) {
                int width = 0, height = 0, channels = 0;
                TextureInfo sourceInfo;
                std::unique_ptr<void, decltype(&stbi_image_free)> pixels(nullptr, stbi_image_free);
                if (stbi_is_hdr(filename.c_str())) {
                    pixels.reset(stbi_loadf(filename.c_str(), &width, &height, &channels, 4));
                    sourceInfo.format = HIP_AD_FORMAT_FLOAT;
                } else if (stbi_is_16_bit(filename.c_str())) {
                    pixels.reset(stbi_load_16(filename.c_str(), &width, &height, &channels, 4));
                    sourceInfo.format = HIP_AD_FORMAT_UNSIGNED_INT16;
                } else {
                    pixels.reset(stbi_load(filename.c_str(), &width, &height, &channels, 4));
                    sourceInfo.format = HIP_AD_FORMAT_UNSIGNED_INT8;
                }
                if (pixels) {
                    sourceInfo.width = width;
                    sourceInfo.height = height;
                    sourceInfo.numChannels = 4;
                    sourceInfo.isValid = true;
                    image = internal::decodeImagePixels(pixels.get(), sourceInfo);
                    loaded = true;
                }
            }
        } else if (hasCached) {
            TextureInfo sourceInfo;
            sourceInfo.width = initWidth;
            sourceInfo.height = initHeight;
            sourceInfo.numChannels = initChannels;
            sourceInfo.format = HIP_AD_FORMAT_UNSIGNED_INT8;
            sourceInfo.isValid = true;
            image = internal::decodeImagePixels(cachedPtr, sourceInfo);
            loaded = true;
        }
        if (!loaded) {
            lock.lock();
            info.lastError = LoaderError::ImageLoadFailed;
            lastError_ = info.lastError;
            logMessage(LogLevel::Error, "loadStorage: failed to read image");
            return false;
        }
        if (imageSource)
            loadedSourceInfo = imageSource->getInfo();
    } catch (const std::bad_alloc&) {
        lock.lock();
        info.lastError = LoaderError::OutOfMemory;
        lastError_ = info.lastError;
        logMessage(LogLevel::Error, "loadStorage: source decode allocation failed");
        return false;
    } catch (const std::exception& e) {
        lock.lock();
        info.lastError = LoaderError::ImageLoadFailed;
        lastError_ = info.lastError;
        logMessage(LogLevel::Error, "loadTexture: image decode failed: %s", e.what());
        return false;
    } catch (...) {
        lock.lock();
        info.lastError = LoaderError::ImageLoadFailed;
        lastError_ = info.lastError;
        logMessage(LogLevel::Error, "loadTexture: unknown image decode failure");
        return false;
    }

    if (desc.sRGB)
        internal::linearizeFloatSRGB(image);
    const int width = static_cast<int>(image.width);
    const int height = static_cast<int>(image.height);
    const hipChannelFormatDesc channelDesc = image.channelDesc();
    const size_t rowBytes = image.rowBytes();

    hipError_t err = hipSuccess;
    LoaderError sourceError = LoaderError::ImageLoadFailed;
    bool success = false;
    bool useMipmaps = desc.generateMipmaps && (width > 1 || height > 1);

    if (useMipmaps) {
        // Check if mipmaps are supported on this GPU (one-time check)
        bool shouldUseMipmaps = true;
        {
            std::lock_guard<std::mutex> capLock(mutex_);
            if (!mipmapsSupportChecked_) {
                // Preserve the legacy capability probe/fallback, but retain
                // and charge a probe whose cleanup fails.
                if (mipmapProbe_) {
                    const hipError_t cleanup = hipFreeMipmappedArray(mipmapProbe_);
                    if (cleanup != hipSuccess) {
                        info.cleanupHipError = cleanup;
                        info.lastError = hipLoaderError(cleanup);
                        lastError_ = info.lastError;
                        return false;
                    }
                    mipmapProbe_ = nullptr;
                    totalMemoryUsage_ -= 4;
                }
                hipChannelFormatDesc testChannelDesc = hipCreateChannelDesc<uchar4>();
                hipExtent testExtent = make_hipExtent(1, 1, 0);
                hipError_t testErr = hipMallocMipmappedArray(&mipmapProbe_, &testChannelDesc, testExtent, 1);
                
                if (testErr != hipSuccess) {
                    mipmapsSupported_ = false;
                    std::cerr << "\n*** WARNING: Mipmapped arrays not supported on this GPU ***\n"
                              << "    HIP Error: " << hipGetErrorString(testErr) << "\n"
                              << "    Falling back to non-mipmapped textures for all loads.\n"
                              << "    This may result in aliasing artifacts at distance.\n" << std::endl;
                } else {
                    totalMemoryUsage_ += 4;
                    const hipError_t cleanup = hipFreeMipmappedArray(mipmapProbe_);
                    if (cleanup != hipSuccess) {
                        info.cleanupHipError = cleanup;
                        info.lastError = hipLoaderError(cleanup);
                        lastError_ = info.lastError;
                        return false;
                    }
                    mipmapProbe_ = nullptr;
                    totalMemoryUsage_ -= 4;
                    mipmapsSupported_ = true;
                }
                mipmapsSupportChecked_ = true;
            }
            shouldUseMipmaps = mipmapsSupported_;
        }
        
        // If mipmaps aren't supported, fall back to non-mipmapped path
        if (!shouldUseMipmaps) {
            useMipmaps = false;
        }
    }

    if (useMipmaps) {
        int numLevels = calculateMipLevels(width, height);
        if (desc.maxMipLevel > 0) {
            numLevels = static_cast<int>(std::min(static_cast<unsigned int>(numLevels), desc.maxMipLevel));
        }
        const size_t allocationBytes = internal::mipImageByteSize(image, numLevels);
        {
            std::lock_guard<std::mutex> accounting(mutex_);
            if (allocationBytes > SIZE_MAX - totalMemoryUsage_) {
                info.lastError = LoaderError::OutOfMemory;
                lastError_ = info.lastError;
                return false;
            }
        }
        hipExtent extent = make_hipExtent(width, height, 0);

        err = hipCalls_.call(HipOperation::AllocateMipmapped, [&] {
            return hipMallocMipmappedArray(&info.mipmapArray, &channelDesc, extent, numLevels);
        });
        if (err != hipSuccess) {
            lock.lock();
            info.primaryHipError = err;
            info.lastError = hipLoaderError(err);
            lastError_ = info.lastError;
            return false;
        }
        {
            std::lock_guard<std::mutex> accounting(mutex_);
            info.memoryUsage = allocationBytes;
            totalMemoryUsage_ += info.memoryUsage;
            info.hasMipmaps = true;
            info.numMipLevels = numLevels;
        }

        hipArray_t level0Array;
        err = hipGetMipmappedArrayLevel(&level0Array, info.mipmapArray, 0);
        if (err == hipSuccess) {
            err = hipCalls_.call(HipOperation::Upload, [&] {
                return hipMemcpy2DToArray(level0Array, 0, 0, image.data(), rowBytes,
                                         rowBytes, height, hipMemcpyHostToDevice);
            });
        }

        if (err == hipSuccess) {
            try {
                success = generateMipLevels(info.mipmapArray, image, numLevels,
                                               imageSource.get(), desc.sRGB, err);
            } catch (const std::bad_alloc&) {
                sourceError = LoaderError::OutOfMemory;
                logMessage(LogLevel::Error, "loadStorage: mip generation allocation failed");
                success = false;
            } catch (const std::exception& e) {
                logMessage(LogLevel::Error, "loadTexture: mip generation failed: %s", e.what());
                success = false;
            }
        }

    } else {
        {
            std::lock_guard<std::mutex> accounting(mutex_);
            if (image.sizeBytes() > SIZE_MAX - totalMemoryUsage_) {
                info.lastError = LoaderError::OutOfMemory;
                lastError_ = info.lastError;
                return false;
            }
        }
        err = hipCalls_.call(HipOperation::AllocateArray, [&] {
            return hipMallocArray(&info.array, &channelDesc, width, height);
        });

        if (err == hipSuccess) {
            {
                std::lock_guard<std::mutex> accounting(mutex_);
                info.memoryUsage = image.sizeBytes();
                totalMemoryUsage_ += info.memoryUsage;
                info.hasMipmaps = false;
                info.numMipLevels = 1;
            }
            err = hipCalls_.call(HipOperation::Upload, [&] {
                return hipMemcpy2DToArray(info.array, 0, 0, image.data(), rowBytes,
                                         rowBytes, height, hipMemcpyHostToDevice);
            });
        }

        success = (err == hipSuccess);
    }

    if (!success) {
        lock.lock();
        info.primaryHipError = err;
        cleanupStorageResources(info);
        info.lastError = err == hipSuccess ? sourceError : hipLoaderError(err);
        lastError_ = info.lastError;
        logMessage(LogLevel::Error, "loadStorage: primary HIP error=%d cleanup HIP error=%d",
                   static_cast<int>(info.primaryHipError), static_cast<int>(info.cleanupHipError));
        return false;
    }

    // Publish results under lock
    lock.lock();
    info.lastError = LoaderError::Success;
    info.width = width;
    info.height = height;
    info.uploadBytesPerPixel = static_cast<unsigned int>(image.bytesPerPixel());
    info.floatPixels = image.isFloat();
    if (!info.imageSource && imageSource) {
        info.imageSource = std::move(imageSource);
        info.sourceInfo = loadedSourceInfo;
    }
    info.uploaded = true;
    info.lastUsedFrame = currentFrame_;
    info.loadedFrame = currentFrame_;
    logMessage(LogLevel::Info, "loadStorage: size=%dx%d mipLevels=%d mem=%.2f MB total=%.2f MB",
               info.width, info.height, info.numMipLevels,
               static_cast<double>(info.memoryUsage) / (1024.0 * 1024.0),
               static_cast<double>(totalMemoryUsage_) / (1024.0 * 1024.0));

    return true;
}

bool DemandTextureLoader::Impl::generateMipLevels(hipMipmappedArray_t mipmapArray,
                                                   const internal::ImageData& baseImage,
                                                   int numLevels, ImageSource* source,
                                                   bool sourceSRGB, hipError_t& error) {
    internal::ImageData ownedCurrent;
    const internal::ImageData* current = &baseImage;
    for (int level = 1; level < numLevels; ++level) {
        internal::ImageData next;
        if (source && static_cast<unsigned int>(level) < source->getInfo().numMipLevels) {
            if (!internal::readImageSource(*source, next, level))
                return false;
            if (sourceSRGB)
                internal::linearizeFloatSRGB(next);
            if (next.width != std::max(1u, current->width / 2) ||
                next.height != std::max(1u, current->height / 2) ||
                next.isFloat() != baseImage.isFloat())
                return false;
        } else {
            next = internal::downsampleImage(*current, sourceSRGB);
        }
        hipArray_t levelArray;
        error = hipGetMipmappedArrayLevel(&levelArray, mipmapArray, level);
        if (error != hipSuccess)
            return false;
        error = hipCalls_.call(HipOperation::Upload, [&] {
            return hipMemcpy2DToArray(levelArray, 0, 0, next.data(), next.rowBytes(),
                                     next.rowBytes(), next.height, hipMemcpyHostToDevice);
        });
        if (error != hipSuccess)
            return false;
        ownedCurrent = std::move(next);
        current = &ownedCurrent;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Texture Unloading & Eviction
// -----------------------------------------------------------------------------

bool DemandTextureLoader::Impl::selectDevice() {
    if (initializationError_ != LoaderError::Success) {
        lastError_ = initializationError_;
        return false;
    }
    const hipError_t error = hipSetDevice(device_);
    if (error != hipSuccess) {
        lastError_ = hipLoaderError(error);
        logMessage(LogLevel::Error, "Loader device selection failed: %s", hipGetErrorString(error));
    }
    return error == hipSuccess;
}

bool DemandTextureLoader::Impl::quiesce(bool retiring) {
    if (!selectDevice())
        return false;
    const hipError_t error = retiring
        ? hipCalls_.call(HipOperation::SynchronizeConsumers, [] { return hipDeviceSynchronize(); })
        : hipDeviceSynchronize();
    if (error != hipSuccess) {
        lastError_ = hipLoaderError(error);
        logMessage(LogLevel::Error, "Loader consumer synchronization failed: %s", hipGetErrorString(error));
    }
    return error == hipSuccess;
}

bool DemandTextureLoader::Impl::publishMappingsLocked(bool retiring) {
    const auto operation = retiring ? HipOperation::InvalidateMappings : HipOperation::PublishMappings;
    const auto textures = [&] {
        if (!texturesDirty_)
            return hipSuccess;
        const size_t begin = dirtyTextureBegin_;
        return hipCalls_.call(operation, [&] {
            return hipMemcpy(deviceContext_.textures + begin, h_textures_ + begin,
                             (dirtyTextureEnd_ - begin + 1) * sizeof(TextureObject), hipMemcpyHostToDevice);
        });
    };
    const auto flags = [&] {
        if (!residentFlagsDirty_)
            return hipSuccess;
        const size_t begin = dirtyResidentWordBegin_;
        return hipCalls_.call(operation, [&] {
            return hipMemcpy(deviceContext_.residentFlags + begin, h_residentFlags_ + begin,
                             (dirtyResidentWordEnd_ - begin + 1) * sizeof(uint32_t), hipMemcpyHostToDevice);
        });
    };
    // A dirty range can mix additions and removals, including retries after a
    // partial failure. Hide it until every new object mapping is in place.
    hipError_t error = hipSuccess;
    if (residentFlagsDirty_) {
        const size_t begin = dirtyResidentWordBegin_;
        error = hipCalls_.call(operation, [&] {
            return hipMemset(deviceContext_.residentFlags + begin, 0,
                             (dirtyResidentWordEnd_ - begin + 1) * sizeof(uint32_t));
        });
    }
    if (error == hipSuccess)
        error = textures();
    if (error == hipSuccess)
        error = flags();
    if (error != hipSuccess) {
        lastError_ = hipLoaderError(error);
        return false;
    }
    clearDirtyLocked();
    return true;
}

bool DemandTextureLoader::Impl::cleanupTextureResources(TextureMetadata& info) {
    if (info.texObj) {
        const hipError_t err = hipCalls_.call(HipOperation::DestroySampler, [&] {
            return hipDestroyTextureObject(info.texObj);
        });
        if (err != hipSuccess) {
            info.cleanupHipError = err;
            lastError_ = hipLoaderError(err);
            return false;
        }
        info.texObj = 0;
    }
    return true;
}

bool DemandTextureLoader::Impl::cleanupStorageResources(ImageStorage& info) {
    for (uint32_t id : info.samplers)
        if (textures_[id].texObj)
            return true;
    if (info.mipmapArray) {
        const hipError_t err = hipCalls_.call(HipOperation::FreeMipmapped, [&] {
            return hipFreeMipmappedArray(info.mipmapArray);
        });
        if (err != hipSuccess) {
            info.cleanupHipError = err;
            lastError_ = hipLoaderError(err);
            return false;
        }
        info.mipmapArray = nullptr;
    }

    if (info.array) {
        const hipError_t err = hipCalls_.call(HipOperation::FreeArray, [&] {
            return hipFreeArray(info.array);
        });
        if (err != hipSuccess) {
            info.cleanupHipError = err;
            lastError_ = hipLoaderError(err);
            return false;
        }
        info.array = nullptr;
    }
    totalMemoryUsage_ -= info.memoryUsage;
    info.memoryUsage = 0;
    info.uploaded = false;
    info.hasMipmaps = false;
    info.numMipLevels = 0;
    return true;
}

void DemandTextureLoader::Impl::destroyTexture(uint32_t texId) {
    TextureMetadata& info = textures_[texId];
    info.resident.store(false, std::memory_order_release);
    h_textures_[texId] = 0;
    uint32_t wordIdx = texId / 32;
    uint32_t bitIdx = texId % 32;
    h_residentFlags_[wordIdx] &= ~(1u << bitIdx);

    markTextureDirtyLocked(texId);
    markResidentWordDirtyLocked(wordIdx);
    if (!quiesce(true) || !publishMappingsLocked(true))
        return;
    if (!cleanupTextureResources(info)) {
        return;
    }
    if (!cleanupStorageResources(*info.storage))
        info.cleanupHipError = info.storage->cleanupHipError;
}

void DemandTextureLoader::Impl::evictIfNeeded(
    size_t requiredMemory, const std::unordered_set<ImageStorage*>& requestedStorage) {
    if (options_.maxTextureMemory == 0) {
        return;
    }

    if (requiredMemory <= options_.maxTextureMemory &&
        totalMemoryUsage_ <= options_.maxTextureMemory - requiredMemory) {
        return;
    }

    logMessage(LogLevel::Debug, "evictIfNeeded: current=%.2f MB required=%.2f MB budget=%.2f MB",
               static_cast<double>(totalMemoryUsage_) / (1024.0 * 1024.0),
               static_cast<double>(requiredMemory) / (1024.0 * 1024.0),
               static_cast<double>(options_.maxTextureMemory) / (1024.0 * 1024.0));

    std::vector<std::tuple<int, uint32_t, uint32_t>> evictionList;
    std::unordered_set<ImageStorage*> visited;
    for (uint32_t i = 0; i < nextTextureId_; ++i) {
        const auto& storage = *textures_[i].storage;
        if (requestedStorage.count(textures_[i].storage.get()))
            continue;
        if (!storage.memoryUsage || !visited.insert(textures_[i].storage.get()).second)
            continue;
        if (currentFrame_ - storage.loadedFrame < options_.minResidentFrames)
            continue;
        int priorityScore = 0;
        for (uint32_t id : storage.samplers) {
            int score = 0;
            switch (textures_[id].desc.evictionPriority) {
                case EvictionPriority::Low:          score = 0; break;
                case EvictionPriority::Normal:       score = 1; break;
                case EvictionPriority::High:         score = 2; break;
                case EvictionPriority::KeepResident: score = 3; break;
            }
            priorityScore = std::max(priorityScore, score);
        }
        if (priorityScore < 3)
            evictionList.push_back({priorityScore, storage.lastUsedFrame, i});
    }

    // Sort by priority first, then by age (oldest first within same priority)
    std::sort(evictionList.begin(), evictionList.end());

    size_t targetMemory = requiredMemory < options_.maxTextureMemory ?
                            options_.maxTextureMemory - requiredMemory : 0;
    for (const auto& [priority, frame, texId] : evictionList) {
        if (totalMemoryUsage_ <= targetMemory) {
            break;
        }
        logMessage(LogLevel::Debug, "evictIfNeeded: evicting texture %u (priority=%d, lastUsed=%u)",
                   texId, priority, frame);
        const auto storage = textures_[texId].storage;
        for (uint32_t id : storage->samplers) {
            textures_[id].resident.store(false, std::memory_order_release);
            h_textures_[id] = 0;
            h_residentFlags_[id / 32] &= ~(1u << (id % 32));
            markTextureDirtyLocked(id);
            markResidentWordDirtyLocked(id / 32);
        }
        // Publish every sibling's invalid mapping before destroying any object.
        if (!quiesce(true) || !publishMappingsLocked(true))
            return;
        for (uint32_t id : storage->samplers)
            cleanupTextureResources(textures_[id]);
        cleanupStorageResources(*storage);
    }
}

void DemandTextureLoader::Impl::cancelTextureLocked(uint32_t texId) {
    // Saturation permanently rejects further work rather than wrapping an
    // epoch into a stale request. This is cancellation, not ID reclamation.
    auto& epoch = textures_[texId].cancellationEpoch;
    if (epoch != UINT64_MAX)
        ++epoch;
}

void DemandTextureLoader::Impl::unloadTexture(uint32_t texId) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (texId >= nextTextureId_) {
            lastError_ = LoaderError::InvalidTextureId;
            logMessage(LogLevel::Error, "unloadTexture: invalid texture ID %u", texId);
            return;
        }
        cancelTextureLocked(texId);
    }
    // Waiting outside mutex_ lets a blocked decoder finish and observe cancel.
    std::lock_guard<std::mutex> operation(operationMutex_);
    std::lock_guard<std::mutex> lock(mutex_);
    // Also cancel snapshots captured while this call was waiting for an
    // operation already in progress to release the coordinator.
    cancelTextureLocked(texId);
    destroyTexture(texId);
}

void DemandTextureLoader::Impl::unloadAll() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (uint32_t i = 0; i < nextTextureId_; ++i)
            cancelTextureLocked(i);
    }
    std::lock_guard<std::mutex> operation(operationMutex_);
    std::lock_guard<std::mutex> lock(mutex_);
    for (uint32_t i = 0; i < nextTextureId_; ++i) {
        cancelTextureLocked(i);
        textures_[i].resident.store(false, std::memory_order_release);
        h_textures_[i] = 0;
        h_residentFlags_[i / 32] &= ~(1u << (i % 32));
        markTextureDirtyLocked(i);
        markResidentWordDirtyLocked(i / 32);
    }
    if (nextTextureId_) {
        if (!quiesce(true) || !publishMappingsLocked(true))
            return;
        for (uint32_t i = 0; i < nextTextureId_; ++i)
            cleanupTextureResources(textures_[i]);
        for (uint32_t i = 0; i < nextTextureId_; ++i)
            if (textures_[i].storage->samplers.front() == i)
                cleanupStorageResources(*textures_[i].storage);
    }
    if (mipmapProbe_ && selectDevice()) {
        const hipError_t error = hipFreeMipmappedArray(mipmapProbe_);
        if (error == hipSuccess) {
            mipmapProbe_ = nullptr;
            totalMemoryUsage_ -= 4;
        } else {
            lastError_ = hipLoaderError(error);
        }
    }
}

// -----------------------------------------------------------------------------
// Statistics & Configuration
// -----------------------------------------------------------------------------

size_t DemandTextureLoader::Impl::getResidentTextureCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (uint32_t i = 0; i < nextTextureId_; ++i) {
        if (textures_[i].resident.load(std::memory_order_relaxed)) count++;
    }
    return count;
}

size_t DemandTextureLoader::Impl::getTotalTextureMemory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return totalMemoryUsage_;
}

size_t DemandTextureLoader::Impl::getRequestCount() const {
    return lastRequestCount_.load(std::memory_order_acquire);
}

bool DemandTextureLoader::Impl::hadRequestOverflow() const {
    return lastRequestOverflow_.load(std::memory_order_acquire);
}

LoaderError DemandTextureLoader::Impl::getLastError() const {
    return lastError_.load(std::memory_order_relaxed);
}

void DemandTextureLoader::Impl::enableEviction(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    options_.enableEviction = enable;
}

void DemandTextureLoader::Impl::setMaxTextureMemory(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    options_.maxTextureMemory = bytes;
}

size_t DemandTextureLoader::Impl::getMaxTextureMemory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return options_.maxTextureMemory;
}

void DemandTextureLoader::Impl::updateEvictionPriority(uint32_t texId, EvictionPriority priority) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (texId < nextTextureId_) {
        TextureDesc desc = textures_[texId].desc;
        desc.evictionPriority = priority;
        if (!validDescriptor(desc)) {
            lastError_ = LoaderError::InvalidParameter;
            logMessage(LogLevel::Error, "updateEvictionPriority: invalid priority for texture %u", texId);
            return;
        }
        textures_[texId].desc = desc;
        textures_[texId].descriptorHash = internal::TextureDescHash{}(desc);
    } else {
        lastError_ = LoaderError::InvalidTextureId;
        logMessage(LogLevel::Error, "updateEvictionPriority: invalid texture ID %u", texId);
    }
}

// -----------------------------------------------------------------------------
// Abort
// -----------------------------------------------------------------------------

void DemandTextureLoader::Impl::abort() {
    // Set aborted flag first to prevent new operations from starting
    aborted_.store(true, std::memory_order_seq_cst);
    
    logMessage(LogLevel::Info, "abort: halting all operations");
    
    // Wait for all in-flight async operations to complete
    {
        std::unique_lock<std::mutex> lock(asyncMutex_);
        asyncCv_.wait(lock, [&] { return inFlightAsync_.load(std::memory_order_acquire) == 0; });
    }
    
    // Runtime teardown, including synchronous request processing, is serialized.
    unloadAll();
    std::lock_guard<std::mutex> operation(operationMutex_);
    if (deviceKnown_)
        hipSetDevice(device_);

    // Stop thread pool from accepting new work and wait for current tasks
    if (threadPool_) {
        threadPool_.reset();
    }
    
    // Release pinned memory pool (frees all pooled pinned buffers)
    if (pinnedMemoryPool_) {
        pinnedMemoryPool_.reset();
    }
    
    // Release HIP event pool (destroys all pooled events)
    if (hipEventPool_) {
        hipEventPool_.reset();
    }
    
    logMessage(LogLevel::Info, "abort: completed gracefully");
}

bool DemandTextureLoader::Impl::isAborted() const {
    return aborted_.load(std::memory_order_acquire);
}

} // namespace hip_demand
