// SPDX-License-Identifier: MIT
// DemandTextureLoader implementation
#include <hip/hip_runtime.h>
#include "DemandTextureLoaderImpl.h"
#include "Internal/TextureMetadata.h"
#include "Internal/Utils.h"

#include <DemandLoading/Logging.h>
#include <DemandLoading/Ticket.h>

#ifdef USE_OIIO
#include <ImageSource/ImageSource.h>
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
using internal::RequestStats;
using internal::calculateMipmapMemory;
using internal::calculateMipLevels;

// -----------------------------------------------------------------------------
// Constructor / Destructor
// -----------------------------------------------------------------------------

DemandTextureLoader::Impl::Impl(const LoaderOptions& options)
    : options_(options), device_(0)
{
    hipError_t err = hipGetDevice(&device_);
    if (err != hipSuccess) {
        lastError_ = LoaderError::HipError;
        return;
    }

    // Create dedicated stream for async request copies.
    err = hipStreamCreateWithFlags(&requestCopyStream_, hipStreamNonBlocking);
    if (err != hipSuccess) {
        lastError_ = LoaderError::HipError;
        return;
    }

    // Allocate device memory for request buffer
    err = hipMalloc(&deviceContext_.requests, options_.maxRequestsPerLaunch * sizeof(uint32_t));
    if (err != hipSuccess) {
        lastError_ = LoaderError::OutOfMemory;
        return;
    }

    // Allocate device memory for texture array
    err = hipMalloc(&deviceContext_.textures, options_.maxTextures * sizeof(TextureObject));
    if (err != hipSuccess) {
        lastError_ = LoaderError::OutOfMemory;
        hipFree(deviceContext_.requests);
        deviceContext_.requests = nullptr;
        return;
    }

    // Allocate device memory for resident flags (32 textures per word)
    size_t flagWords = (options_.maxTextures + 31) / 32;
    err = hipMalloc(&deviceContext_.residentFlags, flagWords * sizeof(uint32_t));
    if (err != hipSuccess) {
        lastError_ = LoaderError::OutOfMemory;
        hipFree(deviceContext_.requests);
        hipFree(deviceContext_.textures);
        deviceContext_.requests = nullptr;
        deviceContext_.textures = nullptr;
        return;
    }

    err = hipMalloc(&d_requestStats_, sizeof(RequestStats));
    if (err != hipSuccess) {
        lastError_ = LoaderError::OutOfMemory;
        hipFree(deviceContext_.requests);
        hipFree(deviceContext_.textures);
        hipFree(deviceContext_.residentFlags);
        deviceContext_.requests = nullptr;
        deviceContext_.textures = nullptr;
        deviceContext_.residentFlags = nullptr;
        return;
    }
    deviceContext_.requestCount = reinterpret_cast<uint32_t*>(d_requestStats_);
    deviceContext_.requestOverflow = deviceContext_.requestCount + 1;

    // Initialize to zero
    err = hipMemset(deviceContext_.residentFlags, 0, flagWords * sizeof(uint32_t));
    if (err != hipSuccess) lastError_ = LoaderError::HipError;

    err = hipMemset(deviceContext_.textures, 0, options_.maxTextures * sizeof(TextureObject));
    if (err != hipSuccess) lastError_ = LoaderError::HipError;

    err = hipMemset(d_requestStats_, 0, sizeof(RequestStats));
    if (err != hipSuccess) lastError_ = LoaderError::HipError;

    // Set limits
    deviceContext_.maxTextures = options_.maxTextures;
    deviceContext_.maxRequests = options_.maxRequestsPerLaunch;

    // Allocate host pinned buffers for async copies
    flagWordCount_ = flagWords;
    if (hipHostMalloc(reinterpret_cast<void**>(&h_residentFlags_), flagWords * sizeof(uint32_t)) != hipSuccess) {
        lastError_ = LoaderError::OutOfMemory;
        return;
    }
    if (hipHostMalloc(reinterpret_cast<void**>(&h_textures_), options_.maxTextures * sizeof(TextureObject)) != hipSuccess) {
        lastError_ = LoaderError::OutOfMemory;
        hipHostFree(h_residentFlags_);
        h_residentFlags_ = nullptr;
        return;
    }
    if (hipHostMalloc(reinterpret_cast<void**>(&h_requests_), options_.maxRequestsPerLaunch * sizeof(uint32_t)) != hipSuccess) {
        lastError_ = LoaderError::OutOfMemory;
        hipHostFree(h_residentFlags_);
        hipHostFree(h_textures_);
        h_residentFlags_ = nullptr;
        h_textures_ = nullptr;
        return;
    }
    if (hipHostMalloc(reinterpret_cast<void**>(&h_requestStats_), sizeof(RequestStats)) != hipSuccess) {
        lastError_ = LoaderError::OutOfMemory;
        hipHostFree(h_residentFlags_);
        hipHostFree(h_textures_);
        hipHostFree(h_requests_);
        h_residentFlags_ = nullptr;
        h_textures_ = nullptr;
        h_requests_ = nullptr;
        return;
    }

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
}

DemandTextureLoader::Impl::~Impl() {
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
        hipStreamDestroy(requestCopyStream_);
        requestCopyStream_ = nullptr;
    }

    unloadAll();

    if (h_residentFlags_) hipHostFree(h_residentFlags_);
    if (h_textures_) hipHostFree(h_textures_);
    if (h_requests_) hipHostFree(h_requests_);
    if (h_requestStats_) hipHostFree(h_requestStats_);

    if (deviceContext_.residentFlags) hipFree(deviceContext_.residentFlags);
    if (deviceContext_.textures) hipFree(deviceContext_.textures);
    if (deviceContext_.requests) hipFree(deviceContext_.requests);
    if (d_requestStats_) hipFree(d_requestStats_);
}

// AsyncGuard destructor
DemandTextureLoader::Impl::AsyncGuard::~AsyncGuard() {
    if (!committed) {
        self->inFlightAsync_.fetch_sub(1, std::memory_order_acq_rel);
        std::lock_guard<std::mutex> lock(self->asyncMutex_);
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

    // Check for duplicate filename (same file already loaded)
    size_t filenameHash = std::hash<std::string>{}(filename);
    auto it = filenameHashToTextureId_.find(filenameHash);
    if (it != filenameHashToTextureId_.end()) {
        uint32_t existingId = it->second;
        TextureMetadata& existing = textures_[existingId];
        // Verify it's actually the same file (hash collision check)
        if (existing.filename == filename) {
            logMessage(LogLevel::Debug, "createTexture: reusing existing texture id=%u for '%s'", existingId, filename.c_str());
            return TextureHandle{existingId, true, existing.width, existing.height, existing.channels, LoaderError::Success};
        }
    }

    if (nextTextureId_ >= options_.maxTextures) {
        lastError_ = LoaderError::MaxTexturesExceeded;
        logMessage(LogLevel::Error, "createTexture: max textures exceeded (%zu)", static_cast<size_t>(options_.maxTextures));
        return TextureHandle{0, false, 0, 0, 0, lastError_};
    }

    uint32_t id = nextTextureId_++;

    // Register in deduplication map
    filenameHashToTextureId_[filenameHash] = id;

    TextureMetadata& info = textures_[id];
    info.filename = filename;
    info.desc = desc;
    info.resident.store(false, std::memory_order_relaxed);
    info.loading.store(false, std::memory_order_relaxed);

    // Try to get image dimensions without loading
#ifdef USE_OIIO
    // Try OIIO first for better format support
    try {
        std::unique_ptr<ImageSource> imgSrc = createImageSource(filename);
        if (imgSrc) {
            hip_demand::TextureInfo texInfo;
            imgSrc->open(&texInfo);
            if (imgSrc->isOpen()) {
                info.width = texInfo.width;
                info.height = texInfo.height;
                info.channels = 4;  // OIIO always converts to RGBA
                imgSrc->close();
            } else {
                // Fall back to stb_image
                int w, h, c;
                if (stbi_info(filename.c_str(), &w, &h, &c)) {
                    info.width = w;
                    info.height = h;
                    info.channels = c;
                } else {
                    info.lastError = LoaderError::FileNotFound;
                }
            }
        }
    } catch (...) {
        // Fall back to stb_image on any exception
        int w, h, c;
        if (stbi_info(filename.c_str(), &w, &h, &c)) {
            info.width = w;
            info.height = h;
            info.channels = c;
        } else {
            info.lastError = LoaderError::FileNotFound;
        }
    }
#else
    int w, h, c;
    if (stbi_info(filename.c_str(), &w, &h, &c)) {
        info.width = w;
        info.height = h;
        info.channels = c;
    } else {
        info.lastError = LoaderError::FileNotFound;
        logMessage(LogLevel::Warn, "createTexture: file not found '%s'", filename.c_str());
    }
#endif

    lastError_ = LoaderError::Success;
    logMessage(LogLevel::Debug, "createTexture: queued '%s' as id=%u (%dx%d ch=%d)", filename.c_str(), id, info.width, info.height, info.channels);
    return TextureHandle{id, true, info.width, info.height, info.channels, LoaderError::Success};
}

TextureHandle DemandTextureLoader::Impl::createTexture(std::shared_ptr<ImageSource> imageSource, const TextureDesc& desc) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!imageSource) {
        lastError_ = LoaderError::InvalidParameter;
        logMessage(LogLevel::Error, "createTexture: null ImageSource");
        return TextureHandle{0, false, 0, 0, 0, lastError_};
    }

    // First check: same ImageSource pointer already registered
    ImageSource* rawPtr = imageSource.get();
    auto ptrIt = imageSourceToTextureId_.find(rawPtr);
    if (ptrIt != imageSourceToTextureId_.end()) {
        uint32_t existingId = ptrIt->second;
        TextureMetadata& existing = textures_[existingId];
        logMessage(LogLevel::Debug, "createTexture: reusing existing texture id=%u for ImageSource %p", existingId, rawPtr);
        return TextureHandle{existingId, true, existing.width, existing.height, existing.channels, LoaderError::Success};
    }

    // Second check: content-based hash deduplication (like OptiX coalesceDuplicateImages)
    // This catches different ImageSource objects that point to the same underlying image
    unsigned long long contentHash = imageSource->getHash(0);
    if (contentHash != 0) {
        auto hashIt = filenameHashToTextureId_.find(static_cast<size_t>(contentHash));
        if (hashIt != filenameHashToTextureId_.end()) {
            uint32_t existingId = hashIt->second;
            TextureMetadata& existing = textures_[existingId];
            // Also register pointer mapping for faster future lookups
            imageSourceToTextureId_[rawPtr] = existingId;
            logMessage(LogLevel::Debug, "createTexture: reusing existing texture id=%u via content hash", existingId);
            return TextureHandle{existingId, true, existing.width, existing.height, existing.channels, LoaderError::Success};
        }
    }

    if (nextTextureId_ >= options_.maxTextures) {
        lastError_ = LoaderError::MaxTexturesExceeded;
        logMessage(LogLevel::Error, "createTexture: max textures exceeded (%zu)", static_cast<size_t>(options_.maxTextures));
        return TextureHandle{0, false, 0, 0, 0, lastError_};
    }

    uint32_t id = nextTextureId_++;

    // Register in both deduplication maps
    imageSourceToTextureId_[rawPtr] = id;
    if (contentHash != 0) {
        filenameHashToTextureId_[static_cast<size_t>(contentHash)] = id;
    }

    TextureMetadata& info = textures_[id];
    info.imageSource = std::move(imageSource);
    info.desc = desc;
    info.resident.store(false, std::memory_order_relaxed);
    info.loading.store(false, std::memory_order_relaxed);

    // Get image dimensions from ImageSource
    try {
        hip_demand::TextureInfo texInfo;
        info.imageSource->open(&texInfo);
        if (info.imageSource->isOpen()) {
            info.width = texInfo.width;
            info.height = texInfo.height;
            info.channels = texInfo.numChannels;
            // Don't close - keep open for later reading, or let ImageSource manage state
        } else {
            info.lastError = LoaderError::ImageLoadFailed;
            logMessage(LogLevel::Warn, "createTexture: failed to open ImageSource");
        }
    } catch (const std::exception& e) {
        info.lastError = LoaderError::ImageLoadFailed;
        logMessage(LogLevel::Error, "createTexture: ImageSource exception: %s", e.what());
    } catch (...) {
        info.lastError = LoaderError::ImageLoadFailed;
        logMessage(LogLevel::Error, "createTexture: unknown ImageSource exception");
    }

    lastError_ = LoaderError::Success;
    logMessage(LogLevel::Debug, "createTexture: queued ImageSource as id=%u (%dx%d ch=%d)", id, info.width, info.height, info.channels);
    return TextureHandle{id, true, info.width, info.height, info.channels, LoaderError::Success};
}

TextureHandle DemandTextureLoader::Impl::createTextureFromMemory(const void* data, int width, int height,
                                                                  int channels, const TextureDesc& desc) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!data || width <= 0 || height <= 0 || channels <= 0) {
        lastError_ = LoaderError::InvalidParameter;
        logMessage(LogLevel::Error, "createTextureFromMemory: invalid parameters (w=%d h=%d ch=%d)", width, height, channels);
        return TextureHandle{0, false, 0, 0, 0, lastError_};
    }

    if (nextTextureId_ >= options_.maxTextures) {
        lastError_ = LoaderError::MaxTexturesExceeded;
        logMessage(LogLevel::Error, "createTextureFromMemory: max textures exceeded (%zu)", static_cast<size_t>(options_.maxTextures));
        return TextureHandle{0, false, 0, 0, 0, lastError_};
    }

    uint32_t id = nextTextureId_++;

    TextureMetadata& info = textures_[id];
    info.filename = "";  // Memory-based texture
    info.desc = desc;
    info.width = width;
    info.height = height;
    info.channels = channels;
    info.resident.store(false, std::memory_order_relaxed);
    info.loading.store(false, std::memory_order_relaxed);

    // Cache the data
    size_t dataSize = width * height * channels;
    info.cachedData = std::make_unique<uint8_t[]>(dataSize);
    std::memcpy(info.cachedData.get(), data, dataSize);

    lastError_ = LoaderError::Success;
    logMessage(LogLevel::Debug, "createTextureFromMemory: created id=%u (%dx%d ch=%d)", id, width, height, channels);
    return TextureHandle{id, true, width, height, channels, LoaderError::Success};
}

// -----------------------------------------------------------------------------
// Launch Prepare
// -----------------------------------------------------------------------------

void DemandTextureLoader::Impl::launchPrepare(hipStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Upload only dirty ranges for resident flags and texture objects.
    hipError_t err = hipSuccess;

    if (residentFlagsDirty_ || texturesDirty_) {
        size_t residentWords = 0;
        size_t textureCount = 0;
        if (residentFlagsDirty_ && dirtyResidentWordBegin_ != std::numeric_limits<size_t>::max() && dirtyResidentWordBegin_ <= dirtyResidentWordEnd_) {
            residentWords = (dirtyResidentWordEnd_ - dirtyResidentWordBegin_ + 1);
        }
        if (texturesDirty_ && dirtyTextureBegin_ != std::numeric_limits<size_t>::max() && dirtyTextureBegin_ <= dirtyTextureEnd_) {
            textureCount = (dirtyTextureEnd_ - dirtyTextureBegin_ + 1);
        }
        logMessage(LogLevel::Debug,
                   "launchPrepare: dirty residentWords=%zu (%.1f KB) textures=%zu (%.1f KB)",
                   residentWords,
                   static_cast<double>(residentWords * sizeof(uint32_t)) / 1024.0,
                   textureCount,
                   static_cast<double>(textureCount * sizeof(TextureObject)) / 1024.0);
    }

    if (residentFlagsDirty_) {
        const size_t begin = dirtyResidentWordBegin_;
        const size_t end = dirtyResidentWordEnd_;
        if (begin < flagWordCount_ && begin <= end) {
            const size_t countWords = std::min(flagWordCount_ - begin, end - begin + 1);
            err = hipMemcpyAsync(deviceContext_.residentFlags + begin,
                                 h_residentFlags_ + begin,
                                 countWords * sizeof(uint32_t),
                                 hipMemcpyHostToDevice,
                                 stream);
            if (err != hipSuccess) {
                lastError_ = LoaderError::HipError;
                logMessage(LogLevel::Error, "launchPrepare: hipMemcpyAsync(residentFlags dirty) failed: %s", hipGetErrorString(err));
                return;
            }
        }
    }

    if (texturesDirty_) {
        const size_t begin = dirtyTextureBegin_;
        const size_t end = dirtyTextureEnd_;
        if (begin < options_.maxTextures && begin <= end) {
            const size_t count = std::min(options_.maxTextures - begin, end - begin + 1);
            err = hipMemcpyAsync(deviceContext_.textures + begin,
                                 h_textures_ + begin,
                                 count * sizeof(TextureObject),
                                 hipMemcpyHostToDevice,
                                 stream);
            if (err != hipSuccess) {
                lastError_ = LoaderError::HipError;
                logMessage(LogLevel::Error, "launchPrepare: hipMemcpyAsync(textures dirty) failed: %s", hipGetErrorString(err));
                return;
            }
        }
    }

    clearDirtyLocked();

    // Reset request counter and overflow flag
    err = hipMemsetAsync(d_requestStats_, 0, sizeof(RequestStats), stream);
    if (err != hipSuccess) {
        lastError_ = LoaderError::HipError;
        logMessage(LogLevel::Error, "launchPrepare: hipMemsetAsync(requestStats) failed: %s", hipGetErrorString(err));
        return;
    }

    currentFrame_++;
    logMessage(LogLevel::Debug, "launchPrepare: frame=%u", currentFrame_);
}

DeviceContext DemandTextureLoader::Impl::getDeviceContext() const {
    return deviceContext_;
}

// -----------------------------------------------------------------------------
// Request Processing
// -----------------------------------------------------------------------------

size_t DemandTextureLoader::Impl::processRequests(hipStream_t stream, const DeviceContext& deviceContext) {
    // Early exit if aborted
    if (aborted_.load(std::memory_order_acquire)) {
        return 0;
    }

    uint32_t requestCount = 0;
    uint32_t overflow = 0;

    const uint32_t copyCount = std::min<uint32_t>(static_cast<uint32_t>(options_.maxRequestsPerLaunch), deviceContext.maxRequests);

    hipError_t err = hipMemcpyAsync(&requestCount, deviceContext.requestCount, sizeof(uint32_t),
                                     hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) {
        lastError_ = LoaderError::HipError;
        return 0;
    }

    err = hipMemcpyAsync(&overflow, deviceContext.requestOverflow, sizeof(uint32_t),
                         hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) {
        lastError_ = LoaderError::HipError;
        return 0;
    }

    // Copy the full request list up-front so we only need one stream sync.
    err = hipMemcpyAsync(h_requests_, deviceContext.requests,
                         copyCount * sizeof(uint32_t),
                         hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) {
        lastError_ = LoaderError::HipError;
        return 0;
    }

    err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        lastError_ = LoaderError::HipError;
        return 0;
    }

    lastRequestOverflow_.store(overflow != 0, std::memory_order_release);
    lastRequestCount_.store(static_cast<size_t>(requestCount), std::memory_order_release);
    if (overflow) {
        logMessage(LogLevel::Warn, "processRequests: overflow flagged (count=%u, cap=%zu)", requestCount, static_cast<size_t>(options_.maxRequestsPerLaunch));
    }
    logMessage(LogLevel::Debug, "processRequests: requestCount=%u", requestCount);

    if (requestCount == 0) {
        return 0;
    }

    requestCount = std::min(requestCount, copyCount);
    return processRequestsHost(requestCount, h_requests_);
}

Ticket DemandTextureLoader::Impl::processRequestsAsync(hipStream_t stream, const DeviceContext& deviceContext) {
    // Increment in-flight counter FIRST to prevent race with destructor.
    inFlightAsync_.fetch_add(1, std::memory_order_seq_cst);

    // RAII guard to decrement inFlightAsync_ on any early return
    AsyncGuard asyncGuard{this};

    if (destroying_.load(std::memory_order_seq_cst)) {
        return Ticket{};
    }

    // Early exit if aborted
    if (aborted_.load(std::memory_order_acquire)) {
        return Ticket{};
    }

    // Acquire pinned buffers from pool (reuses existing allocations when possible)
    const size_t requestsBufferSize = options_.maxRequestsPerLaunch * sizeof(uint32_t);
    auto statsBuffer = pinnedMemoryPool_->acquire(sizeof(RequestStats));
    auto requestsBuffer = pinnedMemoryPool_->acquire(requestsBufferSize);
    if (!statsBuffer || !requestsBuffer) {
        lastError_ = LoaderError::OutOfMemory;
        return Ticket{};
    }
    auto* statsPinned = statsBuffer.as<RequestStats>();
    auto* requestsPinned = requestsBuffer.as<uint32_t>();

    statsPinned->count = 0;
    statsPinned->overflow = 0;

    const uint32_t copyCount = std::min<uint32_t>(static_cast<uint32_t>(options_.maxRequestsPerLaunch), deviceContext.maxRequests);

    // Acquire HIP events from pool (avoids expensive hipEventCreate calls)
    hipEvent_t depsReady = hipEventPool_->acquire();
    if (!depsReady) {
        lastError_ = LoaderError::HipError;
        return Ticket{};
    }
    hipEventRecord(depsReady, stream);

    hipStream_t copyStream = requestCopyStream_ ? requestCopyStream_ : stream;
    if (copyStream != stream) {
        hipError_t waitErr = hipStreamWaitEvent(copyStream, depsReady, 0);
        if (waitErr != hipSuccess) {
            hipEventPool_->release(depsReady);
            lastError_ = LoaderError::HipError;
            return Ticket{};
        }
    }

    hipError_t err = hipMemcpyAsync(&statsPinned->count, deviceContext.requestCount, sizeof(uint32_t),
                                    hipMemcpyDeviceToHost, copyStream);
    if (err != hipSuccess) {
        hipEventPool_->release(depsReady);
        lastError_ = LoaderError::HipError;
        return Ticket{};
    }

    err = hipMemcpyAsync(&statsPinned->overflow, deviceContext.requestOverflow, sizeof(uint32_t),
                         hipMemcpyDeviceToHost, copyStream);
    if (err != hipSuccess) {
        lastError_ = LoaderError::HipError;
        return Ticket{};
    }

    err = hipMemcpyAsync(requestsPinned, deviceContext.requests,
                         copyCount * sizeof(uint32_t),
                         hipMemcpyDeviceToHost, copyStream);
    if (err != hipSuccess) {
        hipEventPool_->release(depsReady);
        lastError_ = LoaderError::HipError;
        return Ticket{};
    }

    hipEvent_t copyDone = hipEventPool_->acquire();
    if (!copyDone) {
        hipEventPool_->release(depsReady);
        lastError_ = LoaderError::HipError;
        return Ticket{};
    }
    hipEventRecord(copyDone, copyStream);

    // Bundle resources into a single shared allocation to reduce overhead
    struct AsyncResources {
        internal::PinnedMemoryPool::BufferHandle statsBuffer;
        internal::PinnedMemoryPool::BufferHandle requestsBuffer;
        AsyncResources(internal::PinnedMemoryPool::BufferHandle&& s, 
                       internal::PinnedMemoryPool::BufferHandle&& r)
            : statsBuffer(std::move(s)), requestsBuffer(std::move(r)) {}
    };
    auto resources = std::make_shared<AsyncResources>(std::move(statsBuffer), std::move(requestsBuffer));

    // Capture event pool pointer for returning events
    auto* eventPool = hipEventPool_.get();

    auto task = [this, eventPool, depsReady, copyDone, copyCount, resources]() {
        struct InFlightGuard {
            DemandTextureLoader::Impl* self;
            ~InFlightGuard() {
                self->inFlightAsync_.fetch_sub(1, std::memory_order_acq_rel);
                std::lock_guard<std::mutex> lock(self->asyncMutex_);
                self->asyncCv_.notify_all();
            }
        } guard{this};

        // Always clean up HIP events - return to pool
        hipEventSynchronize(copyDone);
        eventPool->release(copyDone);
        eventPool->release(depsReady);

        // Check if we're being destroyed - if so, skip processing
        if (destroying_.load(std::memory_order_acquire)) {
            return;
        }

        auto* statsPinned = resources->statsBuffer.as<RequestStats>();
        auto* requestsPinned = resources->requestsBuffer.as<uint32_t>();

        uint32_t requestCount = statsPinned->count;
        uint32_t overflow = statsPinned->overflow;
        lastRequestOverflow_.store(overflow != 0, std::memory_order_release);
        lastRequestCount_.store(static_cast<size_t>(requestCount), std::memory_order_release);
        if (overflow) {
            logMessage(LogLevel::Warn, "processRequestsAsync: overflow flagged (count=%u, cap=%zu)", requestCount, static_cast<size_t>(options_.maxRequestsPerLaunch));
        }
        if (requestCount == 0) {
            return;
        }

        requestCount = std::min(requestCount, copyCount);
        processRequestsHost(requestCount, requestsPinned);
    };

    // Mark guard as committed - the task will handle decrementing inFlightAsync_
    asyncGuard.committed = true;

    auto impl = createTicketImpl(std::move(task), stream);
    return Ticket(std::move(impl));
}

size_t DemandTextureLoader::Impl::processRequestsHost(uint32_t requestCount, const uint32_t* requests) {
    // Deduplicate requests and gather texture info under lock
    std::unordered_set<uint32_t> uniqueRequests;
    std::vector<uint32_t> toLoad;
    size_t estimatedMemoryNeeded = 0;

    {
        std::lock_guard<std::mutex> lock(mutex_);

        for (size_t i = 0; i < requestCount; ++i) {
            uint32_t texId = requests[i];
            if (texId < nextTextureId_ && !textures_[texId].resident.load(std::memory_order_relaxed)) {
                if (uniqueRequests.insert(texId).second) {
                    toLoad.push_back(texId);
                    // Calculate actual memory needed
                    const TextureMetadata& info = textures_[texId];
                    int w = info.width;
                    int h = info.height;
                    if (w > 0 && h > 0) {
                        size_t mipMemory = calculateMipmapMemory(w, h, 4);
                        estimatedMemoryNeeded += mipMemory;
                    }
                }
            }
        }
        logMessage(LogLevel::Debug, "processRequests: unique-to-load=%zu estMem=%.2f MB", toLoad.size(), static_cast<double>(estimatedMemoryNeeded) / (1024.0 * 1024.0));

        // Check if we need eviction (with actual size estimates)
        if (options_.enableEviction && options_.maxTextureMemory > 0 && estimatedMemoryNeeded > 0) {
            evictIfNeeded(estimatedMemoryNeeded);
        }
    }

    // Load textures in parallel using thread pool
    std::atomic<size_t> loaded{0};
    
    if (toLoad.size() == 1 || !threadPool_) {
        // Single texture or no pool - load directly
        for (uint32_t texId : toLoad) {
            if (loadTextureThreadSafe(texId)) {
                loaded.fetch_add(1, std::memory_order_relaxed);
            }
        }
    } else {
        // Parallel loading via thread pool
        for (uint32_t texId : toLoad) {
            threadPool_->submit([this, texId, &loaded]() {
                if (loadTextureThreadSafe(texId)) {
                    loaded.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        threadPool_->waitAll();
    }

    return loaded.load(std::memory_order_relaxed);
}

// -----------------------------------------------------------------------------
// Texture Loading
// -----------------------------------------------------------------------------

bool DemandTextureLoader::Impl::loadTextureThreadSafe(uint32_t texId) {
    return loadTexture(texId);
}

bool DemandTextureLoader::Impl::loadTexture(uint32_t texId) {
    // Early exit if aborted
    if (aborted_.load(std::memory_order_acquire)) {
        return false;
    }

    // Double-checked locking pattern with atomic loading flag
    TextureMetadata& info = textures_[texId];
    if (info.resident.load(std::memory_order_acquire) ||
        info.loading.load(std::memory_order_acquire)) {
        return false;
    }

    // Try to atomically claim the loading slot
    bool expected = false;
    if (!info.loading.compare_exchange_strong(expected, true,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;
    }

    // We now own the loading flag - gather data under lock
    std::unique_lock<std::mutex> lock(mutex_);
    if (info.resident.load(std::memory_order_acquire)) {
        info.loading.store(false, std::memory_order_release);
        return false;
    }
    TextureDesc desc = info.desc;
    std::string filename = info.filename;
    std::shared_ptr<ImageSource> imageSource = info.imageSource;
    int initWidth = info.width;
    int initHeight = info.height;
    int initChannels = info.channels;
    const unsigned char* cachedPtr = info.cachedData.get();
    bool hasCached = (cachedPtr != nullptr);
    lock.unlock();

    // Load image data
    unsigned char* data = nullptr;
    bool needsFree = false;
    int width = initWidth;
    int height = initHeight;
    int channels = initChannels;

    // Priority: 1) user-provided ImageSource, 2) filename, 3) cached memory
    if (imageSource) {
        // Load from user-provided ImageSource
        try {
            hip_demand::TextureInfo texInfo;
            if (!imageSource->isOpen()) {
                imageSource->open(&texInfo);
            } else {
                texInfo = imageSource->getInfo();
            }
            
            if (imageSource->isOpen()) {
                width = texInfo.width;
                height = texInfo.height;
                channels = texInfo.numChannels;
                
                // Always convert to 4 channels for GPU texture
                size_t imageSize = width * height * 4;
                data = new unsigned char[imageSize];
                needsFree = true;
                
                if (channels == 4) {
                    if (!imageSource->readMipLevel(reinterpret_cast<char*>(data), 0, width, height)) {
                        delete[] data;
                        data = nullptr;
                        needsFree = false;
                    }
                } else {
                    // Read native channels then convert
                    std::vector<unsigned char> tempData(width * height * channels);
                    if (imageSource->readMipLevel(reinterpret_cast<char*>(tempData.data()), 0, width, height)) {
                        for (size_t i = 0; i < static_cast<size_t>(width * height); ++i) {
                            if (channels == 1) {
                                data[i*4+0] = tempData[i];
                                data[i*4+1] = tempData[i];
                                data[i*4+2] = tempData[i];
                                data[i*4+3] = 255;
                            } else if (channels == 3) {
                                data[i*4+0] = tempData[i*3+0];
                                data[i*4+1] = tempData[i*3+1];
                                data[i*4+2] = tempData[i*3+2];
                                data[i*4+3] = 255;
                            }
                        }
                    } else {
                        delete[] data;
                        data = nullptr;
                        needsFree = false;
                    }
                }
                channels = 4;
            }
        } catch (const std::exception& e) {
            logMessage(LogLevel::Error, "loadTexture: ImageSource exception: %s", e.what());
            if (data && needsFree) {
                delete[] data;
                data = nullptr;
                needsFree = false;
            }
        } catch (...) {
            logMessage(LogLevel::Error, "loadTexture: unknown ImageSource exception");
            if (data && needsFree) {
                delete[] data;
                data = nullptr;
                needsFree = false;
            }
        }
        
        if (!data) {
            lock.lock();
            info.loading.store(false, std::memory_order_release);
            info.lastError = LoaderError::ImageLoadFailed;
            logMessage(LogLevel::Error, "loadTexture: failed to load from ImageSource");
            return false;
        }
    } else if (!filename.empty()) {
#ifdef USE_OIIO
        bool oiioSuccess = false;
        try {
            std::unique_ptr<ImageSource> imgSrc = createImageSource(filename);
            if (imgSrc) {
                hip_demand::TextureInfo texInfo;
                imgSrc->open(&texInfo);
                if (imgSrc->isOpen()) {
                    width = texInfo.width;
                    height = texInfo.height;
                    channels = 4;

                    size_t imageSize = width * height * 4;
                    data = new unsigned char[imageSize];

                    if (imgSrc->readMipLevel(reinterpret_cast<char*>(data), 0, width, height)) {
                        needsFree = true;
                        oiioSuccess = true;
                    } else {
                        delete[] data;
                        data = nullptr;
                    }
                    imgSrc->close();
                }
            }
        } catch (...) {
            if (data) {
                delete[] data;
                data = nullptr;
            }
            oiioSuccess = false;
        }

        if (!oiioSuccess) {
#endif
            data = stbi_load(filename.c_str(), &width, &height, &channels, 4);
            if (!data) {
                lock.lock();
                info.loading.store(false, std::memory_order_release);
                info.lastError = LoaderError::ImageLoadFailed;
                logMessage(LogLevel::Error, "loadTexture: failed to load image '%s'", filename.c_str());
                return false;
            }
            needsFree = true;
            channels = 4;
#ifdef USE_OIIO
        }
#endif
    } else if (hasCached) {
        width = initWidth;
        height = initHeight;
        channels = initChannels;

        if (channels == 4) {
            data = const_cast<unsigned char*>(cachedPtr);
        } else {
            size_t pixelCount = width * height;
            unsigned char* data4 = new unsigned char[pixelCount * 4];
            needsFree = true;

            for (size_t i = 0; i < pixelCount; ++i) {
                if (channels == 1) {
                    data4[i*4+0] = cachedPtr[i];
                    data4[i*4+1] = cachedPtr[i];
                    data4[i*4+2] = cachedPtr[i];
                    data4[i*4+3] = 255;
                } else if (channels == 3) {
                    data4[i*4+0] = cachedPtr[i*3+0];
                    data4[i*4+1] = cachedPtr[i*3+1];
                    data4[i*4+2] = cachedPtr[i*3+2];
                    data4[i*4+3] = 255;
                }
            }
            data = data4;
            channels = 4;
        }
    } else {
        lock.lock();
        info.loading.store(false, std::memory_order_release);
        info.lastError = LoaderError::InvalidParameter;
        logMessage(LogLevel::Error, "loadTexture: invalid parameters for texId=%u", texId);
        return false;
    }

    int finalWidth = width;
    int finalHeight = height;
    int finalChannels = channels;

    hipError_t err;
    bool success = false;
    bool useMipmaps = desc.generateMipmaps && (width > 1 || height > 1);

    if (useMipmaps) {
        int numLevels = calculateMipLevels(width, height);
        if (desc.maxMipLevel > 0) {
            numLevels = std::min(numLevels, (int)desc.maxMipLevel);
        }

        hipChannelFormatDesc channelDesc = hipCreateChannelDesc<uchar4>();
        hipExtent extent = make_hipExtent(width, height, 0);

        err = hipMallocMipmappedArray(&info.mipmapArray, &channelDesc, extent, numLevels);
        if (err != hipSuccess) {
            if (needsFree) {
                if (!filename.empty()) {
                    stbi_image_free(data);
                } else {
                    delete[] data;
                }
            }
            lock.lock();
            info.loading.store(false, std::memory_order_release);
            info.lastError = LoaderError::OutOfMemory;
            return false;
        }

        hipArray_t level0Array;
        err = hipGetMipmappedArrayLevel(&level0Array, info.mipmapArray, 0);
        if (err == hipSuccess) {
            err = hipMemcpy2DToArray(level0Array, 0, 0, data, width * 4,
                                    width * 4, height, hipMemcpyHostToDevice);
        }

        if (err == hipSuccess) {
            success = generateMipLevels(info.mipmapArray, data, width, height, numLevels);
        }

        if (success) {
            hipResourceDesc resDesc = {};
            resDesc.resType = hipResourceTypeMipmappedArray;
            resDesc.res.mipmap.mipmap = info.mipmapArray;

            hipTextureDesc texDesc = {};
            texDesc.addressMode[0] = desc.addressMode[0];
            texDesc.addressMode[1] = desc.addressMode[1];
            texDesc.filterMode = desc.filterMode;
            texDesc.readMode = hipReadModeNormalizedFloat;
            texDesc.normalizedCoords = desc.normalizedCoords ? 1 : 0;
            texDesc.sRGB = desc.sRGB ? 1 : 0;
            texDesc.maxMipmapLevelClamp = numLevels - 1;
            texDesc.minMipmapLevelClamp = 0;
            texDesc.mipmapFilterMode = hipFilterModeLinear;

            err = hipCreateTextureObject(&info.texObj, &resDesc, &texDesc, nullptr);
            success = (err == hipSuccess);

            if (success) {
                info.hasMipmaps = true;
                info.numMipLevels = numLevels;
                info.memoryUsage = calculateMipmapMemory(width, height, 4);
            }
        }
    } else {
        hipChannelFormatDesc channelDesc = hipCreateChannelDesc<uchar4>();
        err = hipMallocArray(&info.array, &channelDesc, width, height);

        if (err == hipSuccess) {
            err = hipMemcpy2DToArray(info.array, 0, 0, data, width * 4,
                                    width * 4, height, hipMemcpyHostToDevice);
        }

        if (err == hipSuccess) {
            hipResourceDesc resDesc = {};
            resDesc.resType = hipResourceTypeArray;
            resDesc.res.array.array = info.array;

            hipTextureDesc texDesc = {};
            texDesc.addressMode[0] = desc.addressMode[0];
            texDesc.addressMode[1] = desc.addressMode[1];
            texDesc.filterMode = desc.filterMode;
            texDesc.readMode = hipReadModeNormalizedFloat;
            texDesc.normalizedCoords = desc.normalizedCoords ? 1 : 0;
            texDesc.sRGB = desc.sRGB ? 1 : 0;

            err = hipCreateTextureObject(&info.texObj, &resDesc, &texDesc, nullptr);
            success = (err == hipSuccess);

            if (success) {
                info.hasMipmaps = false;
                info.numMipLevels = 1;
                info.memoryUsage = width * height * 4;
            }
        }
    }

    if (needsFree) {
        if (!filename.empty()) {
            stbi_image_free(data);
        } else {
            delete[] data;
        }
    }

    if (!success) {
        if (info.mipmapArray) {
            hipFreeMipmappedArray(info.mipmapArray);
            info.mipmapArray = nullptr;
        }
        if (info.array) {
            hipFreeArray(info.array);
            info.array = nullptr;
        }
        lock.lock();
        info.loading.store(false, std::memory_order_release);
        info.lastError = LoaderError::HipError;
        logMessage(LogLevel::Error, "loadTexture: GPU upload failed for texId=%u", texId);
        return false;
    }

    // Publish results under lock
    lock.lock();
    info.width = finalWidth;
    info.height = finalHeight;
    info.channels = finalChannels;
    h_textures_[texId] = (TextureObject) info.texObj;
    uint32_t wordIdx = texId / 32;
    uint32_t bitIdx = texId % 32;
    h_residentFlags_[wordIdx] |= (1u << bitIdx);
    markTextureDirtyLocked(texId);
    markResidentWordDirtyLocked(wordIdx);
    info.resident.store(true, std::memory_order_release);
    info.loading.store(false, std::memory_order_release);
    info.lastUsedFrame = currentFrame_;
    info.loadedFrame = currentFrame_;  // Track when loaded for thrashing prevention
    totalMemoryUsage_ += info.memoryUsage;
    logMessage(LogLevel::Info, "loadTexture: id=%u size=%dx%d mipLevels=%d mem=%.2f MB total=%.2f MB",
               texId, info.width, info.height, info.numMipLevels,
               static_cast<double>(info.memoryUsage) / (1024.0 * 1024.0),
               static_cast<double>(totalMemoryUsage_) / (1024.0 * 1024.0));

    return true;
}

bool DemandTextureLoader::Impl::generateMipLevels(hipMipmappedArray_t mipmapArray, unsigned char* baseData,
                                                   int baseWidth, int baseHeight, int numLevels) {
    std::vector<unsigned char> currentLevel(baseWidth * baseHeight * 4);
    std::memcpy(currentLevel.data(), baseData, baseWidth * baseHeight * 4);

    int width = baseWidth;
    int height = baseHeight;

    for (int level = 1; level < numLevels; ++level) {
        int prevWidth = width;
        int prevHeight = height;
        width = std::max(1, width / 2);
        height = std::max(1, height / 2);

        std::vector<unsigned char> nextLevel(width * height * 4);

        // Simple box filter downsample
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int sx = x * 2;
                int sy = y * 2;

                for (int c = 0; c < 4; ++c) {
                    int sum = 0;
                    int count = 0;

                    for (int dy = 0; dy < 2 && (sy + dy) < prevHeight; ++dy) {
                        for (int dx = 0; dx < 2 && (sx + dx) < prevWidth; ++dx) {
                            sum += currentLevel[((sy + dy) * prevWidth + (sx + dx)) * 4 + c];
                            count++;
                        }
                    }

                    nextLevel[(y * width + x) * 4 + c] = sum / count;
                }
            }
        }

        hipArray_t levelArray;
        hipError_t err = hipGetMipmappedArrayLevel(&levelArray, mipmapArray, level);
        if (err != hipSuccess) {
            return false;
        }

        err = hipMemcpy2DToArray(levelArray, 0, 0, nextLevel.data(), width * 4,
                                width * 4, height, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            return false;
        }

        currentLevel = std::move(nextLevel);
    }

    return true;
}

// -----------------------------------------------------------------------------
// Texture Unloading & Eviction
// -----------------------------------------------------------------------------

void DemandTextureLoader::Impl::destroyTexture(uint32_t texId) {
    TextureMetadata& info = textures_[texId];

    if (!info.resident.load(std::memory_order_acquire)) return;

    if (info.texObj) {
        hipError_t err = hipDestroyTextureObject(info.texObj);
        if (err != hipSuccess) {
            lastError_ = LoaderError::HipError;
        }
        info.texObj = 0;
    }

    if (info.mipmapArray) {
        hipError_t err = hipFreeMipmappedArray(info.mipmapArray);
        if (err != hipSuccess) {
            lastError_ = LoaderError::HipError;
        }
        info.mipmapArray = nullptr;
    }

    if (info.array) {
        hipError_t err = hipFreeArray(info.array);
        if (err != hipSuccess) {
            lastError_ = LoaderError::HipError;
        }
        info.array = nullptr;
    }

    info.resident.store(false, std::memory_order_release);
    info.hasMipmaps = false;
    info.numMipLevels = 0;

    h_textures_[texId] = 0;
    uint32_t wordIdx = texId / 32;
    uint32_t bitIdx = texId % 32;
    h_residentFlags_[wordIdx] &= ~(1u << bitIdx);

    markTextureDirtyLocked(texId);
    markResidentWordDirtyLocked(wordIdx);

    logMessage(LogLevel::Debug, "destroyTexture: evicted texId=%u freed=%.2f MB",
               texId, static_cast<double>(info.memoryUsage) / (1024.0 * 1024.0));
    totalMemoryUsage_ -= info.memoryUsage;
    info.memoryUsage = 0;
}

void DemandTextureLoader::Impl::evictIfNeeded(size_t requiredMemory) {
    if (options_.maxTextureMemory == 0) {
        return;
    }

    if (totalMemoryUsage_ + requiredMemory <= options_.maxTextureMemory) {
        return;
    }

    logMessage(LogLevel::Debug, "evictIfNeeded: current=%.2f MB required=%.2f MB budget=%.2f MB",
               static_cast<double>(totalMemoryUsage_) / (1024.0 * 1024.0),
               static_cast<double>(requiredMemory) / (1024.0 * 1024.0),
               static_cast<double>(options_.maxTextureMemory) / (1024.0 * 1024.0));

    // Build eviction candidate list with priority and age information
    // Tuple: (priority, lastUsedFrame, textureId)
    // Lower priority value = evicted first, then by oldest last-used frame
    std::vector<std::tuple<int, uint32_t, uint32_t>> evictionList;
    for (uint32_t i = 0; i < nextTextureId_; ++i) {
        const auto& tex = textures_[i];
        if (!tex.resident.load(std::memory_order_relaxed)) {
            continue;
        }
        
        // Skip textures marked as KeepResident
        if (tex.desc.evictionPriority == EvictionPriority::KeepResident) {
            continue;
        }
        
        // Thrashing prevention: don't evict textures that were just loaded
        uint32_t framesResident = currentFrame_ - tex.loadedFrame;
        if (framesResident < options_.minResidentFrames) {
            logMessage(LogLevel::Debug, "evictIfNeeded: skipping texture %u (only %u frames resident)",
                       i, framesResident);
            continue;
        }
        
        // Priority scoring: Low=1, Normal=0, High=2 -> invert so Low evicted first
        // Map: Low(1)->0, Normal(0)->1, High(2)->2 (High never evicted before others)
        int priorityScore;
        switch (tex.desc.evictionPriority) {
            case EvictionPriority::Low:    priorityScore = 0; break;  // Evict first
            case EvictionPriority::Normal: priorityScore = 1; break;  // Evict second
            case EvictionPriority::High:   priorityScore = 2; break;  // Evict last
            default:                       priorityScore = 1; break;
        }
        
        evictionList.push_back({priorityScore, tex.lastUsedFrame, i});
    }

    // Sort by priority first, then by age (oldest first within same priority)
    std::sort(evictionList.begin(), evictionList.end());

    size_t targetMemory = options_.maxTextureMemory - requiredMemory;
    for (const auto& [priority, frame, texId] : evictionList) {
        if (totalMemoryUsage_ <= targetMemory) {
            break;
        }
        logMessage(LogLevel::Debug, "evictIfNeeded: evicting texture %u (priority=%d, lastUsed=%u)",
                   texId, priority, frame);
        destroyTexture(texId);
    }
}

void DemandTextureLoader::Impl::unloadTexture(uint32_t texId) {
    std::lock_guard<std::mutex> lock(mutex_);
    destroyTexture(texId);
}

void DemandTextureLoader::Impl::unloadAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (uint32_t i = 0; i < nextTextureId_; ++i) {
        destroyTexture(i);
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
    return lastError_;
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
        textures_[texId].desc.evictionPriority = priority;
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
    
    // Unload all textures to free GPU resources
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (uint32_t i = 0; i < nextTextureId_; ++i) {
            destroyTexture(i);
        }
    }
    
    logMessage(LogLevel::Info, "abort: completed gracefully");
}

bool DemandTextureLoader::Impl::isAborted() const {
    return aborted_.load(std::memory_order_acquire);
}

} // namespace hip_demand
