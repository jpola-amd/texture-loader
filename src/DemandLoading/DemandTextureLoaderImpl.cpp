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
    if (initializationError_ != LoaderError::Success)
        return registrationFailure(initializationError_);
    if (filename.empty() || filename.find('\0') != std::string::npos || !validDescriptor(desc))
        return registrationFailure(LoaderError::InvalidParameter);

    auto it = filenameToTextureId_.find(filename);
    if (it != filenameToTextureId_.end()) {
        const auto& existing = textures_[it->second];
        return {it->second, true, existing.width, existing.height, existing.channels, LoaderError::Success};
    }
    if (nextTextureId_ >= options_.maxTextures)
        return registrationFailure(LoaderError::MaxTexturesExceeded);

    try {
        TextureMetadata info;
        info.filename = filename;
        info.desc = desc;
        info.uploadBytesPerPixel = stbi_is_hdr(filename.c_str()) || stbi_is_16_bit(filename.c_str()) ? 16 : 4;
#ifdef USE_OIIO
        try {
            auto source = createImageSource(filename);
            if (source) {
                TextureInfo texInfo;
                source->open(&texInfo);
                if (source->isOpen()) {
                    validateRegistrationImage(texInfo);
                    info.width = static_cast<int>(texInfo.width);
                    info.height = static_cast<int>(texInfo.height);
                    info.channels = static_cast<int>(texInfo.numChannels);
                    info.uploadBytesPerPixel = texInfo.format == HIP_AD_FORMAT_UNSIGNED_INT8 ? 4 : 16;
                    info.imageSource = std::move(source);
                }
            }
        } catch (const std::bad_alloc&) {
            throw;
        } catch (const std::exception& e) {
            // Filename registration remains lazy even when metadata probing fails.
            logMessage(LogLevel::Warn, "createTexture: metadata probe for '%s' failed: %s", filename.c_str(), e.what());
        }
#endif
        if (!info.imageSource) {
            int w, h, c;
            if (stbi_info(filename.c_str(), &w, &h, &c)) {
                info.width = w;
                info.height = h;
                info.channels = c;
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
    std::lock_guard<std::mutex> lock(mutex_);
    if (initializationError_ != LoaderError::Success)
        return registrationFailure(initializationError_);
    if (!imageSource || !validDescriptor(desc))
        return registrationFailure(LoaderError::InvalidParameter);

    // First check: same ImageSource pointer already registered
    ImageSource* rawPtr = imageSource.get();
    auto ptrIt = imageSourceToTextureId_.find(rawPtr);
    if (ptrIt != imageSourceToTextureId_.end()) {
        uint32_t existingId = ptrIt->second;
        TextureMetadata& existing = textures_[existingId];
        logMessage(LogLevel::Debug, "createTexture: reusing existing texture id=%u for ImageSource %p", existingId, rawPtr);
        return TextureHandle{existingId, true, existing.width, existing.height, existing.channels, LoaderError::Success};
    }

    unsigned long long contentHash;
    TextureInfo texInfo;
    try {
        contentHash = imageSource->getHash(0);
    } catch (const std::bad_alloc&) {
        return registrationFailure(LoaderError::OutOfMemory);
    } catch (const std::exception& e) {
        logMessage(LogLevel::Error, "createTexture: ImageSource hash failed: %s", e.what());
        return registrationFailure(LoaderError::ImageLoadFailed);
    }
    if (contentHash != 0) {
        auto hashIt = contentHashToTextureId_.find(contentHash);
        if (hashIt != contentHashToTextureId_.end()) {
            uint32_t existingId = hashIt->second;
            TextureMetadata& existing = textures_[existingId];
            // Do not cache an unretained incoming pointer as an identity alias.
            logMessage(LogLevel::Debug, "createTexture: reusing existing texture id=%u via content hash", existingId);
            return TextureHandle{existingId, true, existing.width, existing.height, existing.channels, LoaderError::Success};
        }
    }

    if (nextTextureId_ >= options_.maxTextures)
        return registrationFailure(LoaderError::MaxTexturesExceeded);
    try {
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
    } catch (const std::invalid_argument&) {
        return registrationFailure(LoaderError::InvalidParameter);
    } catch (const std::overflow_error&) {
        return registrationFailure(LoaderError::InvalidParameter);
    }
    TextureMetadata info;
    info.imageSource = std::move(imageSource);
    info.desc = desc;
    info.width = static_cast<int>(texInfo.width);
    info.height = static_cast<int>(texInfo.height);
    info.channels = static_cast<int>(texInfo.numChannels);
    info.uploadBytesPerPixel = texInfo.format == HIP_AD_FORMAT_UNSIGNED_INT8 ? 4 : 16;
    return commitRegistration(std::move(info), contentHash);
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
    TextureMetadata info;
    info.desc = desc;
    info.width = width;
    info.height = height;
    info.channels = channels;
    info.resident.store(false, std::memory_order_relaxed);
    info.loading.store(false, std::memory_order_relaxed);

    try {
        hipCalls_.registrationCheckpoint(HipOperation::CacheData);
        info.cachedData = std::make_unique<uint8_t[]>(dataSize);
        std::memcpy(info.cachedData.get(), data, dataSize);
    } catch (const std::bad_alloc&) {
        return registrationFailure(LoaderError::OutOfMemory);
    }
    return commitRegistration(std::move(info));
}

TextureHandle DemandTextureLoader::Impl::registrationFailure(LoaderError error) {
    lastError_ = error;
    logMessage(LogLevel::Error, "Texture registration failed: %s", getErrorString(error));
    return {InvalidTextureId, false, 0, 0, 0, error};
}

TextureHandle DemandTextureLoader::Impl::commitRegistration(TextureMetadata&& info,
                                                            unsigned long long contentHash) {
    const uint32_t id = nextTextureId_;
    struct Rollback {
        Impl& self;
        const TextureMetadata& info;
        unsigned long long hash;
        bool filename = false, source = false, content = false, committed = false;
        ~Rollback() {
            if (committed) return;
            if (filename) self.filenameToTextureId_.erase(info.filename);
            if (source) self.imageSourceToTextureId_.erase(info.imageSource.get());
            if (content) self.contentHashToTextureId_.erase(hash);
        }
    } rollback{*this, info, contentHash};
    try {
        if (!info.filename.empty()) {
            hipCalls_.registrationCheckpoint(HipOperation::FilenameMap);
            rollback.filename = filenameToTextureId_.emplace(info.filename, id).second;
        }
        if (info.imageSource) {
            hipCalls_.registrationCheckpoint(HipOperation::SourceMap);
            rollback.source = imageSourceToTextureId_.emplace(info.imageSource.get(), id).second;
        }
        if (contentHash) {
            hipCalls_.registrationCheckpoint(HipOperation::ContentMap);
            rollback.content = contentHashToTextureId_.emplace(contentHash, id).second;
        }
    } catch (const std::bad_alloc&) {
        return registrationFailure(LoaderError::OutOfMemory);
    }
    rollback.committed = true;
    textures_[id] = std::move(info);
    ++nextTextureId_;
    lastError_ = LoaderError::Success;
    const auto& stored = textures_[id];
    return {id, true, stored.width, stored.height, stored.channels, LoaderError::Success};
}

// -----------------------------------------------------------------------------
// Launch Prepare
// -----------------------------------------------------------------------------

void DemandTextureLoader::Impl::launchPrepare(hipStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initializationError_ != LoaderError::Success) {
        lastError_ = initializationError_;
        logMessage(LogLevel::Error, "launchPrepare: loader initialization failed");
        return;
    }

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
    return initializationError_ == LoaderError::Success ? deviceContext_ : DeviceContext{};
}

// -----------------------------------------------------------------------------
// Request Processing
// -----------------------------------------------------------------------------

size_t DemandTextureLoader::Impl::processRequests(hipStream_t stream, const DeviceContext& deviceContext) {
    if (initializationError_ != LoaderError::Success) {
        lastError_ = initializationError_;
        logMessage(LogLevel::Error, "processRequests: loader initialization failed");
        return 0;
    }
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
    if (initializationError_ != LoaderError::Success) {
        lastError_ = initializationError_;
        logMessage(LogLevel::Error, "processRequestsAsync: loader initialization failed");
        return Ticket{};
    }
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
    HIP_CHECK(hipEventRecord(depsReady, stream));

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
    HIP_CHECK(hipEventRecord(copyDone, copyStream));

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
        HIP_CHECK(hipEventSynchronize(copyDone));
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
            if (texId >= nextTextureId_) {
                lastError_ = LoaderError::InvalidTextureId;
                logMessage(LogLevel::Error, "processRequests: invalid texture ID %u", texId);
                continue;
            }
            if (texId < nextTextureId_ && !textures_[texId].resident.load(std::memory_order_relaxed)) {
                if (uniqueRequests.insert(texId).second) {
                    toLoad.push_back(texId);
                    // Calculate actual memory needed
                    const TextureMetadata& info = textures_[texId];
                    int w = info.width;
                    int h = info.height;
                    if (w > 0 && h > 0) {
                        const bool floating = info.imageSource && info.imageSource->isOpen() &&
                            info.imageSource->getInfo().format != HIP_AD_FORMAT_UNSIGNED_INT8;
                        int levels = info.desc.generateMipmaps ? calculateMipLevels(w, h) : 1;
                        if (info.desc.maxMipLevel > 0)
                            levels = static_cast<int>(std::min(static_cast<unsigned int>(levels), info.desc.maxMipLevel));
                        try {
                            const size_t mipMemory = internal::mipImageByteSize(w, h,
                                floating ? sizeof(float) : info.uploadBytesPerPixel / 4, levels);
                            if (mipMemory > std::numeric_limits<size_t>::max() - estimatedMemoryNeeded)
                                estimatedMemoryNeeded = std::numeric_limits<size_t>::max();
                            else
                                estimatedMemoryNeeded += mipMemory;
                        } catch (const std::exception&) {
                            // The decoder reports malformed dimensions before reading pixels.
                        }
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
    const hipError_t deviceError = hipSetDevice(device_);
    if (deviceError != hipSuccess) {
        lastError_ = hipLoaderError(deviceError);
        logMessage(LogLevel::Error, "loadTexture: selecting device %d failed: %s", device_, hipGetErrorString(deviceError));
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
    if (info.array || info.mipmapArray || info.texObj) {
        destroyTexture(texId);
        if (info.array || info.mipmapArray || info.texObj) {
            info.loading.store(false, std::memory_order_release);
            return false;
        }
    }
    info.primaryHipError = hipSuccess;
    info.cleanupHipError = hipSuccess;
    TextureDesc desc = info.desc;
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
            } catch (...) {
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
            info.loading.store(false, std::memory_order_release);
            info.lastError = LoaderError::ImageLoadFailed;
            lastError_ = info.lastError;
            logMessage(LogLevel::Error, "loadTexture: failed to read image for texId=%u", texId);
            return false;
        }
    } catch (const std::exception& e) {
        lock.lock();
        info.loading.store(false, std::memory_order_release);
        info.lastError = LoaderError::ImageLoadFailed;
        lastError_ = info.lastError;
        logMessage(LogLevel::Error, "loadTexture: image decode failed: %s", e.what());
        return false;
    } catch (...) {
        lock.lock();
        info.loading.store(false, std::memory_order_release);
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
    bool success = false;
    bool useMipmaps = desc.generateMipmaps && (width > 1 || height > 1);

    if (useMipmaps) {
        // Check if mipmaps are supported on this GPU (one-time check)
        bool shouldUseMipmaps = true;
        {
            std::lock_guard<std::mutex> capLock(mutex_);
            if (!mipmapsSupportChecked_) {
                // Perform capability test with a minimal allocation
                hipMipmappedArray_t testArray = nullptr;
                hipChannelFormatDesc testChannelDesc = hipCreateChannelDesc<uchar4>();
                hipExtent testExtent = make_hipExtent(1, 1, 0);
                hipError_t testErr = hipMallocMipmappedArray(&testArray, &testChannelDesc, testExtent, 1);
                
                if (testErr != hipSuccess) {
                    mipmapsSupported_ = false;
                    std::cerr << "\n*** WARNING: Mipmapped arrays not supported on this GPU ***\n"
                              << "    HIP Error: " << hipGetErrorString(testErr) << "\n"
                              << "    Falling back to non-mipmapped textures for all loads.\n"
                              << "    This may result in aliasing artifacts at distance.\n" << std::endl;
                } else {
                    HIP_CHECK(hipFreeMipmappedArray(testArray)); // Ignore cleanup errors during capability test
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

        hipExtent extent = make_hipExtent(width, height, 0);

        err = hipCalls_.call(HipOperation::AllocateMipmapped, [&] {
            return hipMallocMipmappedArray(&info.mipmapArray, &channelDesc, extent, numLevels);
        });
        if (err != hipSuccess) {
            lock.lock();
            info.loading.store(false, std::memory_order_release);
            info.primaryHipError = err;
            info.lastError = hipLoaderError(err);
            lastError_ = info.lastError;
            return false;
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
            } catch (const std::exception& e) {
                logMessage(LogLevel::Error, "loadTexture: mip generation failed: %s", e.what());
                success = false;
            }
        }

        if (success) {
            hipResourceDesc resDesc = {};
            resDesc.resType = hipResourceTypeMipmappedArray;
            resDesc.res.mipmap.mipmap = info.mipmapArray;

            hipTextureDesc texDesc = {};
            texDesc.addressMode[0] = desc.addressMode[0];
            texDesc.addressMode[1] = desc.addressMode[1];
            texDesc.filterMode = desc.filterMode;
            texDesc.readMode = image.readMode();
            texDesc.normalizedCoords = desc.normalizedCoords ? 1 : 0;
            texDesc.sRGB = desc.sRGB && !image.isFloat() ? 1 : 0;
            texDesc.maxMipmapLevelClamp = numLevels - 1;
            texDesc.minMipmapLevelClamp = 0;
            texDesc.mipmapFilterMode = desc.mipmapFilterMode;

            err = hipCalls_.call(HipOperation::CreateSampler, [&] {
                return hipCreateTextureObject(&info.texObj, &resDesc, &texDesc, nullptr);
            });
            success = (err == hipSuccess);

            if (success) {
                info.hasMipmaps = true;
                info.numMipLevels = numLevels;
                info.memoryUsage = internal::mipImageByteSize(image, numLevels);
            }
        }
    } else {
        err = hipCalls_.call(HipOperation::AllocateArray, [&] {
            return hipMallocArray(&info.array, &channelDesc, width, height);
        });

        if (err == hipSuccess) {
            err = hipCalls_.call(HipOperation::Upload, [&] {
                return hipMemcpy2DToArray(info.array, 0, 0, image.data(), rowBytes,
                                         rowBytes, height, hipMemcpyHostToDevice);
            });
        }

        if (err == hipSuccess) {
            hipResourceDesc resDesc = {};
            resDesc.resType = hipResourceTypeArray;
            resDesc.res.array.array = info.array;

            hipTextureDesc texDesc = {};
            texDesc.addressMode[0] = desc.addressMode[0];
            texDesc.addressMode[1] = desc.addressMode[1];
            texDesc.filterMode = desc.filterMode;
            texDesc.readMode = image.readMode();
            texDesc.normalizedCoords = desc.normalizedCoords ? 1 : 0;
            texDesc.sRGB = desc.sRGB && !image.isFloat() ? 1 : 0;

            err = hipCalls_.call(HipOperation::CreateSampler, [&] {
                return hipCreateTextureObject(&info.texObj, &resDesc, &texDesc, nullptr);
            });
            success = (err == hipSuccess);

            if (success) {
                info.hasMipmaps = false;
                info.numMipLevels = 1;
                info.memoryUsage = image.sizeBytes();
            }
        }
    }

    if (!success) {
        info.primaryHipError = err;
        cleanupTextureResources(info);
        lock.lock();
        info.loading.store(false, std::memory_order_release);
        info.lastError = err == hipSuccess ? LoaderError::ImageLoadFailed : hipLoaderError(err);
        lastError_ = info.lastError;
        logMessage(LogLevel::Error, "loadTexture: texId=%u primary HIP error=%d cleanup HIP error=%d",
                   texId, static_cast<int>(info.primaryHipError), static_cast<int>(info.cleanupHipError));
        return false;
    }

    // Publish results under lock
    lock.lock();
    info.lastError = LoaderError::Success;
    info.width = width;
    info.height = height;
    info.uploadBytesPerPixel = static_cast<unsigned int>(image.bytesPerPixel());
    // Keep the source channel count: cachedData still holds that native layout
    // and must be decoded with the same stride after unload or eviction.
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

bool DemandTextureLoader::Impl::cleanupTextureResources(TextureMetadata& info) {
    if (info.texObj) {
        const hipError_t err = hipCalls_.call(HipOperation::DestroySampler, [&] {
            return hipDestroyTextureObject(info.texObj);
        });
        if (err != hipSuccess) {
            info.cleanupHipError = err;
            return false;
        }
        info.texObj = 0;
    }

    if (info.mipmapArray) {
        const hipError_t err = hipCalls_.call(HipOperation::FreeMipmapped, [&] {
            return hipFreeMipmappedArray(info.mipmapArray);
        });
        if (err != hipSuccess) {
            info.cleanupHipError = err;
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
            return false;
        }
        info.array = nullptr;
    }
    return true;
}

void DemandTextureLoader::Impl::destroyTexture(uint32_t texId) {
    TextureMetadata& info = textures_[texId];
    if (!info.resident.load(std::memory_order_acquire) && !info.texObj && !info.array && !info.mipmapArray)
        return;
    info.resident.store(false, std::memory_order_release);
    info.hasMipmaps = false;
    info.numMipLevels = 0;

    h_textures_[texId] = 0;
    uint32_t wordIdx = texId / 32;
    uint32_t bitIdx = texId % 32;
    h_residentFlags_[wordIdx] &= ~(1u << bitIdx);

    markTextureDirtyLocked(texId);
    markResidentWordDirtyLocked(wordIdx);
    if (!cleanupTextureResources(info)) {
        lastError_ = LoaderError::HipError;
        return;
    }

    logMessage(LogLevel::Debug, "destroyTexture: evicted texId=%u freed=%.2f MB",
               texId, static_cast<double>(info.memoryUsage) / (1024.0 * 1024.0));
    totalMemoryUsage_ -= info.memoryUsage;
    info.memoryUsage = 0;
}

void DemandTextureLoader::Impl::evictIfNeeded(size_t requiredMemory) {
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

    size_t targetMemory = requiredMemory < options_.maxTextureMemory ?
                            options_.maxTextureMemory - requiredMemory : 0;
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
    if (texId >= nextTextureId_) {
        lastError_ = LoaderError::InvalidTextureId;
        logMessage(LogLevel::Error, "unloadTexture: invalid texture ID %u", texId);
        return;
    }
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
        textures_[texId].desc.evictionPriority = priority;
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
