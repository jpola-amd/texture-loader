#include "DemandLoading/DemandTextureLoader.h"
#include "DemandLoading/Logging.h"
#include "DemandLoading/Ticket.h"
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <thread>
#include <atomic>
#include <cstring>
#include <cmath>
#include <limits>

#ifdef USE_OIIO
#include "ImageSource/ImageSource.h"
#include "ImageSource/OIIOReader.h"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace hip_demand {

const char* getErrorString(LoaderError error) {
    switch (error) {
        case LoaderError::Success: return "Success";
        case LoaderError::InvalidTextureId: return "Invalid texture ID";
        case LoaderError::MaxTexturesExceeded: return "Maximum textures exceeded";
        case LoaderError::FileNotFound: return "File not found";
        case LoaderError::ImageLoadFailed: return "Image load failed";
        case LoaderError::OutOfMemory: return "Out of memory";
        case LoaderError::InvalidParameter: return "Invalid parameter";
        case LoaderError::HipError: return "HIP error";
        default: return "Unknown error";
    }
}

// Internal texture metadata (renamed to avoid conflict with ImageSource::TextureInfo)
struct TextureMetadata {
    std::string filename;
    TextureDesc desc;
    hipTextureObject_t texObj = 0;
    hipArray_t array = nullptr;
    hipMipmappedArray_t mipmapArray = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    int numMipLevels = 0;
    size_t memoryUsage = 0;
    uint32_t lastUsedFrame = 0;
    bool resident = false;
    bool loading = false;
    bool hasMipmaps = false;
    std::unique_ptr<uint8_t[]> cachedData;  // For reload after eviction
    LoaderError lastError = LoaderError::Success;
};

struct RequestStats {
    uint32_t count = 0;
    uint32_t overflow = 0;
};

class DemandTextureLoader::Impl {
public:
    explicit Impl(const LoaderOptions& opts) : options_(opts) {
        hipError_t err = hipGetDevice(&device_);
        if (err != hipSuccess) {
            lastError_ = LoaderError::HipError;
            return;
        }

        // Dedicated stream for request-buffer readback. This enables overlap with the render stream
        // when the app queues further work after calling processRequestsAsync().
        hipStreamCreateWithFlags(&requestCopyStream_, hipStreamNonBlocking);
        
        // Allocate device buffers
        size_t flagWords = (options_.maxTextures + 31) / 32;
        err = hipMalloc(&deviceContext_.residentFlags, flagWords * sizeof(uint32_t));
        if (err != hipSuccess) {
            lastError_ = LoaderError::OutOfMemory;
            return;
        }
        
        err = hipMalloc(&deviceContext_.textures, options_.maxTextures * sizeof(hipTextureObject_t));
        if (err != hipSuccess) {
            lastError_ = LoaderError::OutOfMemory;
            hipFree(deviceContext_.residentFlags);
            deviceContext_.residentFlags = nullptr;
            return;
        }
        
        err = hipMalloc(&deviceContext_.requests, options_.maxRequestsPerLaunch * sizeof(uint32_t));
        if (err != hipSuccess) {
            lastError_ = LoaderError::OutOfMemory;
            hipFree(deviceContext_.textures);
            hipFree(deviceContext_.residentFlags);
            deviceContext_.textures = nullptr;
            deviceContext_.residentFlags = nullptr;
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
        
        err = hipMemset(deviceContext_.textures, 0, options_.maxTextures * sizeof(hipTextureObject_t));
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
        if (hipHostMalloc(reinterpret_cast<void**>(&h_textures_), options_.maxTextures * sizeof(hipTextureObject_t)) != hipSuccess) {
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
        std::fill_n(h_textures_, options_.maxTextures, static_cast<hipTextureObject_t>(0));
        std::fill_n(h_requests_, options_.maxRequestsPerLaunch, 0u);
        h_requestStats_->count = 0;
        h_requestStats_->overflow = 0;

        // First launchPrepare must upload the entire state.
        markAllDirty();

        textures_.resize(options_.maxTextures);
    }
    
    ~Impl() {
        // Ensure any async request-processing tasks complete before we start tearing down
        // resources they might touch (e.g., mutex_, textures_, options_, logging).
        destroying_.store(true, std::memory_order_release);
        {
            std::unique_lock<std::mutex> lock(asyncMutex_);
            asyncCv_.wait(lock, [&] { return inFlightAsync_.load(std::memory_order_acquire) == 0; });
        }

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
    
    TextureHandle createTexture(const std::string& filename, const TextureDesc& desc) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (nextTextureId_ >= options_.maxTextures) {
            lastError_ = LoaderError::MaxTexturesExceeded;
            logMessage(LogLevel::Error, "createTexture: max textures exceeded (%zu)", static_cast<size_t>(options_.maxTextures));
            return TextureHandle{0, false, 0, 0, 0, lastError_};
        }
        
        uint32_t id = nextTextureId_++;
        
        TextureMetadata& info = textures_[id];
        info.filename = filename;
        info.desc = desc;
        info.resident = false;
        
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
    
    TextureHandle createTextureFromMemory(const void* data, int width, int height, 
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
        info.resident = false;
        
        // Cache the data
        size_t dataSize = width * height * channels;
        info.cachedData = std::make_unique<uint8_t[]>(dataSize);
        std::memcpy(info.cachedData.get(), data, dataSize);
        
        lastError_ = LoaderError::Success;
        logMessage(LogLevel::Debug, "createTextureFromMemory: created id=%u (%dx%d ch=%d)", id, width, height, channels);
        return TextureHandle{id, true, width, height, channels, LoaderError::Success};
    }
    
    void launchPrepare(hipStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Upload only dirty ranges for resident flags and texture objects.
        // Pinned host memory makes these memcpyAsync operations cheap to enqueue.
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
                       static_cast<double>(textureCount * sizeof(hipTextureObject_t)) / 1024.0);
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
                                     count * sizeof(hipTextureObject_t),
                                     hipMemcpyHostToDevice,
                                     stream);
                if (err != hipSuccess) {
                    lastError_ = LoaderError::HipError;
                    logMessage(LogLevel::Error, "launchPrepare: hipMemcpyAsync(textures dirty) failed: %s", hipGetErrorString(err));
                    return;
                }
            }
        }

        // Mark the current state as uploaded (further changes will re-dirty under the same mutex).
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
    
    DeviceContext getDeviceContext() const {
        return deviceContext_;
    }
    
    size_t processRequests(hipStream_t stream, const DeviceContext& deviceContext) {
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

    Ticket processRequestsAsync(hipStream_t stream, const DeviceContext& deviceContext) {
        if (destroying_.load(std::memory_order_acquire)) {
            return Ticket{};
        }

        // Allocate per-ticket pinned buffers to avoid races if multiple tickets are in-flight.
        auto makePinned = [](size_t bytes) -> std::shared_ptr<void> {
            void* ptr = nullptr;
            if (hipHostMalloc(&ptr, bytes) != hipSuccess) {
                return {};
            }
            return std::shared_ptr<void>(ptr, [](void* p) { hipHostFree(p); });
        };

        auto statsPinnedBase = makePinned(sizeof(RequestStats));
        auto requestsPinnedBase = makePinned(options_.maxRequestsPerLaunch * sizeof(uint32_t));
        if (!statsPinnedBase || !requestsPinnedBase) {
            lastError_ = LoaderError::OutOfMemory;
            return Ticket{};
        }
        auto* statsPinned = static_cast<RequestStats*>(statsPinnedBase.get());
        auto* requestsPinned = static_cast<uint32_t*>(requestsPinnedBase.get());

        statsPinned->count = 0;
        statsPinned->overflow = 0;

        const uint32_t copyCount = std::min<uint32_t>(static_cast<uint32_t>(options_.maxRequestsPerLaunch), deviceContext.maxRequests);

        // Capture a dependency point on the render stream so the copy stream doesn't race the kernel.
        hipEvent_t depsReady{};
        if (hipEventCreateWithFlags(&depsReady, hipEventDisableTiming) != hipSuccess) {
            lastError_ = LoaderError::HipError;
            return Ticket{};
        }
        hipEventRecord(depsReady, stream);

        hipStream_t copyStream = requestCopyStream_ ? requestCopyStream_ : stream;
        if (copyStream != stream) {
            hipError_t waitErr = hipStreamWaitEvent(copyStream, depsReady, 0);
            if (waitErr != hipSuccess) {
                hipEventDestroy(depsReady);
                lastError_ = LoaderError::HipError;
                return Ticket{};
            }
        }

        hipError_t err = hipMemcpyAsync(&statsPinned->count, deviceContext.requestCount, sizeof(uint32_t),
                                        hipMemcpyDeviceToHost, copyStream);
        if (err != hipSuccess) {
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
            lastError_ = LoaderError::HipError;
            return Ticket{};
        }

        hipEvent_t copyDone{};
        if (hipEventCreateWithFlags(&copyDone, hipEventDisableTiming) != hipSuccess) {
            lastError_ = LoaderError::HipError;
            return Ticket{};
        }
        hipEventRecord(copyDone, copyStream);

        inFlightAsync_.fetch_add(1, std::memory_order_acq_rel);

        auto task = [this, deviceContext, depsReady, copyDone, copyCount, statsPinnedBase, requestsPinnedBase]() {
            struct InFlightGuard {
                DemandTextureLoader::Impl* self;
                ~InFlightGuard() {
                    self->inFlightAsync_.fetch_sub(1, std::memory_order_acq_rel);
                    std::lock_guard<std::mutex> lock(self->asyncMutex_);
                    self->asyncCv_.notify_all();
                }
            } guard{this};

            hipEventSynchronize(copyDone);
            hipEventDestroy(copyDone);
            hipEventDestroy(depsReady);

            auto* statsPinned = static_cast<RequestStats*>(statsPinnedBase.get());
            auto* requestsPinned = static_cast<uint32_t*>(requestsPinnedBase.get());

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

        auto impl = createTicketImpl(std::move(task), stream);
        return Ticket(std::move(impl));
    }
    
    size_t getResidentTextureCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t count = 0;
        for (uint32_t i = 0; i < nextTextureId_; ++i) {
            if (textures_[i].resident) count++;
        }
        return count;
    }
    
    size_t getTotalTextureMemory() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return totalMemoryUsage_;
    }
    
    size_t getRequestCount() const {
        return lastRequestCount_.load(std::memory_order_acquire);
    }
    
    bool hadRequestOverflow() const {
        return lastRequestOverflow_.load(std::memory_order_acquire);
    }
    
    LoaderError getLastError() const {
        return lastError_;
    }
    
    void enableEviction(bool enable) {
        std::lock_guard<std::mutex> lock(mutex_);
        options_.enableEviction = enable;
    }
    
    void setMaxTextureMemory(size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        options_.maxTextureMemory = bytes;
    }
    
    size_t getMaxTextureMemory() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return options_.maxTextureMemory;
    }
    
    void unloadTexture(uint32_t texId) {
        std::lock_guard<std::mutex> lock(mutex_);
        destroyTexture(texId);
    }
    
    void unloadAll() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (uint32_t i = 0; i < nextTextureId_; ++i) {
            destroyTexture(i);
        }
    }
    
private:
    void markAllDirty() {
        // Requires mutex_ held by caller.
        residentFlagsDirty_ = true;
        texturesDirty_ = true;
        dirtyResidentWordBegin_ = 0;
        dirtyResidentWordEnd_ = flagWordCount_ ? (flagWordCount_ - 1) : 0;
        dirtyTextureBegin_ = 0;
        dirtyTextureEnd_ = options_.maxTextures ? (options_.maxTextures - 1) : 0;
    }

    void clearDirtyLocked() {
        // Requires mutex_ held by caller.
        residentFlagsDirty_ = false;
        texturesDirty_ = false;
        dirtyResidentWordBegin_ = std::numeric_limits<size_t>::max();
        dirtyResidentWordEnd_ = 0;
        dirtyTextureBegin_ = std::numeric_limits<size_t>::max();
        dirtyTextureEnd_ = 0;
    }

    void markTextureDirtyLocked(uint32_t texId) {
        // Requires mutex_ held by caller.
        texturesDirty_ = true;
        dirtyTextureBegin_ = std::min(dirtyTextureBegin_, static_cast<size_t>(texId));
        dirtyTextureEnd_ = std::max(dirtyTextureEnd_, static_cast<size_t>(texId));
    }

    void markResidentWordDirtyLocked(uint32_t wordIdx) {
        // Requires mutex_ held by caller.
        residentFlagsDirty_ = true;
        dirtyResidentWordBegin_ = std::min(dirtyResidentWordBegin_, static_cast<size_t>(wordIdx));
        dirtyResidentWordEnd_ = std::max(dirtyResidentWordEnd_, static_cast<size_t>(wordIdx));
    }

    // Calculate total memory needed for mipmaps
    size_t calculateMipmapMemory(int width, int height, int bytesPerPixel) const {
        size_t total = 0;
        while (width > 0 && height > 0) {
            total += width * height * bytesPerPixel;
            width /= 2;
            height /= 2;
        }
        return total;
    }
    
    // Calculate number of mip levels
    int calculateMipLevels(int width, int height) const {
        int levels = 1;
        while (width > 1 || height > 1) {
            width = std::max(1, width / 2);
            height = std::max(1, height / 2);
            levels++;
        }
        return levels;
    }
    
    // Thread-safe texture loading wrapper
    bool loadTextureThreadSafe(uint32_t texId) {
        return loadTexture(texId);
    }

    size_t processRequestsHost(uint32_t requestCount, const uint32_t* requests) {
        // Deduplicate requests and gather texture info under lock
        std::unordered_set<uint32_t> uniqueRequests;
        std::vector<uint32_t> toLoad;
        size_t estimatedMemoryNeeded = 0;

        {
            std::lock_guard<std::mutex> lock(mutex_);

            for (size_t i = 0; i < requestCount; ++i) {
                uint32_t texId = requests[i];
                if (texId < nextTextureId_ && !textures_[texId].resident) {
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

            // Check if we need eviction (with actual size estimates). A maxTextureMemory of 0 means
            // "no budget", so skip eviction entirely in that case.
            if (options_.enableEviction && options_.maxTextureMemory > 0 && estimatedMemoryNeeded > 0) {
                evictIfNeeded(estimatedMemoryNeeded);
            }
        }

        // Load textures outside the lock to allow concurrency
        size_t loaded = 0;
        for (uint32_t texId : toLoad) {
            if (loadTextureThreadSafe(texId)) {
                loaded++;
            }
        }

        return loaded;
    }
    
    // Generate mipmap levels using simple box filter
    bool generateMipLevels(hipMipmappedArray_t mipmapArray, unsigned char* baseData, 
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
            
            // Upload to GPU
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
    bool loadTexture(uint32_t texId) {
        std::unique_lock<std::mutex> lock(mutex_);
        TextureMetadata& info = textures_[texId];
        if (info.resident || info.loading) {
            return false;
        }
        info.loading = true;
        TextureDesc desc = info.desc;
        std::string filename = info.filename;
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
        
        if (!filename.empty()) {
#ifdef USE_OIIO
            // Try OIIO first for better format support
            bool oiioSuccess = false;
            try {
                std::unique_ptr<ImageSource> imgSrc = createImageSource(filename);
                if (imgSrc) {
                    hip_demand::TextureInfo texInfo;
                    imgSrc->open(&texInfo);
                    if (imgSrc->isOpen()) {
                        width = texInfo.width;
                        height = texInfo.height;
                        channels = 4;  // OIIO always provides RGBA
                        
                        // Allocate memory for base level
                        size_t imageSize = width * height * 4;
                        data = new unsigned char[imageSize];
                        
                        // Read base mip level
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
                // Fall through to stb_image
                if (data) {
                    delete[] data;
                    data = nullptr;
                }
                oiioSuccess = false;
            }
            
            // Fall back to stb_image if OIIO failed
            if (!oiioSuccess) {
#endif
                // Force 4 channels for consistency
                data = stbi_load(filename.c_str(), &width, &height, &channels, 4);
                if (!data) {
                    lock.lock();
                    info.loading = false;
                    info.lastError = LoaderError::ImageLoadFailed;
                    logMessage(LogLevel::Error, "loadTexture: failed to load image '%s'", filename.c_str());
                    return false;
                }
                needsFree = true;
                channels = 4;  // stbi_load forces 4 channels
#ifdef USE_OIIO
            }
#endif
        } else if (hasCached) {
            // Use cached data - convert to 4 channels if needed
            width = initWidth;
            height = initHeight;
            channels = initChannels;
            
            if (channels == 4) {
                data = const_cast<unsigned char*>(cachedPtr);
            } else {
                // Convert to 4 channels
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
            info.loading = false;
            info.lastError = LoaderError::InvalidParameter;
            logMessage(LogLevel::Error, "loadTexture: invalid parameters for texId=%u", texId);
            return false;
        }
        
        // Update dimensions if not set
        int finalWidth = width;
        int finalHeight = height;
        int finalChannels = channels;
        
        hipError_t err;
        bool success = false;
        
        // Check if we should generate mipmaps
        bool useMipmaps = desc.generateMipmaps && (width > 1 || height > 1);
        
        if (useMipmaps) {
            // Create mipmapped array
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
                info.loading = false;
                info.lastError = LoaderError::OutOfMemory;
                return false;
            }
            
            // Get level 0 array and copy data
            hipArray_t level0Array;
            err = hipGetMipmappedArrayLevel(&level0Array, info.mipmapArray, 0);
            if (err == hipSuccess) {
                err = hipMemcpy2DToArray(level0Array, 0, 0, data, width * 4,
                                        width * 4, height, hipMemcpyHostToDevice);
            }
            
            if (err == hipSuccess) {
                // Generate remaining mip levels
                success = generateMipLevels(info.mipmapArray, data, width, height, numLevels);
            }
            
            if (success) {
                // Create resource descriptor for mipmapped array
                hipResourceDesc resDesc = {};
                resDesc.resType = hipResourceTypeMipmappedArray;
                resDesc.res.mipmap.mipmap = info.mipmapArray;
                
                // Create texture descriptor
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
            // Create simple non-mipmapped array
            hipChannelFormatDesc channelDesc = hipCreateChannelDesc<uchar4>();
            err = hipMallocArray(&info.array, &channelDesc, width, height);
            
            if (err == hipSuccess) {
                err = hipMemcpy2DToArray(info.array, 0, 0, data, width * 4,
                                        width * 4, height, hipMemcpyHostToDevice);
            }
            
            if (err == hipSuccess) {
                // Create resource descriptor
                hipResourceDesc resDesc = {};
                resDesc.resType = hipResourceTypeArray;
                resDesc.res.array.array = info.array;
                
                // Create texture descriptor
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
        
        // Free stbi data or converted data
        if (needsFree) {
            if (!filename.empty()) {
                stbi_image_free(data);
            } else {
                delete[] data;
            }
        }
        
        if (!success) {
            // Clean up on failure
            if (info.mipmapArray) {
                hipError_t cleanupErr = hipFreeMipmappedArray(info.mipmapArray);
                if (cleanupErr != hipSuccess) {
                    lastError_ = LoaderError::HipError;
                }
                info.mipmapArray = nullptr;
            }
            if (info.array) {
                hipError_t cleanupErr = hipFreeArray(info.array);
                if (cleanupErr != hipSuccess) {
                    lastError_ = LoaderError::HipError;
                }
                info.array = nullptr;
            }
            lock.lock();
            info.loading = false;
            info.lastError = LoaderError::HipError;
            logMessage(LogLevel::Error, "loadTexture: GPU upload failed for texId=%u", texId);
            return false;
        }
        
        // Publish results under lock
        lock.lock();
        info.width = finalWidth;
        info.height = finalHeight;
        info.channels = finalChannels;
        h_textures_[texId] = info.texObj;
        uint32_t wordIdx = texId / 32;
        uint32_t bitIdx = texId % 32;
        h_residentFlags_[wordIdx] |= (1u << bitIdx);
        markTextureDirtyLocked(texId);
        markResidentWordDirtyLocked(wordIdx);
        info.resident = true;
        info.loading = false;
        info.lastUsedFrame = currentFrame_;
        totalMemoryUsage_ += info.memoryUsage;
        logMessage(LogLevel::Info, "loadTexture: id=%u size=%dx%d mipLevels=%d mem=%.2f MB total=%.2f MB", texId, info.width, info.height, info.numMipLevels, static_cast<double>(info.memoryUsage) / (1024.0 * 1024.0), static_cast<double>(totalMemoryUsage_) / (1024.0 * 1024.0));
        
        return true;
    }
    
    void destroyTexture(uint32_t texId) {
        TextureMetadata& info = textures_[texId];
        
        if (!info.resident) return;
        
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
        
        info.resident = false;
        info.hasMipmaps = false;
        info.numMipLevels = 0;
        
        // Update host arrays
        h_textures_[texId] = 0;
        uint32_t wordIdx = texId / 32;
        uint32_t bitIdx = texId % 32;
        h_residentFlags_[wordIdx] &= ~(1u << bitIdx);

        // Mark dirty so the next launchPrepare updates device-side state.
        markTextureDirtyLocked(texId);
        markResidentWordDirtyLocked(wordIdx);
        
        logMessage(LogLevel::Debug, "destroyTexture: evicted texId=%u freed=%.2f MB", texId, static_cast<double>(info.memoryUsage) / (1024.0 * 1024.0));
        totalMemoryUsage_ -= info.memoryUsage;
        info.memoryUsage = 0;
    }
    
    void evictIfNeeded(size_t requiredMemory) {
        // A budget of 0 means unlimited; never evict.
        if (options_.maxTextureMemory == 0) {
            return;
        }

        // Check if we need to free space
        if (totalMemoryUsage_ + requiredMemory <= options_.maxTextureMemory) {
            return;
        }

        logMessage(LogLevel::Debug, "evictIfNeeded: current=%.2f MB required=%.2f MB budget=%.2f MB", static_cast<double>(totalMemoryUsage_) / (1024.0 * 1024.0), static_cast<double>(requiredMemory) / (1024.0 * 1024.0), static_cast<double>(options_.maxTextureMemory) / (1024.0 * 1024.0));
        
        // Find LRU textures to evict
        std::vector<std::pair<uint32_t, uint32_t>> lruList;  // (frame, texId)
        for (uint32_t i = 0; i < nextTextureId_; ++i) {
            if (textures_[i].resident) {
                lruList.push_back({textures_[i].lastUsedFrame, i});
            }
        }
        
        std::sort(lruList.begin(), lruList.end());
        
        // Evict oldest until we have enough space
        size_t targetMemory = options_.maxTextureMemory - requiredMemory;
        for (const auto& [frame, texId] : lruList) {
            if (totalMemoryUsage_ <= targetMemory) {
                break;
            }
            destroyTexture(texId);
        }
    }
    
    LoaderOptions options_;
    int device_;
    std::mutex mutable mutex_;
    
    // Device context with all device pointers
    DeviceContext deviceContext_{};
    RequestStats* d_requestStats_ = nullptr;  // For allocation management

    hipStream_t requestCopyStream_ = nullptr;
    
    // Host pinned buffers
    uint32_t* h_residentFlags_ = nullptr;
    hipTextureObject_t* h_textures_ = nullptr;
    uint32_t* h_requests_ = nullptr;
    RequestStats* h_requestStats_ = nullptr;
    size_t flagWordCount_ = 0;

    // Dirty tracking for deviceContext_ updates (requires mutex_).
    bool residentFlagsDirty_ = false;
    bool texturesDirty_ = false;
    size_t dirtyResidentWordBegin_ = std::numeric_limits<size_t>::max();
    size_t dirtyResidentWordEnd_ = 0;
    size_t dirtyTextureBegin_ = std::numeric_limits<size_t>::max();
    size_t dirtyTextureEnd_ = 0;
    
    // Texture storage
    std::vector<TextureMetadata> textures_;
    uint32_t nextTextureId_ = 0;
    uint32_t currentFrame_ = 0;
    size_t totalMemoryUsage_ = 0;
    
    // Statistics
    std::atomic<size_t> lastRequestCount_{0};
    std::atomic<bool> lastRequestOverflow_{false};

    std::atomic<int> inFlightAsync_{0};
    std::atomic<bool> destroying_{false};
    mutable std::mutex asyncMutex_;
    mutable std::condition_variable asyncCv_;
    LoaderError lastError_ = LoaderError::Success;
};

// Public API implementation
DemandTextureLoader::DemandTextureLoader(const LoaderOptions& options)
    : impl_(std::make_unique<Impl>(options)) {}

DemandTextureLoader::~DemandTextureLoader() = default;

TextureHandle DemandTextureLoader::createTexture(const std::string& filename, 
                                                 const TextureDesc& desc) {
    return impl_->createTexture(filename, desc);
}

TextureHandle DemandTextureLoader::createTextureFromMemory(const void* data, 
                                                           int width, int height, int channels,
                                                           const TextureDesc& desc) {
    return impl_->createTextureFromMemory(data, width, height, channels, desc);
}

void DemandTextureLoader::launchPrepare(hipStream_t stream) {
    impl_->launchPrepare(stream);
}

DeviceContext DemandTextureLoader::getDeviceContext() const {
    return impl_->getDeviceContext();
}

size_t DemandTextureLoader::processRequests(hipStream_t stream, const DeviceContext& deviceContext) {
    return impl_->processRequests(stream, deviceContext);
}

Ticket DemandTextureLoader::processRequestsAsync(hipStream_t stream, const DeviceContext& deviceContext) {
    return impl_->processRequestsAsync(stream, deviceContext);
}

size_t DemandTextureLoader::getResidentTextureCount() const {
    return impl_->getResidentTextureCount();
}

size_t DemandTextureLoader::getTotalTextureMemory() const {
    return impl_->getTotalTextureMemory();
}

size_t DemandTextureLoader::getRequestCount() const {
    return impl_->getRequestCount();
}

bool DemandTextureLoader::hadRequestOverflow() const {
    return impl_->hadRequestOverflow();
}

LoaderError DemandTextureLoader::getLastError() const {
    return impl_->getLastError();
}

void DemandTextureLoader::enableEviction(bool enable) {
    impl_->enableEviction(enable);
}

void DemandTextureLoader::setMaxTextureMemory(size_t bytes) {
    impl_->setMaxTextureMemory(bytes);
}

size_t DemandTextureLoader::getMaxTextureMemory() const {
    return impl_->getMaxTextureMemory();
}

void DemandTextureLoader::unloadTexture(uint32_t textureId) {
    impl_->unloadTexture(textureId);
}

void DemandTextureLoader::unloadAll() {
    impl_->unloadAll();
}

} // namespace hip_demand
