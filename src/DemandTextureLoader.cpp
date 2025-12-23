#include "DemandTextureLoader.h"
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <thread>
#include <atomic>
#include <cstring>
#include <cmath>

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

// Texture metadata
struct TextureInfo {
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

class DemandTextureLoader::Impl {
public:
    explicit Impl(const LoaderOptions& opts) : options_(opts) {
        hipError_t err = hipGetDevice(&device_);
        if (err != hipSuccess) {
            lastError_ = LoaderError::HipError;
            return;
        }
        
        // Allocate device buffers
        size_t flagWords = (options_.maxTextures + 31) / 32;
        err = hipMalloc(&d_residentFlags_, flagWords * sizeof(uint32_t));
        if (err != hipSuccess) {
            lastError_ = LoaderError::OutOfMemory;
            return;
        }
        
        err = hipMalloc(&d_textures_, options_.maxTextures * sizeof(hipTextureObject_t));
        if (err != hipSuccess) {
            lastError_ = LoaderError::OutOfMemory;
            hipFree(d_residentFlags_);
            d_residentFlags_ = nullptr;
            return;
        }
        
        err = hipMalloc(&d_requests_, options_.maxRequestsPerLaunch * sizeof(uint32_t));
        if (err != hipSuccess) {
            lastError_ = LoaderError::OutOfMemory;
            hipFree(d_textures_);
            hipFree(d_residentFlags_);
            d_textures_ = nullptr;
            d_residentFlags_ = nullptr;
            return;
        }
        
        err = hipMalloc(&d_requestCount_, sizeof(uint32_t));
        if (err != hipSuccess) {
            lastError_ = LoaderError::OutOfMemory;
            hipFree(d_requests_);
            hipFree(d_textures_);
            hipFree(d_residentFlags_);
            d_requests_ = nullptr;
            d_textures_ = nullptr;
            d_residentFlags_ = nullptr;
            return;
        }
        
        err = hipMalloc(&d_requestOverflow_, sizeof(uint32_t));
        if (err != hipSuccess) {
            lastError_ = LoaderError::OutOfMemory;
            hipFree(d_requestCount_);
            hipFree(d_requests_);
            hipFree(d_textures_);
            hipFree(d_residentFlags_);
            d_requestCount_ = nullptr;
            d_requests_ = nullptr;
            d_textures_ = nullptr;
            d_residentFlags_ = nullptr;
            return;
        }
        
        // Initialize to zero
        err = hipMemset(d_residentFlags_, 0, flagWords * sizeof(uint32_t));
        if (err != hipSuccess) lastError_ = LoaderError::HipError;
        
        err = hipMemset(d_textures_, 0, options_.maxTextures * sizeof(hipTextureObject_t));
        if (err != hipSuccess) lastError_ = LoaderError::HipError;
        
        err = hipMemset(d_requestCount_, 0, sizeof(uint32_t));
        if (err != hipSuccess) lastError_ = LoaderError::HipError;
        
        err = hipMemset(d_requestOverflow_, 0, sizeof(uint32_t));
        if (err != hipSuccess) lastError_ = LoaderError::HipError;
        
        // Allocate host buffers
        h_residentFlags_.resize(flagWords, 0);
        h_textures_.resize(options_.maxTextures, 0);
        h_requests_.resize(options_.maxRequestsPerLaunch, 0);
        
        textures_.resize(options_.maxTextures);
    }
    
    ~Impl() {
        unloadAll();
        
        if (d_residentFlags_) hipFree(d_residentFlags_);
        if (d_textures_) hipFree(d_textures_);
        if (d_requests_) hipFree(d_requests_);
        if (d_requestCount_) hipFree(d_requestCount_);
        if (d_requestOverflow_) hipFree(d_requestOverflow_);
    }
    
    TextureHandle createTexture(const std::string& filename, const TextureDesc& desc) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (nextTextureId_ >= options_.maxTextures) {
            lastError_ = LoaderError::MaxTexturesExceeded;
            return TextureHandle{0, false, 0, 0, 0, lastError_};
        }
        
        uint32_t id = nextTextureId_++;
        
        TextureInfo& info = textures_[id];
        info.filename = filename;
        info.desc = desc;
        info.resident = false;
        
        // Try to get image dimensions without loading
        int w, h, c;
        if (stbi_info(filename.c_str(), &w, &h, &c)) {
            info.width = w;
            info.height = h;
            info.channels = c;
        } else {
            info.lastError = LoaderError::FileNotFound;
        }
        
        lastError_ = LoaderError::Success;
        return TextureHandle{id, true, info.width, info.height, info.channels, LoaderError::Success};
    }
    
    TextureHandle createTextureFromMemory(const void* data, int width, int height, 
                                         int channels, const TextureDesc& desc) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!data || width <= 0 || height <= 0 || channels <= 0) {
            lastError_ = LoaderError::InvalidParameter;
            return TextureHandle{0, false, 0, 0, 0, lastError_};
        }
        
        if (nextTextureId_ >= options_.maxTextures) {
            lastError_ = LoaderError::MaxTexturesExceeded;
            return TextureHandle{0, false, 0, 0, 0, lastError_};
        }
        
        uint32_t id = nextTextureId_++;
        
        TextureInfo& info = textures_[id];
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
        return TextureHandle{id, true, width, height, channels, LoaderError::Success};
    }
    
    void launchPrepare(hipStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Upload resident flags and texture array
        size_t flagWords = (options_.maxTextures + 31) / 32;
        hipError_t err = hipMemcpyAsync(d_residentFlags_, h_residentFlags_.data(), 
                      flagWords * sizeof(uint32_t), 
                      hipMemcpyHostToDevice, stream);
        if (err != hipSuccess) {
            lastError_ = LoaderError::HipError;
            return;
        }
        
        err = hipMemcpyAsync(d_textures_, h_textures_.data(), 
                      options_.maxTextures * sizeof(hipTextureObject_t),
                      hipMemcpyHostToDevice, stream);
        if (err != hipSuccess) {
            lastError_ = LoaderError::HipError;
            return;
        }
        
        // Reset request counter and overflow flag
        err = hipMemsetAsync(d_requestCount_, 0, sizeof(uint32_t), stream);
        if (err != hipSuccess) {
            lastError_ = LoaderError::HipError;
            return;
        }
        
        err = hipMemsetAsync(d_requestOverflow_, 0, sizeof(uint32_t), stream);
        if (err != hipSuccess) {
            lastError_ = LoaderError::HipError;
            return;
        }
        
        currentFrame_++;
    }
    
    DeviceContext getDeviceContext() const {
        DeviceContext ctx;
        ctx.residentFlags = d_residentFlags_;
        ctx.textures = d_textures_;
        ctx.requests = d_requests_;
        ctx.requestCount = d_requestCount_;
        ctx.requestOverflow = d_requestOverflow_;
        ctx.maxTextures = options_.maxTextures;
        ctx.maxRequests = options_.maxRequestsPerLaunch;
        return ctx;
    }
    
    size_t processRequests(hipStream_t stream) {
        // Download request count and overflow flag
        uint32_t requestCount = 0;
        uint32_t overflow = 0;
        hipError_t err = hipMemcpyAsync(&requestCount, d_requestCount_, sizeof(uint32_t),
                      hipMemcpyDeviceToHost, stream);
        if (err != hipSuccess) {
            lastError_ = LoaderError::HipError;
            return 0;
        }
        
        err = hipMemcpyAsync(&overflow, d_requestOverflow_, sizeof(uint32_t),
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
        
        lastRequestOverflow_ = (overflow != 0);
        lastRequestCount_ = requestCount;
        
        if (requestCount == 0) {
            return 0;
        }
        
        // Download requests
        requestCount = std::min(requestCount, (uint32_t)options_.maxRequestsPerLaunch);
        err = hipMemcpyAsync(h_requests_.data(), d_requests_, 
                      requestCount * sizeof(uint32_t),
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
        
        // Deduplicate requests and gather texture info under lock
        std::unordered_set<uint32_t> uniqueRequests;
        std::vector<uint32_t> toLoad;
        size_t estimatedMemoryNeeded = 0;
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            
            for (size_t i = 0; i < requestCount; ++i) {
                uint32_t texId = h_requests_[i];
                if (texId < nextTextureId_ && !textures_[texId].resident) {
                    if (uniqueRequests.insert(texId).second) {
                        toLoad.push_back(texId);
                        // Calculate actual memory needed
                        const TextureInfo& info = textures_[texId];
                        int w = info.width;
                        int h = info.height;
                        if (w > 0 && h > 0) {
                            size_t mipMemory = calculateMipmapMemory(w, h, 4);
                            estimatedMemoryNeeded += mipMemory;
                        }
                    }
                }
            }
            
            // Check if we need eviction (with actual size estimates)
            if (options_.enableEviction && estimatedMemoryNeeded > 0) {
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
        return lastRequestCount_;
    }
    
    bool hadRequestOverflow() const {
        return lastRequestOverflow_;
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
        std::lock_guard<std::mutex> lock(mutex_);
        return loadTexture(texId);
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
        TextureInfo& info = textures_[texId];
        
        if (info.resident || info.loading) {
            return false;
        }
        
        info.loading = true;
        
        // Load image data
        unsigned char* data = nullptr;
        bool needsFree = false;
        int width, height, channels;
        
        if (!info.filename.empty()) {
            // Force 4 channels for consistency
            data = stbi_load(info.filename.c_str(), &width, &height, &channels, 4);
            if (!data) {
                info.loading = false;
                info.lastError = LoaderError::ImageLoadFailed;
                return false;
            }
            needsFree = true;
            channels = 4;  // stbi_load forces 4 channels
        } else if (info.cachedData) {
            // Use cached data - convert to 4 channels if needed
            width = info.width;
            height = info.height;
            channels = info.channels;
            
            if (channels == 4) {
                data = info.cachedData.get();
            } else {
                // Convert to 4 channels
                size_t pixelCount = width * height;
                unsigned char* data4 = new unsigned char[pixelCount * 4];
                needsFree = true;
                
                for (size_t i = 0; i < pixelCount; ++i) {
                    if (channels == 1) {
                        data4[i*4+0] = info.cachedData[i];
                        data4[i*4+1] = info.cachedData[i];
                        data4[i*4+2] = info.cachedData[i];
                        data4[i*4+3] = 255;
                    } else if (channels == 3) {
                        data4[i*4+0] = info.cachedData[i*3+0];
                        data4[i*4+1] = info.cachedData[i*3+1];
                        data4[i*4+2] = info.cachedData[i*3+2];
                        data4[i*4+3] = 255;
                    }
                }
                data = data4;
                channels = 4;
            }
        } else {
            info.loading = false;
            info.lastError = LoaderError::InvalidParameter;
            return false;
        }
        
        // Update dimensions if not set
        info.width = width;
        info.height = height;
        info.channels = channels;
        
        hipError_t err;
        bool success = false;
        
        // Check if we should generate mipmaps
        bool useMipmaps = info.desc.generateMipmaps && (width > 1 || height > 1);
        
        if (useMipmaps) {
            // Create mipmapped array
            int numLevels = calculateMipLevels(width, height);
            if (info.desc.maxMipLevel > 0) {
                numLevels = std::min(numLevels, (int)info.desc.maxMipLevel);
            }
            
            hipChannelFormatDesc channelDesc = hipCreateChannelDesc<uchar4>();
            hipExtent extent = make_hipExtent(width, height, 0);
            
            err = hipMallocMipmappedArray(&info.mipmapArray, &channelDesc, extent, numLevels);
            if (err != hipSuccess) {
                if (needsFree) {
                    if (!info.filename.empty()) {
                        stbi_image_free(data);
                    } else {
                        delete[] data;
                    }
                }
                info.loading = false;
                info.lastError = LoaderError::HipError;
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
                texDesc.addressMode[0] = info.desc.addressMode;
                texDesc.addressMode[1] = info.desc.addressMode;
                texDesc.filterMode = info.desc.filterMode;
                texDesc.readMode = hipReadModeNormalizedFloat;
                texDesc.normalizedCoords = info.desc.normalizedCoords ? 1 : 0;
                texDesc.sRGB = info.desc.sRGB ? 1 : 0;
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
                texDesc.addressMode[0] = info.desc.addressMode;
                texDesc.addressMode[1] = info.desc.addressMode;
                texDesc.filterMode = info.desc.filterMode;
                texDesc.readMode = hipReadModeNormalizedFloat;
                texDesc.normalizedCoords = info.desc.normalizedCoords ? 1 : 0;
                texDesc.sRGB = info.desc.sRGB ? 1 : 0;
                
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
            if (!info.filename.empty()) {
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
            info.loading = false;
            info.lastError = LoaderError::HipError;
            return false;
        }
        
        // Update host arrays
        h_textures_[texId] = info.texObj;
        uint32_t wordIdx = texId / 32;
        uint32_t bitIdx = texId % 32;
        h_residentFlags_[wordIdx] |= (1u << bitIdx);
        
        info.resident = true;
        info.loading = false;
        info.lastUsedFrame = currentFrame_;
        totalMemoryUsage_ += info.memoryUsage;
        
        return true;
    }
    
    void destroyTexture(uint32_t texId) {
        TextureInfo& info = textures_[texId];
        
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
        
        totalMemoryUsage_ -= info.memoryUsage;
        info.memoryUsage = 0;
    }
    
    void evictIfNeeded(size_t requiredMemory) {
        // Check if we need to free space
        if (totalMemoryUsage_ + requiredMemory <= options_.maxTextureMemory) {
            return;
        }
        
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
    
    // Device pointers
    uint32_t* d_residentFlags_ = nullptr;
    hipTextureObject_t* d_textures_ = nullptr;
    uint32_t* d_requests_ = nullptr;
    uint32_t* d_requestCount_ = nullptr;
    uint32_t* d_requestOverflow_ = nullptr;
    
    // Host buffers
    std::vector<uint32_t> h_residentFlags_;
    std::vector<hipTextureObject_t> h_textures_;
    std::vector<uint32_t> h_requests_;
    
    // Texture storage
    std::vector<TextureInfo> textures_;
    uint32_t nextTextureId_ = 0;
    uint32_t currentFrame_ = 0;
    size_t totalMemoryUsage_ = 0;
    
    // Statistics
    size_t lastRequestCount_ = 0;
    bool lastRequestOverflow_ = false;
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

size_t DemandTextureLoader::processRequests(hipStream_t stream) {
    return impl_->processRequests(stream);
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
