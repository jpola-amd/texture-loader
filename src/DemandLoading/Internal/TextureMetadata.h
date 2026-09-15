// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime.h>
#include <DemandLoading/DemandTextureLoader.h>
#include <ImageSource/ImageSource.h>
#include <ImageSource/TextureInfo.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

static_assert(sizeof(hip_demand::TextureObject) == sizeof(hipTextureObject_t),
              "TextureObject must be binary-compatible with hipTextureObject_t");

namespace hip_demand {
namespace internal {

// The registry and active operations retain this owner. HIP destruction is
// explicit: failed cleanup must retain both the handle and its payload charge.
struct ImageStorage {
    std::string filename;
    std::shared_ptr<ImageSource> imageSource;
    TextureInfo sourceInfo{};
    unsigned long long contentHash = 0;
    std::unique_ptr<uint8_t[]> cachedData;
    std::vector<uint32_t> samplers;

    hipArray_t array = nullptr;
    hipMipmappedArray_t mipmapArray = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned int uploadBytesPerPixel = 4;
    bool uploaded = false;
    bool hasMipmaps = false;
    bool floatPixels = false;
    int numMipLevels = 0;
    size_t memoryUsage = 0;
    uint32_t lastUsedFrame = 0;
    uint32_t loadedFrame = 0;
    LoaderError lastError = LoaderError::Success;
    hipError_t primaryHipError = hipSuccess;
    hipError_t cleanupHipError = hipSuccess;
    capability_v1::Failure primary{}, cleanup{}, fallback{};
    capability_v1::Reason reason = capability_v1::Reason::None;
    unsigned int originalLevels = 0;
};

struct TextureMetadata {
    std::shared_ptr<ImageStorage> storage;
    TextureDesc desc{};
    size_t descriptorHash = 0;
    hipTextureObject_t texObj = 0;
    std::atomic<bool> resident{false};
    uint64_t cancellationEpoch = 0;
    LoaderError lastError = LoaderError::Success;
    hipError_t primaryHipError = hipSuccess;
    hipError_t cleanupHipError = hipSuccess;
    capability_v1::MipPolicy policy = capability_v1::MipPolicy::LegacyCompatibility;
    anisotropy_v1::Request anisotropy = anisotropy_v1::Request::legacy();
    capability_v1::Status status{};

    TextureMetadata() = default;
    TextureMetadata(TextureMetadata&& other) noexcept { *this = std::move(other); }
    TextureMetadata& operator=(TextureMetadata&& other) noexcept {
        if (this != &other) {
            storage = std::move(other.storage);
            desc = other.desc;
            descriptorHash = other.descriptorHash;
            texObj = other.texObj;
            resident.store(other.resident.load(std::memory_order_relaxed), std::memory_order_relaxed);
            cancellationEpoch = other.cancellationEpoch;
            lastError = other.lastError;
            primaryHipError = other.primaryHipError;
            cleanupHipError = other.cleanupHipError;
            policy = other.policy;
            anisotropy = other.anisotropy;
            status = other.status;
        }
        return *this;
    }
    TextureMetadata(const TextureMetadata&) = delete;
    TextureMetadata& operator=(const TextureMetadata&) = delete;
};

struct RequestStats {
    uint32_t count;
    uint32_t overflow;
};

} // namespace internal
} // namespace hip_demand
