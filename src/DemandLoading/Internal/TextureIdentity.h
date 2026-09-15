// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime.h>
#include <DemandLoading/DemandTextureLoader.h>
#include <ImageSource/TextureInfo.h>

#include <functional>

namespace hip_demand {
namespace internal {

inline void hashTextureField(size_t& hash, size_t value) noexcept {
    hash ^= value + static_cast<size_t>(0x9e3779b9U) + (hash << 6) + (hash >> 2);
}

struct TextureDescHash {
    size_t operator()(const TextureDesc& desc) const noexcept {
        size_t hash = 0;
        hashTextureField(hash, static_cast<size_t>(desc.addressMode[0]));
        hashTextureField(hash, static_cast<size_t>(desc.addressMode[1]));
        hashTextureField(hash, static_cast<size_t>(desc.filterMode));
        hashTextureField(hash, static_cast<size_t>(desc.mipmapFilterMode));
        hashTextureField(hash, desc.normalizedCoords);
        hashTextureField(hash, desc.sRGB);
        hashTextureField(hash, desc.generateMipmaps);
        hashTextureField(hash, desc.maxMipLevel);
        hashTextureField(hash, static_cast<size_t>(desc.evictionPriority));
        return hash;
    }
};

enum class StorageIdentity { Filename, Source, Content };

struct StorageKey {
    StorageIdentity identity = StorageIdentity::Filename;
    std::string filename;
    const ImageSource* source = nullptr;
    unsigned long long contentHash = 0;
    TextureInfo metadata{};
    bool sRGB = false;
    bool generateMipmaps = true;
    unsigned int maxMipLevel = 0;

    bool operator==(const StorageKey& other) const noexcept {
        return identity == other.identity && filename == other.filename &&
               source == other.source && contentHash == other.contentHash &&
               metadata == other.metadata && sRGB == other.sRGB &&
               generateMipmaps == other.generateMipmaps && maxMipLevel == other.maxMipLevel;
    }
};

struct StorageKeyHash {
    size_t operator()(const StorageKey& key) const noexcept {
        size_t hash = static_cast<size_t>(key.identity);
        hashTextureField(hash, std::hash<std::string>{}(key.filename));
        hashTextureField(hash, std::hash<const ImageSource*>{}(key.source));
        hashTextureField(hash, std::hash<unsigned long long>{}(key.contentHash));
        hashTextureField(hash, key.metadata.width);
        hashTextureField(hash, key.metadata.height);
        hashTextureField(hash, static_cast<size_t>(key.metadata.format));
        hashTextureField(hash, key.metadata.numChannels);
        hashTextureField(hash, key.metadata.numMipLevels);
        hashTextureField(hash, key.metadata.isValid);
        hashTextureField(hash, key.metadata.isTiled);
        hashTextureField(hash, key.sRGB);
        hashTextureField(hash, key.generateMipmaps);
        hashTextureField(hash, key.maxMipLevel);
        return hash;
    }
};

} // namespace internal
} // namespace hip_demand
