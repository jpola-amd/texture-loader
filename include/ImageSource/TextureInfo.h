#pragma once

/// \file TextureInfo.h
/// Image metadata and format information

#include <hip/hip_runtime.h>

namespace hip_demand {

/// Image info, including dimensions and format
struct TextureInfo
{
    unsigned int width = 0;
    unsigned int height = 0;
    hipArray_Format format = HIP_AD_FORMAT_UNSIGNED_INT8;
    unsigned int numChannels = 0;
    unsigned int numMipLevels = 0;
    bool isValid = false;
    bool isTiled = false;
};

/// Get the channel size in bytes
inline unsigned int getBytesPerChannel(hipArray_Format format)
{
    switch (format)
    {
        case HIP_AD_FORMAT_UNSIGNED_INT8:
        case HIP_AD_FORMAT_SIGNED_INT8:
            return 1;
        case HIP_AD_FORMAT_UNSIGNED_INT16:
        case HIP_AD_FORMAT_SIGNED_INT16:
        case HIP_AD_FORMAT_HALF:
            return 2;
        case HIP_AD_FORMAT_UNSIGNED_INT32:
        case HIP_AD_FORMAT_SIGNED_INT32:
        case HIP_AD_FORMAT_FLOAT:
            return 4;
        default:
            return 0;
    }
}

/// Get total texture size in bytes (all mip levels)
inline size_t getTextureSizeInBytes(const TextureInfo& info)
{
    if (!info.isValid) return 0;
    
    size_t total = 0;
    unsigned int w = info.width;
    unsigned int h = info.height;
    unsigned int bytesPerPixel = getBytesPerChannel(info.format) * info.numChannels;
    
    for (unsigned int level = 0; level < info.numMipLevels; ++level)
    {
        total += w * h * bytesPerPixel;
        w = (w > 1) ? w / 2 : 1;
        h = (h > 1) ? h / 2 : 1;
    }
    
    return total;
}

/// Check equality
inline bool operator==(const TextureInfo& a, const TextureInfo& b)
{
    return a.width == b.width &&
           a.height == b.height &&
           a.format == b.format &&
           a.numChannels == b.numChannels &&
           a.numMipLevels == b.numMipLevels &&
           a.isValid == b.isValid &&
           a.isTiled == b.isTiled;
}

}  // namespace hip_demand
