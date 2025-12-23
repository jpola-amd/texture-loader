#pragma once

/// \file TextureInfo.h
/// Image metadata and format information

#include <hip/hip_runtime.h>

namespace hip_demand {

/// Pixel format enumeration
enum class PixelFormat
{
    UINT8,     // 8-bit unsigned integer per channel
    UINT16,    // 16-bit unsigned integer per channel
    FLOAT16,   // 16-bit float per channel
    FLOAT32    // 32-bit float per channel
};

/// Image info, including dimensions and format
struct TextureInfo
{
    unsigned int width = 0;
    unsigned int height = 0;
    PixelFormat format = PixelFormat::UINT8;
    unsigned int numChannels = 0;
    unsigned int numMipLevels = 0;
    bool isValid = false;
    bool isTiled = false;
};

/// Get the channel size in bytes
inline unsigned int getBytesPerChannel(PixelFormat format)
{
    switch (format)
    {
        case PixelFormat::UINT8:   return 1;
        case PixelFormat::UINT16:  return 2;
        case PixelFormat::FLOAT16: return 2;
        case PixelFormat::FLOAT32: return 4;
        default:                   return 0;
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
