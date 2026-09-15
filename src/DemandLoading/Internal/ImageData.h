// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime.h>
#include <ImageSource/ImageSource.h>
#include <ImageSource/TextureInfo.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hip_demand {
namespace internal {

inline size_t imageByteSize(size_t width, size_t height, size_t channels,
                            size_t bytesPerChannel) {
    if (!width || !height || width > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        channels < 1 || channels > 4 || !bytesPerChannel) {
        throw std::invalid_argument("Invalid image dimensions or channel format");
    }
    size_t size = width;
    for (size_t factor : {height, channels, bytesPerChannel}) {
        if (size > std::numeric_limits<size_t>::max() / factor)
            throw std::overflow_error("Image byte size overflows size_t");
        size *= factor;
    }
    return size;
}

// Texture sampling returns float4. Preserve the byte path (including hardware
// sRGB conversion), and use float RGBA for other source formats so HDR values
// and normalized integer values survive decoding, filtering, and upload.
struct ImageData {
    unsigned int width = 0;
    unsigned int height = 0;
    std::vector<unsigned char> bytes;
    std::vector<float> floats;

    bool isFloat() const { return !floats.empty(); }
    size_t bytesPerPixel() const { return isFloat() ? sizeof(float) * 4 : 4; }
    size_t rowBytes() const { return static_cast<size_t>(width) * bytesPerPixel(); }
    size_t sizeBytes() const { return rowBytes() * height; }
    const void* data() const { return isFloat() ? static_cast<const void*>(floats.data())
                                               : static_cast<const void*>(bytes.data()); }
    hipChannelFormatDesc channelDesc() const {
        return isFloat() ? hipCreateChannelDesc<float4>() : hipCreateChannelDesc<uchar4>();
    }
    hipTextureReadMode readMode() const {
        return isFloat() ? hipReadModeElementType : hipReadModeNormalizedFloat;
    }

    void reset(unsigned int w, unsigned int h, bool floating) {
        const size_t size = imageByteSize(w, h, 4, floating ? sizeof(float) : 1);
        width = w;
        height = h;
        bytes.clear();
        floats.clear();
        if (floating)
            floats.resize(size / sizeof(float));
        else
            bytes.resize(size);
    }
};

template<class T> inline T readChannel(const unsigned char* source) {
    T result;
    std::memcpy(&result, source, sizeof(result));
    return result;
}

inline float halfToFloat(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t fraction = value & 0x3ffu;
    uint32_t result;
    if (exponent == 0) {
        if (!fraction) {
            result = sign;
        } else {
            int shift = 0;
            while (!(fraction & 0x400u)) {
                fraction <<= 1;
                ++shift;
            }
            result = sign | (static_cast<uint32_t>(113 - shift) << 23) |
                     ((fraction & 0x3ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        result = sign | 0x7f800000u | (fraction << 13);
    } else {
        result = sign | ((exponent + 112) << 23) | (fraction << 13);
    }
    float output;
    std::memcpy(&output, &result, sizeof(output));
    return output;
}

template<class T> inline float normalizedChannel(const unsigned char* source) {
    const double value = static_cast<double>(readChannel<T>(source));
    const double maximum = static_cast<double>(std::numeric_limits<T>::max());
    return static_cast<float>(std::max(-1.0, value / maximum));
}

inline float floatChannel(const unsigned char* source, hipArray_Format format) {
    switch (format) {
    case HIP_AD_FORMAT_UNSIGNED_INT8: return normalizedChannel<uint8_t>(source);
    case HIP_AD_FORMAT_SIGNED_INT8: return normalizedChannel<int8_t>(source);
    case HIP_AD_FORMAT_UNSIGNED_INT16: return normalizedChannel<uint16_t>(source);
    case HIP_AD_FORMAT_SIGNED_INT16: return normalizedChannel<int16_t>(source);
    case HIP_AD_FORMAT_UNSIGNED_INT32: return normalizedChannel<uint32_t>(source);
    case HIP_AD_FORMAT_SIGNED_INT32: return normalizedChannel<int32_t>(source);
    case HIP_AD_FORMAT_HALF: return halfToFloat(readChannel<uint16_t>(source));
    case HIP_AD_FORMAT_FLOAT: return readChannel<float>(source);
    default: throw std::invalid_argument("Unsupported ImageSource channel format");
    }
}

inline ImageData decodeImagePixels(const void* pixels, const TextureInfo& info) {
    const size_t channelBytes = getBytesPerChannel(info.format);
    imageByteSize(info.width, info.height, info.numChannels, channelBytes);
    if (!info.isValid || !pixels)
        throw std::invalid_argument("Invalid ImageSource pixels");
    ImageData image;
    image.reset(info.width, info.height, info.format != HIP_AD_FORMAT_UNSIGNED_INT8);
    if (info.numChannels == 4 && info.format == HIP_AD_FORMAT_UNSIGNED_INT8) {
        std::memcpy(image.bytes.data(), pixels, image.sizeBytes());
        return image;
    }
    if (info.numChannels == 4 && info.format == HIP_AD_FORMAT_FLOAT) {
        std::memcpy(image.floats.data(), pixels, image.sizeBytes());
        return image;
    }
    const auto* input = static_cast<const unsigned char*>(pixels);
    const size_t pixelCount = static_cast<size_t>(info.width) * info.height;
    for (size_t i = 0; i < pixelCount; ++i) {
        if (image.isFloat()) {
            float* output = image.floats.data() + 4 * i;
            output[0] = output[1] = output[2] = 0.0f;
            output[3] = 1.0f;
            for (unsigned int c = 0; c < info.numChannels; ++c)
                output[c] = floatChannel(input + (i * info.numChannels + c) * channelBytes,
                                          info.format);
        } else {
            unsigned char* output = image.bytes.data() + 4 * i;
            output[0] = output[1] = output[2] = 0;
            output[3] = 255;
            std::memcpy(output, input + i * info.numChannels, info.numChannels);
        }
    }
    return image;
}

inline bool readImageSource(ImageSource& source, ImageData& image, unsigned int mipLevel = 0) {
    TextureInfo info;
    if (!source.isOpen())
        source.open(&info);
    else
        info = source.getInfo();
    if (!source.isOpen() || !info.isValid)
        return false;
    if (mipLevel > 0) {
        if (mipLevel >= info.numMipLevels || mipLevel >= 32)
            return false;
        info.width = std::max(1u, info.width >> mipLevel);
        info.height = std::max(1u, info.height >> mipLevel);
    }
    const size_t size = imageByteSize(info.width, info.height, info.numChannels,
                                       getBytesPerChannel(info.format));
    if (info.numChannels == 4 && (info.format == HIP_AD_FORMAT_UNSIGNED_INT8 ||
                                   info.format == HIP_AD_FORMAT_FLOAT)) {
        image.reset(info.width, info.height, info.format == HIP_AD_FORMAT_FLOAT);
        char* destination = image.isFloat() ? reinterpret_cast<char*>(image.floats.data())
                                             : reinterpret_cast<char*>(image.bytes.data());
        return source.readMipLevel(destination, mipLevel, info.width, info.height);
    }
    // This buffer must match the source's native channel width, even when the
    // upload representation later changes. Arnold can write FLOAT RGBA here.
    std::vector<unsigned char> native(size);
    if (!source.readMipLevel(reinterpret_cast<char*>(native.data()), mipLevel,
                              info.width, info.height))
        return false;
    image = decodeImagePixels(native.data(), info);
    return true;
}

inline float srgbToLinear(float value) {
    return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

inline float linearToSRGB(float value) {
    return value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.f / 2.4f) - 0.055f;
}

inline void linearizeFloatSRGB(ImageData& image) {
    if (!image.isFloat())
        return;
    for (size_t i = 0; i < image.floats.size(); i += 4) {
        for (unsigned int c = 0; c < 3; ++c) {
            float& value = image.floats[i + c];
            value = srgbToLinear(value);
        }
    }
}

inline ImageData downsampleImage(const ImageData& previous, bool srgb = false) {
    const bool roundBytes = !previous.isFloat() && !srgb;
    const uint64_t weight = static_cast<uint64_t>(previous.width) * previous.height;
    if (roundBytes && weight > std::numeric_limits<uint64_t>::max() / 255)
        throw std::overflow_error("Byte mip weighted sum overflows uint64_t");
    ImageData next;
    next.reset(std::max(1u, previous.width / 2), std::max(1u, previous.height / 2),
                 previous.isFloat());
    if (roundBytes) {
        // Scaled integer overlap lengths describe the same area footprint without
        // moving exact half ties below .5 through floating-point edge arithmetic.
        for (unsigned int y = 0; y < next.height; ++y) {
            const uint64_t top = static_cast<uint64_t>(y) * previous.height;
            const uint64_t bottom = static_cast<uint64_t>(y + 1) * previous.height;
            for (unsigned int x = 0; x < next.width; ++x) {
                const uint64_t left = static_cast<uint64_t>(x) * previous.width;
                const uint64_t right = static_cast<uint64_t>(x + 1) * previous.width;
                uint64_t sums[4]{};
                for (uint64_t sy = top / next.height; sy * next.height < bottom; ++sy) {
                    const uint64_t wy = std::min(bottom, (sy + 1) * next.height) -
                                        std::max(top, sy * next.height);
                    for (uint64_t sx = left / next.width; sx * next.width < right; ++sx) {
                        const uint64_t wx = std::min(right, (sx + 1) * next.width) -
                                            std::max(left, sx * next.width);
                        const size_t index = (sy * previous.width + sx) * 4;
                        for (unsigned int c = 0; c < 4; ++c)
                            sums[c] += previous.bytes[index + c] * wx * wy;
                    }
                }
                const size_t index = (static_cast<size_t>(y) * next.width + x) * 4;
                for (unsigned int c = 0; c < 4; ++c) {
                    const uint64_t rounded = sums[c] / weight +
                        (sums[c] % weight >= weight / 2 + weight % 2);
                    next.bytes[index + c] = static_cast<unsigned char>(std::min(rounded, uint64_t{255}));
                }
            }
        }
        return next;
    }
    for (unsigned int y = 0; y < next.height; ++y) {
        const double top = static_cast<double>(y) * previous.height / next.height;
        const double bottom = static_cast<double>(y + 1) * previous.height / next.height;
        for (unsigned int x = 0; x < next.width; ++x) {
            const double left = static_cast<double>(x) * previous.width / next.width;
            const double right = static_cast<double>(x + 1) * previous.width / next.width;
            for (unsigned int c = 0; c < 4; ++c) {
                long double sum = 0;
                long double totalWeight = 0;
                for (unsigned int sy = static_cast<unsigned int>(top);
                     sy < std::min(previous.height, static_cast<unsigned int>(std::ceil(bottom))); ++sy) {
                    const double yWeight = std::min(bottom, sy + 1.0) - std::max(top, static_cast<double>(sy));
                    for (unsigned int sx = static_cast<unsigned int>(left);
                         sx < std::min(previous.width, static_cast<unsigned int>(std::ceil(right))); ++sx) {
                        const double xWeight = std::min(right, sx + 1.0) - std::max(left, static_cast<double>(sx));
                        const long double weight = static_cast<long double>(xWeight) * yWeight;
                        const size_t index = (static_cast<size_t>(sy) * previous.width + sx) * 4 + c;
                        const bool srgbByte = srgb && !previous.isFloat() && c < 3;
                        const float value = previous.isFloat() ? previous.floats[index] :
                            (srgbByte ? srgbToLinear(previous.bytes[index] / 255.f) : previous.bytes[index]);
                        sum += value * weight;
                        totalWeight += weight;
                    }
                }
                const size_t index = (static_cast<size_t>(y) * next.width + x) * 4 + c;
                if (next.isFloat())
                    next.floats[index] = static_cast<float>(sum / totalWeight);
                else if (srgb && c < 3)
                    next.bytes[index] = static_cast<unsigned char>(std::clamp(
                        std::lround(linearToSRGB(static_cast<float>(sum / totalWeight)) * 255.f), 0l, 255l));
                else
                    // Preserve the existing truncation of alpha in sRGB byte images.
                    next.bytes[index] = static_cast<unsigned char>(sum / totalWeight);
            }
        }
    }
    return next;
}

inline size_t mipImageByteSize(unsigned int width, unsigned int height,
                                size_t bytesPerChannel, int levels) {
    size_t total = 0;
    for (int level = 0; level < levels; ++level) {
        const size_t size = imageByteSize(width, height, 4, bytesPerChannel);
        if (size > std::numeric_limits<size_t>::max() - total)
            throw std::overflow_error("Mip image byte size overflows size_t");
        total += size;
        width = std::max(1u, width / 2);
        height = std::max(1u, height / 2);
    }
    return total;
}

inline size_t mipImageByteSize(const ImageData& image, int levels) {
    return mipImageByteSize(image.width, image.height, image.bytesPerPixel() / 4, levels);
}

} // namespace internal
} // namespace hip_demand
