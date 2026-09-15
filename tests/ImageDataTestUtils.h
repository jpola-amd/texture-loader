// SPDX-License-Identifier: MIT
#pragma once
#include <gtest/gtest.h>
#include "DemandLoading/Internal/ImageData.h"
#include <array>
#include <stdexcept>

namespace hip_demand { namespace test {
class TypedImageSource : public ImageSource {
public:
    TextureInfo info;
    std::vector<unsigned char> pixels;
    std::vector<std::vector<unsigned char>> mipPixels;
    std::vector<unsigned int> readLevels;
    void* destination = nullptr;
    bool opened = false;
    bool failRead = false;
    unsigned int reads = 0;
    unsigned int baseColorReads = 0;

    void open(TextureInfo* result) override { opened = true; if (result) *result = info; }
    void close() override { opened = false; }
    bool isOpen() const override { return opened; }
    const TextureInfo& getInfo() const override { return info; }
    bool readMipLevel(char* dest, unsigned int level, unsigned int width,
                        unsigned int height, hipStream_t) override {
        ++reads;
        readLevels.push_back(level);
        destination = dest;
        EXPECT_EQ(width, std::max(1u, info.width >> level));
        EXPECT_EQ(height, std::max(1u, info.height >> level));
        if (failRead) return false;
        if (mipPixels.empty()) {
            EXPECT_EQ(level, 0u);
            std::memcpy(dest, pixels.data(), pixels.size());
        } else {
            if (level >= mipPixels.size()) return false;
            std::memcpy(dest, mipPixels[level].data(), mipPixels[level].size());
        }
        return true;
    }
    bool readBaseColor(float4&) override { ++baseColorReads; return false; }
    unsigned long long getNumBytesRead() const override { return reads * pixels.size(); }
    double getTotalReadTime() const override { return 0.0; }
};

template<class T> std::vector<unsigned char> packed(std::array<T, 4> values) {
    std::vector<unsigned char> bytes(sizeof(values));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

struct FormatCase {
    hipArray_Format format;
    std::vector<unsigned char> values;
    std::array<float, 4> expected;
};

const std::array<FormatCase, 8> formats = {{
    {HIP_AD_FORMAT_UNSIGNED_INT8, packed<uint8_t>({0, 128, 255, 64}), {0, 128.f/255, 1, 64.f/255}},
    {HIP_AD_FORMAT_SIGNED_INT8, packed<int8_t>({-128, -64, 127, 0}), {-1, -64.f/127, 1, 0}},
    {HIP_AD_FORMAT_UNSIGNED_INT16, packed<uint16_t>({0, 32768, 65535, 1}), {0, 32768.f/65535, 1, 1.f/65535}},
    {HIP_AD_FORMAT_SIGNED_INT16, packed<int16_t>({-32768, -16384, 32767, 0}), {-1, -16384.f/32767, 1, 0}},
    {HIP_AD_FORMAT_UNSIGNED_INT32, packed<uint32_t>({0, 2147483648u, 4294967295u, 1}), {0, .5f, 1, 1.f/4294967295.0f}},
    {HIP_AD_FORMAT_SIGNED_INT32, packed<int32_t>({std::numeric_limits<int32_t>::min(), -1073741824, 2147483647, 0}), {-1, -.5f, 1, 0}},
    {HIP_AD_FORMAT_HALF, packed<uint16_t>({0xc000, 0x3800, 0x4200, 0x0001}), {-2, .5f, 3, 0x1p-24f}},
    {HIP_AD_FORMAT_FLOAT, packed<float>({-.25f, 2.5f, .5f, 1}), {-.25f, 2.5f, .5f, 1}}
}};

inline TypedImageSource makeSource(unsigned int formatIndex, unsigned int channels) {
    TypedImageSource source;
    const auto& format = formats.at(formatIndex);
    source.info.width = 3;
    source.info.height = 2;
    source.info.numChannels = channels;
    source.info.numMipLevels = 1;
    source.info.format = format.format;
    source.info.isValid = true;
    const size_t channelBytes = getBytesPerChannel(format.format);
    source.pixels.resize(3 * 2 * channels * channelBytes);
    for (size_t i = 0; i < 6; ++i)
        for (size_t c = 0; c < channels; ++c)
            std::memcpy(source.pixels.data() + (i * channels + c) * channelBytes,
                           format.values.data() + ((i + c) % 4) * channelBytes, channelBytes);
    return source;
}

inline constexpr std::array<std::array<uint8_t, 4>, 4> authoredByteMipColors{{
    {17, 53, 101, 239}, {211, 29, 149, 127}, {67, 223, 13, 191}, {251, 109, 181, 43}
}};
inline constexpr std::array<std::array<float, 4>, 4> authoredFloatMipColors{{
    {-2, .25f, 3, .5f}, {4, -1, .125f, 1}, {.5f, 2, -3, .25f}, {8, .75f, 1.5f, 0}
}};

inline TypedImageSource makeAuthoredMipSource(bool floating) {
    TypedImageSource source;
    source.info.width = source.info.height = 8;
    source.info.numChannels = 4;
    source.info.numMipLevels = 4;
    source.info.format = floating ? HIP_AD_FORMAT_FLOAT : HIP_AD_FORMAT_UNSIGNED_INT8;
    source.info.isValid = true;
    for (unsigned int level = 0; level < source.info.numMipLevels; ++level) {
        const auto color = floating ? packed(authoredFloatMipColors[level]) : packed(authoredByteMipColors[level]);
        const unsigned int side = 8u >> level;
        auto& pixels = source.mipPixels.emplace_back(side * side * color.size());
        for (size_t pixel = 0; pixel < side * side; ++pixel)
            std::memcpy(pixels.data() + pixel * color.size(), color.data(), color.size());
    }
    source.pixels = source.mipPixels.front();
    return source;
}

inline TypedImageSource makeBoundaryPatternSource(bool floating) {
    TypedImageSource source;
    source.info.width = source.info.height = 4;
    source.info.numChannels = 4;
    source.info.numMipLevels = 1;
    source.info.format = floating ? HIP_AD_FORMAT_FLOAT : HIP_AD_FORMAT_UNSIGNED_INT8;
    source.info.isValid = true;
    for (unsigned int y = 0; y < 4; ++y) {
        for (unsigned int x = 0; x < 4; ++x) {
            const auto color = floating
                ? packed<float>({-2 + .75f*x + .125f*y, .25f - .5f*x + 1.25f*y,
                                 3 - .25f*x - .5f*y, .125f + .125f*x + .0625f*y})
                : packed<uint8_t>({static_cast<uint8_t>(17 + 41*x + 7*y),
                                   static_cast<uint8_t>(13 + 11*x + 47*y),
                                   static_cast<uint8_t>(29 + 23*x + 19*y),
                                   static_cast<uint8_t>(251 - 13*x - 17*y)});
            source.pixels.insert(source.pixels.end(), color.begin(), color.end());
        }
    }
    return source;
}

inline constexpr std::array<std::array<float, 4>, 5> filteringMipColors{{
    {-2, .25f, 3, .625f}, {4, -1, .125f, .625f}, {.5f, 2, -3, .625f},
    {8, .75f, 1.5f, .625f}, {-4, 3, 6, .625f}
}};

inline TypedImageSource makeFilteringMipSource(unsigned int width = 16, unsigned int height = 16,
                                               bool spatialPattern = false) {
    if (width == 0 || height == 0 || width > 31 || height > 31)
        throw std::invalid_argument("Filtering fixture dimensions must be in [1,31]");
    TypedImageSource source;
    source.info.width = width;
    source.info.height = height;
    source.info.numChannels = 4;
    source.info.numMipLevels = calculateNumMipLevels(width, height);
    source.info.format = HIP_AD_FORMAT_FLOAT;
    source.info.isValid = true;
    for (unsigned int level = 0; level < source.info.numMipLevels; ++level) {
        const unsigned int w = std::max(1u, width >> level), h = std::max(1u, height >> level);
        auto& pixels = source.mipPixels.emplace_back();
        pixels.reserve(size_t(w) * h * 16);
        for (unsigned int y = 0; y < h; ++y) {
            for (unsigned int x = 0; x < w; ++x) {
                auto color = filteringMipColors[level];
                if (spatialPattern) {
                    color[0] += .125f * x + .0625f * y;
                    color[1] += -.0625f * x + .125f * y;
                }
                const auto bytes = packed(color);
                pixels.insert(pixels.end(), bytes.begin(), bytes.end());
            }
        }
    }
    source.pixels = source.mipPixels.front();
    return source;
}

} }
