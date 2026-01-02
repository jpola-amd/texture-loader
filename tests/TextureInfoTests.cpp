// SPDX-License-Identifier: MIT
// Unit tests for TextureInfo

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <ImageSource/TextureInfo.h>

namespace hip_demand {
namespace test {

// ============================================================================
// getBytesPerChannel Tests
// ============================================================================

TEST(TextureInfoTest, BytesPerChannelUnsigned8) {
    EXPECT_EQ(getBytesPerChannel(HIP_AD_FORMAT_UNSIGNED_INT8), 1u);
}

TEST(TextureInfoTest, BytesPerChannelSigned8) {
    EXPECT_EQ(getBytesPerChannel(HIP_AD_FORMAT_SIGNED_INT8), 1u);
}

TEST(TextureInfoTest, BytesPerChannelUnsigned16) {
    EXPECT_EQ(getBytesPerChannel(HIP_AD_FORMAT_UNSIGNED_INT16), 2u);
}

TEST(TextureInfoTest, BytesPerChannelSigned16) {
    EXPECT_EQ(getBytesPerChannel(HIP_AD_FORMAT_SIGNED_INT16), 2u);
}

TEST(TextureInfoTest, BytesPerChannelHalf) {
    EXPECT_EQ(getBytesPerChannel(HIP_AD_FORMAT_HALF), 2u);
}

TEST(TextureInfoTest, BytesPerChannelUnsigned32) {
    EXPECT_EQ(getBytesPerChannel(HIP_AD_FORMAT_UNSIGNED_INT32), 4u);
}

TEST(TextureInfoTest, BytesPerChannelSigned32) {
    EXPECT_EQ(getBytesPerChannel(HIP_AD_FORMAT_SIGNED_INT32), 4u);
}

TEST(TextureInfoTest, BytesPerChannelFloat) {
    EXPECT_EQ(getBytesPerChannel(HIP_AD_FORMAT_FLOAT), 4u);
}

// ============================================================================
// TextureInfo Tests
// ============================================================================

TEST(TextureInfoTest, DefaultConstruction) {
    TextureInfo info;
    EXPECT_EQ(info.width, 0u);
    EXPECT_EQ(info.height, 0u);
    EXPECT_EQ(info.format, HIP_AD_FORMAT_UNSIGNED_INT8);
    EXPECT_EQ(info.numChannels, 0u);
    EXPECT_EQ(info.numMipLevels, 0u);
    EXPECT_FALSE(info.isValid);
    EXPECT_FALSE(info.isTiled);
}

TEST(TextureInfoTest, Equality) {
    TextureInfo a, b;
    EXPECT_TRUE(a == b);
    
    a.width = 256;
    EXPECT_FALSE(a == b);
    
    b.width = 256;
    EXPECT_TRUE(a == b);
    
    a.format = HIP_AD_FORMAT_FLOAT;
    EXPECT_FALSE(a == b);
}

// ============================================================================
// getTextureSizeInBytes Tests
// ============================================================================

TEST(TextureInfoTest, SizeInBytesInvalid) {
    TextureInfo info;
    info.isValid = false;
    EXPECT_EQ(getTextureSizeInBytes(info), 0u);
}

TEST(TextureInfoTest, SizeInBytesSingleLevel) {
    TextureInfo info;
    info.width = 256;
    info.height = 256;
    info.format = HIP_AD_FORMAT_UNSIGNED_INT8;
    info.numChannels = 4;
    info.numMipLevels = 1;
    info.isValid = true;
    
    // 256 * 256 * 4 * 1 = 262144 bytes
    EXPECT_EQ(getTextureSizeInBytes(info), 256u * 256u * 4u);
}

TEST(TextureInfoTest, SizeInBytesWithMips) {
    TextureInfo info;
    info.width = 256;
    info.height = 256;
    info.format = HIP_AD_FORMAT_UNSIGNED_INT8;
    info.numChannels = 4;
    info.numMipLevels = 9;  // 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    info.isValid = true;
    
    // Sum: 256² + 128² + 64² + 32² + 16² + 8² + 4² + 2² + 1² = 65536 + 16384 + 4096 + 1024 + 256 + 64 + 16 + 4 + 1
    // = 87381 pixels * 4 channels = 349524 bytes
    size_t expected = 0;
    for (int level = 0; level < 9; ++level) {
        int w = 256 >> level;
        int h = 256 >> level;
        if (w < 1) w = 1;
        if (h < 1) h = 1;
        expected += w * h * 4;
    }
    
    EXPECT_EQ(getTextureSizeInBytes(info), expected);
}

TEST(TextureInfoTest, SizeInBytesFloat32) {
    TextureInfo info;
    info.width = 128;
    info.height = 128;
    info.format = HIP_AD_FORMAT_FLOAT;
    info.numChannels = 3;
    info.numMipLevels = 1;
    info.isValid = true;
    
    // 128 * 128 * 3 * 4 = 196608 bytes
    EXPECT_EQ(getTextureSizeInBytes(info), 128u * 128u * 3u * 4u);
}

TEST(TextureInfoTest, SizeInBytesNonSquare) {
    TextureInfo info;
    info.width = 512;
    info.height = 128;
    info.format = HIP_AD_FORMAT_UNSIGNED_INT8;
    info.numChannels = 4;
    info.numMipLevels = 1;
    info.isValid = true;
    
    EXPECT_EQ(getTextureSizeInBytes(info), 512u * 128u * 4u);
}

}  // namespace test
}  // namespace hip_demand
