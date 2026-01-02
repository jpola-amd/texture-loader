// SPDX-License-Identifier: MIT
// Unit tests for DemandTextureLoader

#include "TestUtils.h"
#include <DemandLoading/DemandTextureLoader.h>
#include <DemandLoading/DeviceContext.h>
#include <ImageSource/ImageSource.h>
#include <ImageSource/TextureInfo.h>

#include <thread>
#include <chrono>
#include <set>

namespace hip_demand {
namespace test {

// ============================================================================
// Mock ImageSource for Testing
// ============================================================================

class MockImageSource : public ImageSource {
public:
    MockImageSource(unsigned int width, unsigned int height, unsigned int channels,
                    unsigned long long contentHash = 0)
        : width_(width), height_(height), channels_(channels), contentHash_(contentHash) {
        // Generate test data
        data_.resize(width * height * channels);
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] = static_cast<unsigned char>(i % 256);
        }
    }
    
    void open(TextureInfo* info) override {
        isOpen_ = true;
        if (info) {
            info->width = width_;
            info->height = height_;
            info->format = HIP_AD_FORMAT_UNSIGNED_INT8;
            info->numChannels = channels_;
            info->numMipLevels = calculateNumMipLevels(width_, height_);
            info->isValid = true;
            info->isTiled = false;
        }
    }
    
    void close() override { isOpen_ = false; }
    bool isOpen() const override { return isOpen_; }
    
    const TextureInfo& getInfo() const override {
        static TextureInfo info;
        info.width = width_;
        info.height = height_;
        info.format = HIP_AD_FORMAT_UNSIGNED_INT8;
        info.numChannels = channels_;
        info.numMipLevels = calculateNumMipLevels(width_, height_);
        info.isValid = true;
        info.isTiled = false;
        return info;
    }
    
    bool readMipLevel(char* dest, unsigned int mipLevel,
                     unsigned int expectedWidth, unsigned int expectedHeight,
                     hipStream_t stream = 0) override {
        if (mipLevel != 0) return false;
        if (expectedWidth != width_ || expectedHeight != height_) return false;
        std::memcpy(dest, data_.data(), data_.size());
        bytesRead_ += data_.size();
        return true;
    }
    
    bool readBaseColor(float4& dest) override {
        dest = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
        return true;
    }
    
    unsigned long long getNumBytesRead() const override { return bytesRead_; }
    double getTotalReadTime() const override { return 0.001; }
    
    /// Return the content hash for deduplication
    unsigned long long getHash(hipStream_t stream = 0) const override {
        (void)stream;
        return contentHash_;
    }

private:
    unsigned int width_;
    unsigned int height_;
    unsigned int channels_;
    unsigned long long contentHash_;
    std::vector<unsigned char> data_;
    bool isOpen_ = false;
    unsigned long long bytesRead_ = 0;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(HipTestFixture, DefaultConstruction) {
    DemandTextureLoader loader;
    EXPECT_EQ(loader.getLastError(), LoaderError::Success);
    EXPECT_EQ(loader.getResidentTextureCount(), 0u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    EXPECT_FALSE(loader.isAborted());
}

TEST_F(HipTestFixture, CustomOptions) {
    LoaderOptions options;
    options.maxTextures = 128;
    options.maxRequestsPerLaunch = 512;
    options.maxTextureMemory = 512 * 1024 * 1024;
    options.maxThreads = 4;
    
    DemandTextureLoader loader(options);
    EXPECT_EQ(loader.getLastError(), LoaderError::Success);
    EXPECT_EQ(loader.getMaxTextureMemory(), 512u * 1024u * 1024u);
}

// ============================================================================
// Texture Creation Tests
// ============================================================================

TEST_F(LoaderTestFixture, CreateTextureFromMemory) {
    auto imageData = generateTestImage(64, 64, 4);
    
    TextureHandle handle = loader_->createTextureFromMemory(
        imageData.data(), 64, 64, 4);
    
    EXPECT_TRUE(handle.valid);
    EXPECT_EQ(handle.error, LoaderError::Success);
    EXPECT_EQ(handle.width, 64);
    EXPECT_EQ(handle.height, 64);
    EXPECT_EQ(handle.channels, 4);
    EXPECT_EQ(handle.id, 0u);  // First texture
}

TEST_F(LoaderTestFixture, CreateTextureFromImageSource) {
    auto imgSource = std::make_shared<MockImageSource>(128, 128, 4);
    
    TextureHandle handle = loader_->createTexture(imgSource);
    
    EXPECT_TRUE(handle.valid);
    EXPECT_EQ(handle.error, LoaderError::Success);
    EXPECT_EQ(handle.width, 128);
    EXPECT_EQ(handle.height, 128);
    EXPECT_EQ(handle.channels, 4);
    EXPECT_EQ(handle.id, 0u);
}

TEST_F(LoaderTestFixture, CreateTextureFromImageSourceWithDesc) {
    auto imgSource = std::make_shared<MockImageSource>(64, 64, 3);
    
    TextureDesc desc;
    desc.addressMode[0] = hipAddressModeClamp;
    desc.addressMode[1] = hipAddressModeClamp;
    desc.filterMode = hipFilterModePoint;
    
    TextureHandle handle = loader_->createTexture(imgSource, desc);
    
    EXPECT_TRUE(handle.valid);
    EXPECT_EQ(handle.error, LoaderError::Success);
    EXPECT_EQ(handle.width, 64);
    EXPECT_EQ(handle.height, 64);
    EXPECT_EQ(handle.channels, 3);
}

TEST_F(LoaderTestFixture, CreateTextureFromNullImageSource) {
    std::shared_ptr<ImageSource> nullSource = nullptr;
    
    TextureHandle handle = loader_->createTexture(nullSource);
    
    EXPECT_FALSE(handle.valid);
    EXPECT_EQ(handle.error, LoaderError::InvalidParameter);
}

TEST_F(LoaderTestFixture, CreateMultipleTextures) {
    std::vector<TextureHandle> handles;
    
    for (int i = 0; i < 10; ++i) {
        auto imageData = generateTestImage(32, 32, 4, i * 25, 128, 255 - i * 25);
        TextureHandle handle = loader_->createTextureFromMemory(
            imageData.data(), 32, 32, 4);
        
        EXPECT_TRUE(handle.valid);
        EXPECT_EQ(handle.id, static_cast<uint32_t>(i));
        handles.push_back(handle);
    }
    
    EXPECT_EQ(handles.size(), 10u);
}

TEST_F(LoaderTestFixture, CreateTextureInvalidParams) {
    // Null data
    TextureHandle handle = loader_->createTextureFromMemory(nullptr, 64, 64, 4);
    EXPECT_FALSE(handle.valid);
    EXPECT_EQ(handle.error, LoaderError::InvalidParameter);
    
    // Zero width
    auto imageData = generateTestImage(64, 64, 4);
    handle = loader_->createTextureFromMemory(imageData.data(), 0, 64, 4);
    EXPECT_FALSE(handle.valid);
    EXPECT_EQ(handle.error, LoaderError::InvalidParameter);
    
    // Zero height
    handle = loader_->createTextureFromMemory(imageData.data(), 64, 0, 4);
    EXPECT_FALSE(handle.valid);
    EXPECT_EQ(handle.error, LoaderError::InvalidParameter);
    
    // Zero channels
    handle = loader_->createTextureFromMemory(imageData.data(), 64, 64, 0);
    EXPECT_FALSE(handle.valid);
    EXPECT_EQ(handle.error, LoaderError::InvalidParameter);
}

TEST_F(LoaderTestFixture, CreateTextureWithDescriptor) {
    auto imageData = generateTestImage(64, 64, 4);
    
    TextureDesc desc;
    desc.addressMode[0] = hipAddressModeClamp;
    desc.addressMode[1] = hipAddressModeClamp;
    desc.filterMode = hipFilterModePoint;
    desc.sRGB = true;
    
    TextureHandle handle = loader_->createTextureFromMemory(
        imageData.data(), 64, 64, 4, desc);
    
    EXPECT_TRUE(handle.valid);
    EXPECT_EQ(handle.error, LoaderError::Success);
}

// ============================================================================
// Device Context Tests
// ============================================================================

TEST_F(LoaderTestFixture, GetDeviceContext) {
    DeviceContext ctx = loader_->getDeviceContext();
    
    EXPECT_NE(ctx.textures, nullptr);
    EXPECT_NE(ctx.requests, nullptr);
    EXPECT_NE(ctx.residentFlags, nullptr);
    EXPECT_NE(ctx.requestCount, nullptr);
    EXPECT_NE(ctx.requestOverflow, nullptr);
    EXPECT_EQ(ctx.maxTextures, 64u);
    EXPECT_EQ(ctx.maxRequests, 256u);
}

TEST_F(LoaderTestFixture, LaunchPrepare) {
    auto imageData = generateTestImage(64, 64, 4);
    loader_->createTextureFromMemory(imageData.data(), 64, 64, 4);
    
    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    
    // Should not throw or fail
    loader_->launchPrepare(stream);
    
    DeviceContext ctx = loader_->getDeviceContext();
    EXPECT_NE(ctx.textures, nullptr);
    
    hipStreamDestroy(stream);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(LoaderTestFixture, InitialStatistics) {
    EXPECT_EQ(loader_->getResidentTextureCount(), 0u);
    EXPECT_EQ(loader_->getTotalTextureMemory(), 0u);
    EXPECT_EQ(loader_->getRequestCount(), 0u);
    EXPECT_FALSE(loader_->hadRequestOverflow());
}

// ============================================================================
// Eviction Control Tests
// ============================================================================

TEST_F(LoaderTestFixture, SetMaxTextureMemory) {
    size_t newLimit = 128 * 1024 * 1024;  // 128 MB
    loader_->setMaxTextureMemory(newLimit);
    EXPECT_EQ(loader_->getMaxTextureMemory(), newLimit);
}

TEST_F(LoaderTestFixture, EnableDisableEviction) {
    // These should not throw
    loader_->enableEviction(true);
    loader_->enableEviction(false);
    loader_->enableEviction(true);
}

TEST_F(LoaderTestFixture, UpdateEvictionPriority) {
    auto imageData = generateTestImage(64, 64, 4);
    TextureHandle handle = loader_->createTextureFromMemory(
        imageData.data(), 64, 64, 4);
    
    ASSERT_TRUE(handle.valid);
    
    // These should not throw
    loader_->updateEvictionPriority(handle.id, EvictionPriority::Low);
    loader_->updateEvictionPriority(handle.id, EvictionPriority::High);
    loader_->updateEvictionPriority(handle.id, EvictionPriority::KeepResident);
    loader_->updateEvictionPriority(handle.id, EvictionPriority::Normal);
}

// ============================================================================
// Unload Tests
// ============================================================================

TEST_F(LoaderTestFixture, UnloadTexture) {
    auto imageData = generateTestImage(64, 64, 4);
    TextureHandle handle = loader_->createTextureFromMemory(
        imageData.data(), 64, 64, 4);
    
    ASSERT_TRUE(handle.valid);
    
    // Should not throw
    loader_->unloadTexture(handle.id);
}

TEST_F(LoaderTestFixture, UnloadAll) {
    // Create several textures
    for (int i = 0; i < 5; ++i) {
        auto imageData = generateTestImage(32, 32, 4);
        loader_->createTextureFromMemory(imageData.data(), 32, 32, 4);
    }
    
    // Should not throw
    loader_->unloadAll();
    
    EXPECT_EQ(loader_->getResidentTextureCount(), 0u);
}

// ============================================================================
// Abort Tests
// ============================================================================

TEST_F(LoaderTestFixture, AbortLoader) {
    EXPECT_FALSE(loader_->isAborted());
    
    loader_->abort();
    
    EXPECT_TRUE(loader_->isAborted());
}

TEST_F(LoaderTestFixture, AbortPreventsNewRequests) {
    auto imageData = generateTestImage(64, 64, 4);
    TextureHandle handle = loader_->createTextureFromMemory(
        imageData.data(), 64, 64, 4);
    
    ASSERT_TRUE(handle.valid);
    
    loader_->abort();
    
    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    
    DeviceContext ctx = loader_->getDeviceContext();
    
    // processRequests should return 0 after abort
    size_t loaded = loader_->processRequests(stream, ctx);
    EXPECT_EQ(loaded, 0u);
    
    hipStreamDestroy(stream);
}

// ============================================================================
// Texture Deduplication Tests
// ============================================================================

TEST_F(LoaderTestFixture, DeduplicateSameImageSourcePointer) {
    // Same ImageSource pointer should return the same texture ID
    auto imgSource = std::make_shared<MockImageSource>(64, 64, 4);
    
    TextureHandle handle1 = loader_->createTexture(imgSource);
    TextureHandle handle2 = loader_->createTexture(imgSource);
    
    EXPECT_TRUE(handle1.valid);
    EXPECT_TRUE(handle2.valid);
    EXPECT_EQ(handle1.id, handle2.id);  // Same texture ID
}

TEST_F(LoaderTestFixture, DeduplicateSameContentHash) {
    // Different ImageSource objects with same content hash should return same texture ID
    const unsigned long long sharedHash = 0xDEADBEEF12345678ULL;
    
    auto imgSource1 = std::make_shared<MockImageSource>(64, 64, 4, sharedHash);
    auto imgSource2 = std::make_shared<MockImageSource>(64, 64, 4, sharedHash);
    
    // Verify they are different objects
    EXPECT_NE(imgSource1.get(), imgSource2.get());
    
    // But same hash
    EXPECT_EQ(imgSource1->getHash(), imgSource2->getHash());
    
    TextureHandle handle1 = loader_->createTexture(imgSource1);
    TextureHandle handle2 = loader_->createTexture(imgSource2);
    
    EXPECT_TRUE(handle1.valid);
    EXPECT_TRUE(handle2.valid);
    EXPECT_EQ(handle1.id, handle2.id);  // Same texture ID due to content hash match
}

TEST_F(LoaderTestFixture, NoDuplicateForDifferentContentHash) {
    // Different content hashes should create different textures
    auto imgSource1 = std::make_shared<MockImageSource>(64, 64, 4, 0x1111111111111111ULL);
    auto imgSource2 = std::make_shared<MockImageSource>(64, 64, 4, 0x2222222222222222ULL);
    
    TextureHandle handle1 = loader_->createTexture(imgSource1);
    TextureHandle handle2 = loader_->createTexture(imgSource2);
    
    EXPECT_TRUE(handle1.valid);
    EXPECT_TRUE(handle2.valid);
    EXPECT_NE(handle1.id, handle2.id);  // Different texture IDs
}

TEST_F(LoaderTestFixture, NoDuplicateForZeroHash) {
    // Zero hash (default) should not trigger content-based deduplication
    auto imgSource1 = std::make_shared<MockImageSource>(64, 64, 4, 0);  // Zero hash
    auto imgSource2 = std::make_shared<MockImageSource>(64, 64, 4, 0);  // Zero hash
    
    TextureHandle handle1 = loader_->createTexture(imgSource1);
    TextureHandle handle2 = loader_->createTexture(imgSource2);
    
    EXPECT_TRUE(handle1.valid);
    EXPECT_TRUE(handle2.valid);
    EXPECT_NE(handle1.id, handle2.id);  // Different texture IDs (no dedup for zero hash)
}

TEST_F(LoaderTestFixture, DeduplicateAfterPointerCheckFails) {
    // Test that content hash check works even after pointer check fails
    const unsigned long long sharedHash = 0xCAFEBABE00000001ULL;
    
    auto imgSource1 = std::make_shared<MockImageSource>(128, 128, 4, sharedHash);
    TextureHandle handle1 = loader_->createTexture(imgSource1);
    EXPECT_TRUE(handle1.valid);
    EXPECT_EQ(handle1.id, 0u);
    
    // Create a different ImageSource with the same hash
    auto imgSource2 = std::make_shared<MockImageSource>(128, 128, 4, sharedHash);
    TextureHandle handle2 = loader_->createTexture(imgSource2);
    
    // Should reuse the first texture via content hash
    EXPECT_TRUE(handle2.valid);
    EXPECT_EQ(handle2.id, handle1.id);
    
    // Now use the same pointer again - should also work (via pointer check)
    TextureHandle handle3 = loader_->createTexture(imgSource2);
    EXPECT_TRUE(handle3.valid);
    EXPECT_EQ(handle3.id, handle1.id);
}

TEST_F(LoaderTestFixture, DeduplicateMixedImageSources) {
    // Mix of ImageSource objects with different deduplication scenarios
    const unsigned long long hash1 = 0xAAAAAAAAAAAAAAAAULL;
    const unsigned long long hash2 = 0xBBBBBBBBBBBBBBBBULL;
    
    auto srcA1 = std::make_shared<MockImageSource>(32, 32, 4, hash1);
    auto srcA2 = std::make_shared<MockImageSource>(32, 32, 4, hash1);  // Same hash as A1
    auto srcB1 = std::make_shared<MockImageSource>(32, 32, 4, hash2);
    auto srcB2 = std::make_shared<MockImageSource>(32, 32, 4, hash2);  // Same hash as B1
    auto srcC = std::make_shared<MockImageSource>(32, 32, 4, 0);       // Zero hash
    auto srcD = std::make_shared<MockImageSource>(32, 32, 4, 0);       // Zero hash (different)
    
    TextureHandle hA1 = loader_->createTexture(srcA1);
    TextureHandle hB1 = loader_->createTexture(srcB1);
    TextureHandle hC = loader_->createTexture(srcC);
    TextureHandle hD = loader_->createTexture(srcD);
    TextureHandle hA2 = loader_->createTexture(srcA2);  // Should match A1
    TextureHandle hB2 = loader_->createTexture(srcB2);  // Should match B1
    
    // All valid
    EXPECT_TRUE(hA1.valid);
    EXPECT_TRUE(hA2.valid);
    EXPECT_TRUE(hB1.valid);
    EXPECT_TRUE(hB2.valid);
    EXPECT_TRUE(hC.valid);
    EXPECT_TRUE(hD.valid);
    
    // A1 and A2 should match (same hash)
    EXPECT_EQ(hA1.id, hA2.id);
    
    // B1 and B2 should match (same hash)
    EXPECT_EQ(hB1.id, hB2.id);
    
    // A and B should be different
    EXPECT_NE(hA1.id, hB1.id);
    
    // C and D should be different (zero hash = no content dedup)
    EXPECT_NE(hC.id, hD.id);
    
    // We should have 4 unique texture IDs: A, B, C, D
    std::set<uint32_t> uniqueIds = {hA1.id, hB1.id, hC.id, hD.id};
    EXPECT_EQ(uniqueIds.size(), 4u);
}

// ============================================================================
// Error String Tests
// ============================================================================

TEST(ErrorStringTest, AllErrorCodes) {
    EXPECT_STREQ(getErrorString(LoaderError::Success), "Success");
    EXPECT_STREQ(getErrorString(LoaderError::InvalidTextureId), "Invalid texture ID");
    EXPECT_STREQ(getErrorString(LoaderError::MaxTexturesExceeded), "Maximum textures exceeded");
    EXPECT_STREQ(getErrorString(LoaderError::FileNotFound), "File not found");
    EXPECT_STREQ(getErrorString(LoaderError::ImageLoadFailed), "Image load failed");
    EXPECT_STREQ(getErrorString(LoaderError::OutOfMemory), "Out of memory");
    EXPECT_STREQ(getErrorString(LoaderError::InvalidParameter), "Invalid parameter");
    EXPECT_STREQ(getErrorString(LoaderError::HipError), "HIP error");
}

}  // namespace test
}  // namespace hip_demand
