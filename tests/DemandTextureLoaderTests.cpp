// SPDX-License-Identifier: MIT
// Unit tests for DemandTextureLoader

#include "TestUtils.h"
#include <DemandLoading/DemandTextureLoader.h>
#include <DemandLoading/DeviceContext.h>

#include <thread>
#include <chrono>

namespace hip_demand {
namespace test {

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
