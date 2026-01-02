// SPDX-License-Identifier: MIT
// Test utilities and fixtures

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <DemandLoading/DemandTextureLoader.h>

#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>

namespace hip_demand {
namespace test {

/// Test fixture that initializes HIP
class HipTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        hipError_t err = hipInit(0);
        ASSERT_EQ(err, hipSuccess) << "Failed to initialize HIP";
        
        int deviceCount = 0;
        err = hipGetDeviceCount(&deviceCount);
        ASSERT_EQ(err, hipSuccess) << "Failed to get device count";
        ASSERT_GT(deviceCount, 0) << "No HIP devices available";
        
        err = hipSetDevice(0);
        ASSERT_EQ(err, hipSuccess) << "Failed to set device";
    }
    
    void TearDown() override {
        hipDeviceReset();
    }
};

/// Test fixture with a DemandTextureLoader instance
class LoaderTestFixture : public HipTestFixture {
protected:
    void SetUp() override {
        HipTestFixture::SetUp();
        
        LoaderOptions options;
        options.maxTextures = 64;
        options.maxRequestsPerLaunch = 256;
        options.maxTextureMemory = 256 * 1024 * 1024;  // 256 MB for tests
        options.maxThreads = 2;
        
        loader_ = std::make_unique<DemandTextureLoader>(options);
        ASSERT_NE(loader_, nullptr);
        ASSERT_EQ(loader_->getLastError(), LoaderError::Success);
    }
    
    void TearDown() override {
        loader_.reset();
        HipTestFixture::TearDown();
    }
    
    std::unique_ptr<DemandTextureLoader> loader_;
};

/// Generate test image data (solid color or gradient)
inline std::vector<uint8_t> generateTestImage(int width, int height, int channels, 
                                               uint8_t r = 128, uint8_t g = 128, 
                                               uint8_t b = 128, uint8_t a = 255) {
    std::vector<uint8_t> data(width * height * channels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t idx = (y * width + x) * channels;
            data[idx + 0] = r;
            if (channels > 1) data[idx + 1] = g;
            if (channels > 2) data[idx + 2] = b;
            if (channels > 3) data[idx + 3] = a;
        }
    }
    return data;
}

/// Generate gradient test image
inline std::vector<uint8_t> generateGradientImage(int width, int height, int channels) {
    std::vector<uint8_t> data(width * height * channels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t idx = (y * width + x) * channels;
            data[idx + 0] = static_cast<uint8_t>(x * 255 / width);
            if (channels > 1) data[idx + 1] = static_cast<uint8_t>(y * 255 / height);
            if (channels > 2) data[idx + 2] = static_cast<uint8_t>((x + y) * 127 / (width + height));
            if (channels > 3) data[idx + 3] = 255;
        }
    }
    return data;
}

/// Get test images directory path
inline std::string getTestImagesPath() {
    // Look for test_images relative to the executable or source
    std::filesystem::path candidates[] = {
        "test_images",
        "../test_images",
        "../../test_images",
    };
    
    for (const auto& path : candidates) {
        if (std::filesystem::exists(path)) {
            return path.string();
        }
    }
    return "test_images";  // Default
}

}  // namespace test
}  // namespace hip_demand
