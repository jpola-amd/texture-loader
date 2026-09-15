// SPDX-License-Identifier: MIT
#pragma once

#include "TextureSamplingTestData.h"
#include <DemandLoading/DeviceContext.h>
#include <filesystem>
#include <functional>
#include <vector>

namespace hip_demand { namespace test {

std::filesystem::path samplingModulePath();

class TextureSamplingHarness {
public:
    static constexpr uint32_t MaxSamples = 256;

    TextureSamplingHarness() = default;
    ~TextureSamplingHarness();
    TextureSamplingHarness(const TextureSamplingHarness&) = delete;
    TextureSamplingHarness& operator=(const TextureSamplingHarness&) = delete;

    hipError_t open(const std::filesystem::path& modulePath, const char* symbol = "sampleLegacyTextures");
    // The caller must finish publishing context/storage before sampling on the private stream.
    // Returns only after the kernel and result transfer have completed.
    // afterLaunch exercises retirement while a consumer is queued on a nonblocking stream.
    hipError_t sample(DeviceContext context, const std::vector<SamplingInput>& inputs,
                      std::vector<SamplingResult>& results,
                      const std::function<void()>& afterLaunch = {});
    hipError_t close();
    bool isClosed() const { return !module_ && !stream_ && !inputs_ && !outputs_; }

private:
    hipModule_t module_ = nullptr;
    hipFunction_t kernel_ = nullptr;
    hipStream_t stream_ = nullptr;
    SamplingInput* inputs_ = nullptr;
    SamplingResult* outputs_ = nullptr;
};

} }
