// SPDX-License-Identifier: MIT
#pragma once

#include "WholeMipTestData.h"
#include "TextureSamplingHarness.h"
#include "SamplingTestSupport.h"
#include <gtest/gtest.h>

namespace hip_demand { namespace test {

inline std::filesystem::path wholeMipModulePath() {
    const auto filename = std::filesystem::path(WHOLE_MIP_KERNEL_PATH).filename();
    if (const char* directory = std::getenv("HIP_DEMAND_TEST_KERNEL_DIR"))
        return findSamplingModule({std::filesystem::path(directory) / filename});
    // Reuse legacy executable/install discovery; sibling code objects are
    // deployed together by the existing test packaging.
    return findSamplingModule({samplingModulePath().parent_path() / filename, WHOLE_MIP_KERNEL_PATH});
}

class WholeMipHarness {
public:
    static constexpr uint32_t MaxSamples = 512;
    WholeMipHarness() = default;
    WholeMipHarness(const WholeMipHarness&) = delete;
    WholeMipHarness& operator=(const WholeMipHarness&) = delete;
    ~WholeMipHarness() { EXPECT_EQ(close(), hipSuccess); }

    hipStream_t stream() const { return stream_; }
    hipError_t open() { return open(wholeMipModulePath()); }
    hipError_t open(const std::filesystem::path& path) {
        hipError_t error = close();
        if (error != hipSuccess)
            return error;
        error = hipModuleLoad(&module_, path.string().c_str());
        if (error == hipSuccess)
            error = hipModuleGetFunction(&kernel_, module_, "sampleWholeMipTextures");
        if (error == hipSuccess)
            error = hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking);
        if (error == hipSuccess)
            error = hipMalloc(&inputs_, MaxSamples * sizeof(WholeMipInput));
        if (error == hipSuccess)
            error = hipMalloc(&outputs_, MaxSamples * sizeof(WholeMipResult));
        if (error != hipSuccess)
            EXPECT_EQ(close(), hipSuccess) << "Whole-mip harness rollback, primary=" << error;
        return error;
    }

    // Caller prepares on stream(), then explicitly drains processRequests().
    // The callback may resize/unload after submission to test consumer fencing.
    hipError_t sample(whole_mip_v1::DeviceContext context, const std::vector<WholeMipInput>& inputs,
                      std::vector<WholeMipResult>& results,
                      const std::function<void()>& afterLaunch = {}) {
        if (!kernel_ || inputs.empty() || inputs.size() > MaxSamples)
            return hipErrorInvalidValue;
        uint32_t count = static_cast<uint32_t>(inputs.size());
        std::vector<WholeMipResult> output(count);
        hipError_t error = hipMemcpyAsync(inputs_, inputs.data(), count * sizeof(WholeMipInput),
                                          hipMemcpyHostToDevice, stream_);
        if (error == hipSuccess) {
            void* arguments[]{&context, &inputs_, &outputs_, &count};
            error = hipModuleLaunchKernel(kernel_, (count + 63) / 64, 1, 1, 64, 1, 1,
                                           0, stream_, arguments, nullptr);
        }
        if (error == hipSuccess && afterLaunch) {
            struct CallbackFence {
                hipStream_t stream;
                ~CallbackFence() { EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess); }
            } fence{stream_};
            afterLaunch();
        }
        if (error == hipSuccess)
            error = hipMemcpyAsync(output.data(), outputs_, count * sizeof(WholeMipResult),
                                   hipMemcpyDeviceToHost, stream_);
        const auto synchronized = hipStreamSynchronize(stream_);
        if (error != hipSuccess)
            return error;
        if (synchronized != hipSuccess)
            return synchronized;
        results = std::move(output);
        return hipSuccess;
    }

    hipError_t close() {
        if (stream_) {
            const auto error = hipStreamSynchronize(stream_);
            if (error != hipSuccess)
                return error;
        }
        hipError_t first = hipSuccess;
        const auto release = [&first](auto& handle, auto destroy) {
            if (!handle)
                return;
            const auto error = destroy(handle);
            if (error == hipSuccess)
                handle = nullptr;
            else if (first == hipSuccess)
                first = error;
        };
        release(inputs_, [](auto p) { return hipFree(p); });
        release(outputs_, [](auto p) { return hipFree(p); });
        release(stream_, [](auto p) { return hipStreamDestroy(p); });
        release(module_, [](auto p) { return hipModuleUnload(p); });
        if (!module_)
            kernel_ = nullptr;
        return first;
    }

private:
    hipModule_t module_ = nullptr;
    hipFunction_t kernel_ = nullptr;
    hipStream_t stream_ = nullptr;
    WholeMipInput* inputs_ = nullptr;
    WholeMipResult* outputs_ = nullptr;
};

} }
