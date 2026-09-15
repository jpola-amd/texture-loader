// SPDX-License-Identifier: MIT
#include "TextureSamplingHarness.h"
#include "SamplingTestSupport.h"
#include <gtest/gtest.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace hip_demand { namespace test {
namespace {

std::filesystem::path executablePath() {
#ifdef _WIN32
    std::vector<wchar_t> path(512);
    for (;;) {
        const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (!length)
            throw std::runtime_error("GetModuleFileNameW failed locating sampling test executable");
        if (length < path.size())
            return std::filesystem::path(std::wstring(path.data(), length));
        path.resize(path.size() * 2);
    }
#else
    return std::filesystem::read_symlink("/proc/self/exe");
#endif
}

} // namespace

std::filesystem::path samplingModulePath() {
    return findSamplingModule(samplingModuleCandidates(
        executablePath(), SAMPLING_KERNEL_PATH, std::getenv("HIP_DEMAND_TEST_KERNEL_DIR")));
}

TextureSamplingHarness::~TextureSamplingHarness() {
    EXPECT_EQ(close(), hipSuccess) << "Sampling fixture cleanup failed";
}

hipError_t TextureSamplingHarness::open(const std::filesystem::path& path, const char* symbol) {
    hipError_t error = close();
    if (error != hipSuccess)
        return error;
    error = hipModuleLoad(&module_, path.string().c_str());
    if (error == hipSuccess)
        error = hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking);
    if (error == hipSuccess)
        error = hipMalloc(&inputs_, sizeof(SamplingInput) * MaxSamples);
    if (error == hipSuccess)
        error = hipMalloc(&outputs_, sizeof(SamplingResult) * MaxSamples);
    if (error == hipSuccess)
        error = hipModuleGetFunction(&kernel_, module_, symbol);
    if (error != hipSuccess)
        EXPECT_EQ(close(), hipSuccess) << "Sampling fixture rollback failed; primary HIP error=" << error;
    return error;
}

hipError_t TextureSamplingHarness::sample(DeviceContext context, const std::vector<SamplingInput>& inputs,
                                          std::vector<SamplingResult>& results,
                                          const std::function<void()>& afterLaunch) {
    if (!kernel_ || inputs.empty() || inputs.size() > MaxSamples)
        return hipErrorInvalidValue;
    uint32_t count = static_cast<uint32_t>(inputs.size());
    std::vector<SamplingResult> output(count);
    hipError_t error = hipMemcpyAsync(inputs_, inputs.data(), inputs.size() * sizeof(SamplingInput),
                                       hipMemcpyHostToDevice, stream_);
    if (error == hipSuccess)
        error = hipMemsetAsync(outputs_, 0xff, count * sizeof(SamplingResult), stream_);
    if (error == hipSuccess) {
        void* arguments[]{&context, &inputs_, &outputs_, &count};
        error = hipModuleLaunchKernel(kernel_, (count + 63) / 64, 1, 1, 64, 1, 1,
                                       0, stream_, arguments, nullptr);
    }
    if (error == hipSuccess && afterLaunch)
        afterLaunch();
    if (error == hipSuccess)
        error = hipMemcpyAsync(output.data(), outputs_, count * sizeof(SamplingResult),
                               hipMemcpyDeviceToHost, stream_);
    // Finish queued work even on failure before any host input/output storage can die.
    const hipError_t synchronization = hipStreamSynchronize(stream_);
    if (error != hipSuccess)
        return error;
    if (synchronization != hipSuccess)
        return synchronization;
    results = std::move(output);
    return hipSuccess;
}

hipError_t TextureSamplingHarness::close() {
    if (stream_) {
        const hipError_t error = hipStreamSynchronize(stream_);
        if (error != hipSuccess)
            return error;
    }
    hipError_t firstError = hipSuccess;
    const auto release = [&firstError](auto& handle, auto destroy) {
        if (!handle)
            return;
        const hipError_t error = destroy(handle);
        if (error == hipSuccess)
            handle = nullptr;
        else if (firstError == hipSuccess)
            firstError = error;
    };
    release(outputs_, [](auto pointer) { return hipFree(pointer); });
    release(inputs_, [](auto pointer) { return hipFree(pointer); });
    release(stream_, [](auto stream) { return hipStreamDestroy(stream); });
    release(module_, [](auto module) { return hipModuleUnload(module); });
    if (!module_)
        kernel_ = nullptr;
    return firstError;
}

} }
