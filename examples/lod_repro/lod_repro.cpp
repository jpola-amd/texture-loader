// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>

constexpr unsigned SampleCount = 17;
constexpr unsigned BaseWidth = 8;

#ifdef LOD_REPRO_KERNEL

extern "C" __global__ void sampleLod(hipTextureObject_t texture, float4* output) {
    const unsigned i = threadIdx.x;
    if (i >= SampleCount)
        return;
    const float lod = i / 16.0f;
    output[i] = tex2DLod<float4>(texture, 0.5f, 0.5f, lod);
    const float footprint = exp2f(lod) / BaseWidth;
    output[SampleCount + i] = tex2DGrad<float4>(
        texture, 0.5f, 0.5f, make_float2(footprint, 0), make_float2(0, footprint));
}

#else

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

bool check(hipError_t error, const char* operation) {
    if (error == hipSuccess)
        return true;
    std::cerr << operation << ": " << hipGetErrorName(error) << " ("
              << static_cast<int>(error) << "): " << hipGetErrorString(error) << '\n';
    return false;
}

struct Resources {
    bool& cleanupFailed;
    hipModule_t module = nullptr;
    hipMipmappedArray_t mipmap = nullptr;
    hipTextureObject_t texture{};
    float4* output = nullptr;
    bool launched = false;

    ~Resources() {
        if (launched && !check(hipDeviceSynchronize(), "cleanup: synchronize")) {
            cleanupFailed = true;
            return;
        }
        if (output && !check(hipFree(output), "cleanup: output"))
            cleanupFailed = true;
        const bool samplerFreed = !texture || check(hipDestroyTextureObject(texture), "cleanup: sampler");
        if (!samplerFreed)
            cleanupFailed = true;
        if (samplerFreed && mipmap && !check(hipFreeMipmappedArray(mipmap), "cleanup: mipmap"))
            cleanupFailed = true;
        if (module && !check(hipModuleUnload(module), "cleanup: module"))
            cleanupFailed = true;
    }
};

#define HIP_TRY(operation) do { if (!check((operation), #operation)) return 2; } while (false)

float errorFromGray(float4 value, float expected) {
    if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
        !std::isfinite(value.z) || !std::isfinite(value.w))
        return std::numeric_limits<float>::infinity();
    return std::max({std::abs(value.x - expected), std::abs(value.y - expected),
                     std::abs(value.z - expected), std::abs(value.w - 1.0f)});
}

int run(Resources& resources, const char* kernelPath, int device) {
    int devices = 0;
    HIP_TRY(hipGetDeviceCount(&devices));
    if (device >= devices) {
        std::cerr << "Device " << device << " is unavailable; found " << devices << " device(s).\n";
        return 2;
    }
    HIP_TRY(hipSetDevice(device));
    hipDeviceProp_t properties{};
    HIP_TRY(hipGetDeviceProperties(&properties, device));
    int runtime = 0, driver = 0;
    HIP_TRY(hipRuntimeGetVersion(&runtime));
    HIP_TRY(hipDriverGetVersion(&driver));
#ifdef _WIN32
    std::cout << "OS: Windows\n";
#else
    std::cout << "OS: Linux\n";
#endif
    std::cout << "Device: " << device << " / " << properties.name << " / " << properties.gcnArchName
              << "\nHIP headers: " << HIP_VERSION_MAJOR << '.' << HIP_VERSION_MINOR << '.' << HIP_VERSION_PATCH
              << "\nHIP runtime API version: " << runtime << "\nHIP driver API version: " << driver
              << "\nKernel: " << kernelPath << '\n';
    HIP_TRY(hipModuleLoad(&resources.module, kernelPath));
    hipFunction_t function = nullptr;
    HIP_TRY(hipModuleGetFunction(&function, resources.module, "sampleLod"));

    // Authored FLOAT RGBA mips: level 0 is black; level 1 is white.
    // The sampled RGB value is therefore the blend weight itself.
    const auto channel = hipCreateChannelDesc<float4>();
    HIP_TRY(hipMallocMipmappedArray(&resources.mipmap, &channel,
                                    make_hipExtent(BaseWidth, BaseWidth, 0), 2));
    for (unsigned level = 0; level < 2; ++level) {
        const unsigned width = BaseWidth >> level;
        const float gray = static_cast<float>(level);
        std::vector<float4> pixels(width * width, make_float4(gray, gray, gray, 1));
        std::vector<float4> readback(pixels.size());
        hipArray_t array = nullptr;
        HIP_TRY(hipGetMipmappedArrayLevel(&array, resources.mipmap, level));
        const size_t rowBytes = width * sizeof(float4);
        HIP_TRY(hipMemcpy2DToArray(array, 0, 0, pixels.data(), rowBytes, rowBytes, width, hipMemcpyHostToDevice));
        HIP_TRY(hipMemcpy2DFromArray(readback.data(), rowBytes, array, 0, 0, rowBytes, width, hipMemcpyDeviceToHost));
        if (std::memcmp(pixels.data(), readback.data(), pixels.size() * sizeof(float4)) != 0) {
            std::cerr << "Mip " << level << " upload/readback mismatch; cannot test filtering.\n";
            return 2;
        }
    }
    std::cout << "Mip upload/readback: exact (8x8 black, 4x4 white, FLOAT RGBA)\n";

    hipResourceDesc resource{};
    resource.resType = hipResourceTypeMipmappedArray;
    resource.res.mipmap.mipmap = resources.mipmap;
    hipTextureDesc sampler{};
    sampler.addressMode[0] = sampler.addressMode[1] = hipAddressModeClamp;
    sampler.filterMode = hipFilterModeLinear;
    sampler.mipmapFilterMode = hipFilterModeLinear;
    sampler.readMode = hipReadModeElementType;
    sampler.normalizedCoords = 1;
    sampler.minMipmapLevelClamp = 0;
    sampler.maxMipmapLevelClamp = 1;
    HIP_TRY(hipCreateTextureObject(&resources.texture, &resource, &sampler, nullptr));

    hipTextureDesc returned{};
    HIP_TRY(hipGetTextureObjectTextureDesc(&returned, resources.texture));
    std::cout << "Requested/returned mip filter: " << sampler.mipmapFilterMode << '/' << returned.mipmapFilterMode
              << "; spatial filter: " << sampler.filterMode << '/' << returned.filterMode
              << "; anisotropy: " << sampler.maxAnisotropy << '/' << returned.maxAnisotropy << '\n';

    std::array<float4, 2 * SampleCount> results{};
    HIP_TRY(hipMalloc(&resources.output, sizeof(results)));
    void* arguments[]{&resources.texture, &resources.output};
    resources.launched = true;
    HIP_TRY(hipModuleLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, nullptr, arguments, nullptr));
    HIP_TRY(hipDeviceSynchronize());
    resources.launched = false;
    HIP_TRY(hipMemcpy(results.data(), resources.output, sizeof(results), hipMemcpyDeviceToHost));

    constexpr float ExplicitTolerance = 1e-6f;
    // Allow the existing harness's fractional-gradient LOD/weight precision allowance.
    constexpr float GradientTolerance = 1e-6f + 1.0f / 128;
    float maxExplicitError = 0, maxGradientError = 0;
    std::cout << "\nLOD,expected,explicit_LOD,gradient,explicit_error,gradient_error\n" << std::fixed << std::setprecision(6);
    for (unsigned i = 0; i < SampleCount; ++i) {
        const float expected = i / 16.0f;
        const float explicitError = errorFromGray(results[i], expected);
        const float gradientError = errorFromGray(results[SampleCount + i], expected);
        maxExplicitError = std::max(maxExplicitError, explicitError);
        maxGradientError = std::max(maxGradientError, gradientError);
        std::cout << expected << ',' << expected << ',' << results[i].x << ',' << results[SampleCount + i].x
                  << ',' << explicitError << ',' << gradientError << '\n';
    }
    std::cout << "\nMax explicit error: " << maxExplicitError << " (tolerance " << ExplicitTolerance << ')'
              << "\nMax gradient error: " << maxGradientError << " (tolerance " << GradientTolerance << ")\n";
    const bool matches = maxExplicitError <= ExplicitTolerance && maxGradientError <= GradientTolerance;
    std::cout << "RESULT: " << (matches ? "MATCH" : "DIFFERENT")
              << " from analytical linear mip blending.\n";
    return matches ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: lod_repro <lod_repro_kernel.co> [device ordinal, default 0]\n"
                     "Exit codes: 0=matching samples, 1=numerical difference, 2=setup/runtime/cleanup error.\n";
        return 0;
    }
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: lod_repro <lod_repro_kernel.co> [device ordinal, default 0]\n";
        return 2;
    }
    int device = 0;
    if (argc == 3) {
        const std::string_view text = argv[2];
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), device);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || device < 0) {
            std::cerr << "Device ordinal must be a nonnegative integer.\n";
            return 2;
        }
    }
    bool cleanupFailed = false;
    int result;
    {
        Resources resources{cleanupFailed};
        result = run(resources, argv[1], device);
    }
    return cleanupFailed ? 2 : result;
}
#endif
