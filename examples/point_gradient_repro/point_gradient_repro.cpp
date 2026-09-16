// SPDX-License-Identifier: MIT
// Native HIP only. Compile once as a code object and once as a host executable.
#include <hip/hip_runtime.h>
#include <cmath>

constexpr unsigned int Width = 19;
constexpr unsigned int Levels = 5;
constexpr unsigned int Samples = 4;

#ifdef POINT_GRADIENT_KERNEL
extern "C" __global__ void samplePointGradients(
    hipTextureObject_t full, hipTextureObject_t suffix, float4* output) {
    const unsigned int index = threadIdx.x;
    if (index >= Samples) return;
    const float lods[Samples]{.515625f, .75f, 1.f, 1.25f};
    const float lod = lods[index];
    const float extent = exp2f(lod);
    const float2 dx = make_float2(extent / Width, 0);
    const float2 dy = make_float2(0, extent / Width);
    const float correction = float(Width) / (9.f * 2.f);
    const float2 sx = make_float2(dx.x * correction, 0);
    const float2 sy = make_float2(0, dy.y * correction);
    output[index * 4] = tex2DLod<float4>(full, 1.125f, -.125f, lod);
    output[index * 4 + 1] = tex2DGrad<float4>(full, 1.125f, -.125f, dx, dy);
    output[index * 4 + 2] = tex2DLod<float4>(suffix, 1.125f, -.125f, lod - 1.f);
    output[index * 4 + 3] = tex2DGrad<float4>(suffix, 1.125f, -.125f, sx, sy);
}
#else
#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

bool checked(hipError_t error, const char* operation) {
    if (error == hipSuccess) return true;
    std::cerr << operation << ": " << hipGetErrorString(error) << " (" << int(error) << ")\n";
    return false;
}

float4 authored(unsigned int mip, int x, int y) {
    return make_float4(float(mip) + .125f * x + .0625f * y,
                       .25f * mip - .0625f * x + .125f * y, float(mip), 1.f);
}

struct Resources {
    hipMipmappedArray_t arrays[2]{};
    hipTextureObject_t textures[2]{};
    hipModule_t module = nullptr;
    float4* output = nullptr;
    bool close() {
        if (!checked(hipDeviceSynchronize(), "cleanup synchronize")) return false;
        bool success = true;
        for (unsigned int resource = 0; resource < 2; ++resource) {
            if (textures[resource]) {
                if (!checked(hipDestroyTextureObject(textures[resource]), "destroy sampler")) {
                    success = false;
                    continue;
                }
                textures[resource] = nullptr;
            }
            if (arrays[resource]) {
                if (checked(hipFreeMipmappedArray(arrays[resource]), "free array")) arrays[resource] = nullptr;
                else success = false;
            }
        }
        if (output) {
            if (checked(hipFree(output), "free output")) output = nullptr;
            else success = false;
        }
        if (module) {
            if (checked(hipModuleUnload(module), "unload module")) module = nullptr;
            else success = false;
        }
        return success;
    }
};

bool run(Resources& resources, const char* modulePath, bool& matches) {
    const auto channel = hipCreateChannelDesc<float4>();
    for (unsigned int first = 0; first < 2; ++first) {
        const auto size = Width >> first;
        if (!checked(hipMallocMipmappedArray(&resources.arrays[first], &channel,
            make_hipExtent(size, size, 0), Levels - first), "allocate mipmapped array")) return false;
        for (unsigned int mip = first; mip < Levels; ++mip) {
            const auto side = std::max(1u, Width >> mip);
            std::vector<float4> pixels(side * side), readback(pixels.size());
            for (unsigned int y = 0; y < side; ++y)
                for (unsigned int x = 0; x < side; ++x)
                    pixels[y * side + x] = authored(mip, x, y);
            hipArray_t level{};
            if (!checked(hipGetMipmappedArrayLevel(&level, resources.arrays[first], mip - first),
                         "get level")) return false;
            hipChannelFormatDesc actualChannel{};
            hipExtent actualExtent{};
            unsigned int flags = 0;
            if (!checked(hipArrayGetInfo(&actualChannel, &actualExtent, &flags, level),
                         "array dimensions")) return false;
            if (actualExtent.width != side || actualExtent.height != side) {
                std::cerr << "Unexpected allocated mip dimensions\n";
                return false;
            }
            const auto row = side * sizeof(float4);
            if (!checked(hipMemcpy2DToArray(level, 0, 0, pixels.data(), row, row, side,
                                            hipMemcpyHostToDevice), "upload")) return false;
            if (!checked(hipMemcpy2DFromArray(readback.data(), row, level, 0, 0, row, side,
                                              hipMemcpyDeviceToHost), "readback")) return false;
            if (std::memcmp(pixels.data(), readback.data(), pixels.size() * sizeof(float4))) {
                std::cerr << "Uploaded pixels differ from readback\n";
                return false;
            }
        }
        hipResourceDesc resource{};
        resource.resType = hipResourceTypeMipmappedArray;
        resource.res.mipmap.mipmap = resources.arrays[first];
        hipTextureDesc sampler{};
        sampler.addressMode[0] = sampler.addressMode[1] = hipAddressModeWrap;
        sampler.normalizedCoords = 1;
        sampler.readMode = hipReadModeElementType;
        sampler.filterMode = sampler.mipmapFilterMode = hipFilterModePoint;
        sampler.maxMipmapLevelClamp = float(Levels - first - 1);
        if (!checked(hipCreateTextureObject(&resources.textures[first], &resource, &sampler, nullptr),
                     "create point sampler (legacy anisotropy zero)")) return false;
        hipTextureDesc returned{};
        if (!checked(hipGetTextureObjectTextureDesc(&returned, resources.textures[first]),
                     "sampler readback")) return false;
        if (returned.filterMode != hipFilterModePoint || returned.mipmapFilterMode != hipFilterModePoint ||
            returned.maxAnisotropy != 0 || !returned.normalizedCoords || returned.sRGB) {
            std::cerr << "Unexpected native sampler settings\n";
            return false;
        }
    }
    if (!checked(hipModuleLoad(&resources.module, modulePath), "load module")) return false;
    hipFunction_t kernel{};
    if (!checked(hipModuleGetFunction(&kernel, resources.module, "samplePointGradients"),
                 "find kernel")) return false;
    if (!checked(hipMalloc(&resources.output, Samples * 4 * sizeof(float4)), "allocate results")) return false;
    void* args[]{&resources.textures[0], &resources.textures[1], &resources.output};
    if (!checked(hipModuleLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr),
                 "launch")) return false;
    if (!checked(hipDeviceSynchronize(), "sample synchronize")) return false;
    std::array<float4, Samples * 4> results;
    if (!checked(hipMemcpy(results.data(), resources.output, sizeof(results), hipMemcpyDeviceToHost),
                 "sample results")) return false;
    const float lods[Samples]{.515625f, .75f, 1.f, 1.25f};
    const char* paths[]{"full_lod", "full_grad", "suffix_lod", "suffix_grad"};
    std::cout << "All original mip dimensions/pixels read back exactly.\n"
              << "LOD,path,R,G,B,A,expected_R,expected_G,max_channel_error\n" << std::fixed << std::setprecision(9);
    matches = true;
    float maxError = 0, maxDifference = 0;
    for (unsigned int i = 0; i < Samples; ++i) {
        const auto mip = static_cast<unsigned int>(std::floor(lods[i] + .5f));
        const int side = std::max(1u, Width >> mip);
        const int x = int(std::floor(1.125f * side)) % side;
        const int y = (int(std::floor(-.125f * side)) % side + side) % side;
        const auto expected = authored(mip, x, y);
        for (unsigned int path = 0; path < 4; ++path) {
            const auto p = results[i * 4 + path];
            float error = std::max({std::abs(p.x - expected.x), std::abs(p.y - expected.y),
                                   std::abs(p.z - expected.z), std::abs(p.w - expected.w)});
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || !std::isfinite(p.w))
                error = INFINITY;
            maxError = std::max(maxError, error);
            matches = matches && error <= 1e-6f;
            std::cout << lods[i] << ',' << paths[path] << ',' << p.x << ',' << p.y << ',' << p.z << ','
                      << p.w << ',' << expected.x << ',' << expected.y << ',' << error << '\n';
        }
        const auto a = results[i * 4 + 1], b = results[i * 4 + 3];
        maxDifference = std::max({maxDifference, std::abs(a.x - b.x), std::abs(a.y - b.y),
                                  std::abs(a.z - b.z), std::abs(a.w - b.w)});
    }
    std::cout << "max_analytical_error=" << maxError << " max_full_suffix_gradient_difference="
              << maxDifference << " tolerance=0.000001\n";
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: point_gradient_repro <point_gradient_kernel.co> [device]\n";
        return 2;
    }
    int device = 0;
    if (argc == 3) {
        const char* end = argv[2] + std::strlen(argv[2]);
        const auto parsed = std::from_chars(argv[2], end, device);
        if (parsed.ec != std::errc{} || parsed.ptr != end || device < 0) {
            std::cerr << "Invalid device ordinal\n";
            return 2;
        }
    }
    if (!checked(hipSetDevice(device), "select device")) return 2;
    hipDeviceProp_t properties{};
    int runtime = 0, driver = 0;
    if (!checked(hipGetDeviceProperties(&properties, device), "device properties") ||
        !checked(hipRuntimeGetVersion(&runtime), "runtime version") ||
        !checked(hipDriverGetVersion(&driver), "driver API version")) return 2;
    std::cout << properties.name << ' ' << properties.gcnArchName << " device=" << device
              << " HIP headers=" << HIP_VERSION << " runtime=" << runtime << " driver_API=" << driver << '\n';
    Resources resources;
    bool matches = false;
    const bool ran = run(resources, argv[1], matches);
    const bool cleaned = resources.close();
    return !ran || !cleaned ? 2 : (matches ? 0 : 1);
}
#endif
