#include <hip/hip_runtime.h>
#include "DemandLoading/DemandTextureLoader.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

// Minimal module loader helper
struct KernelModule {
    hipModule_t module{nullptr};
    hipFunction_t kernel{nullptr};
    ~KernelModule() { if (module) hipModuleUnload(module); }

    bool load(const char* path, const char* name) {
        if (hipModuleLoad(&module, path) != hipSuccess) return false;
        if (hipModuleGetFunction(&kernel, module, name) != hipSuccess) return false;
        return true;
    }
};

static void generateTextureData(std::vector<uint8_t>& data, int size, int seed) {
    data.resize(size * size * 4);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int idx = (y * size + x) * 4;
            data[idx + 0] = static_cast<uint8_t>((seed * 31 + x) % 256);
            data[idx + 1] = static_cast<uint8_t>((y * 255) / size);
            data[idx + 2] = static_cast<uint8_t>((x * 255) / size);
            data[idx + 3] = 255;
        }
    }
}

int main() {
    namespace fs = std::filesystem;

    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No HIP devices found\n";
        return 1;
    }

    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << "\n";

    KernelModule module;
    if (!module.load("render_kernel.co", "renderKernelWrapper")) {
        std::cerr << "Failed to load render_kernel.co\n";
        return 1;
    }

    auto runScenario = [&](const char* label, bool useAsync) {
        std::cout << "\n=== " << label << " ===\n";

        hip_demand::LoaderOptions options;
        options.maxTextureMemory = 512 * 1024 * 1024;
        options.maxTextures = 256;
        options.maxRequestsPerLaunch = 1920 * 1080;
        options.enableEviction = true;

        hip_demand::DemandTextureLoader loader(options);

        // Build textures
        const int numTexturesHost = 12;
        std::vector<uint32_t> textureIds;
        textureIds.reserve(numTexturesHost);
        for (int i = 0; i < numTexturesHost; ++i) {
            int size = 512 + i * 64;
            std::vector<uint8_t> data;
            generateTextureData(data, size, i);
            hip_demand::TextureDesc desc;
            desc.addressMode[0] = hipAddressModeWrap;
            desc.addressMode[1] = hipAddressModeWrap;
            desc.filterMode = hipFilterModeLinear;
            desc.generateMipmaps = true;
            auto handle = loader.createTextureFromMemory(data.data(), size, size, 4, desc);
            if (handle.valid) {
                textureIds.push_back(handle.id);
            }
        }

        const int width = 1920;
        const int height = 1080;
        float4* d_output = nullptr;
        hipMalloc(&d_output, width * height * sizeof(float4));

        uint32_t* d_textureIds = nullptr;
        hipMalloc(&d_textureIds, textureIds.size() * sizeof(uint32_t));
        hipMemcpy(d_textureIds, textureIds.data(), textureIds.size() * sizeof(uint32_t), hipMemcpyHostToDevice);

        hipStream_t stream;
        hipStreamCreate(&stream);

        int maxPasses = 6;
        size_t totalLoaded = 0;
        auto start = std::chrono::high_resolution_clock::now();

        for (int pass = 0; pass < maxPasses; ++pass) {
            loader.launchPrepare(stream);
            auto ctx = loader.getDeviceContext();

            float time = pass * 0.1f;
            int numTextures = static_cast<int>(textureIds.size());
            int k_width = width;
            int k_height = height;
            void* args[] = { &ctx, &d_output, &k_width, &k_height, &d_textureIds, &numTextures, &time };

            dim3 blockSize(16, 16);
            dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                          (height + blockSize.y - 1) / blockSize.y);

            hipError_t err = hipModuleLaunchKernel(
                module.kernel,
                gridSize.x, gridSize.y, gridSize.z,
                blockSize.x, blockSize.y, blockSize.z,
                0,
                stream,
                args,
                nullptr);
            if (err != hipSuccess) {
                std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << "\n";
                break;
            }

            hipStreamSynchronize(stream);

            size_t loaded = 0;
            if (useAsync) {
                auto ticket = loader.processRequestsAsync(stream, ctx);
                ticket.wait();
                loaded = loader.getRequestCount();
            } else {
                loaded = loader.processRequests(stream, ctx);
            }
            totalLoaded += loaded;

            std::cout << "Pass " << pass << ": " << loaded
                      << (useAsync ? " requests processed" : " textures loaded")
                      << ", resident=" << loader.getResidentTextureCount()
                      << " mem=" << (loader.getTotalTextureMemory() / (1024 * 1024)) << "MB";
            if (loader.hadRequestOverflow()) std::cout << " (overflow)";
            std::cout << "\n";

            if (loaded == 0 && pass > 1) break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Summary (" << label << "): loaded=" << totalLoaded
                  << " resident=" << loader.getResidentTextureCount()
                  << " mem=" << (loader.getTotalTextureMemory() / (1024 * 1024)) << "MB"
                  << " time=" << ms << " ms\n";

        hipFree(d_output);
        hipFree(d_textureIds);
        hipStreamDestroy(stream);
    };

    runScenario("Sync", false);
    runScenario("Async", true);

    return 0;
}
