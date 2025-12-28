#include "DemandLoading/DemandTextureLoader.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <filesystem>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct KernelModule {
    hipModule_t module{nullptr};
    hipFunction_t kernel{nullptr};
    ~KernelModule() { if (module) hipModuleUnload(module); }

    bool load(const char* modulePath, const char* kernelName) {
        hipError_t err = hipModuleLoad(&module, modulePath);
        if (err != hipSuccess) {
            std::cerr << "Failed to load HIP module: " << hipGetErrorString(err) << "\n";
            return false;
        }
        err = hipModuleGetFunction(&kernel, module, kernelName);
        if (err != hipSuccess) {
            std::cerr << "Failed to get kernel function: " << hipGetErrorString(err) << "\n";
            hipModuleUnload(module);
            module = nullptr;
            return false;
        }
        return true;
    }
};

int main() {
    namespace fs = std::filesystem;
    std::cout << "Tiled render example\n";

    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No HIP devices found!\n";
        return 1;
    }

    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << "\n";

    KernelModule module;
    if (!module.load("tiled_render_kernel.co", "renderKernelWrapper")) {
        return 1;
    }

    hip_demand::LoaderOptions options;
    options.maxTextureMemory = 512 * 1024 * 1024;
    options.maxTextures = 256;
    options.maxRequestsPerLaunch = 1920 * 1080;
    options.enableEviction = true;

    hip_demand::DemandTextureLoader loader(options);

    std::vector<uint32_t> textureIds;
    std::cout << "Creating textures...\n";
    for (int i = 0; i < 16; ++i) {
        int size = 512 + i * 128;
        char filename[64];
        sprintf(filename, "texture_%02d.png", i);

        hip_demand::TextureHandle handle;

        if (fs::exists(filename)) {
            hip_demand::TextureDesc desc;
            desc.addressMode[0] = hipAddressModeWrap;
            desc.addressMode[1] = hipAddressModeWrap;
            desc.filterMode = hipFilterModeLinear;
            desc.generateMipmaps = true;
            handle = loader.createTexture(filename, desc);
            if (handle.valid) {
                std::cout << "  Loaded existing: " << filename << " (" << handle.width << "x" << handle.height << ")\n";
            }
        } else {
            std::vector<uint8_t> data(size * size * 4);
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    int idx = (y * size + x) * 4;
                    data[idx + 0] = static_cast<uint8_t>((i * 16) % 256);
                    data[idx + 1] = static_cast<uint8_t>((x * 255) / size);
                    data[idx + 2] = static_cast<uint8_t>((y * 255) / size);
                    data[idx + 3] = 255;
                }
            }
            stbi_write_png(filename, size, size, 4, data.data(), size * 4);

            hip_demand::TextureDesc desc;
            desc.addressMode[0] = hipAddressModeWrap;
            desc.addressMode[1] = hipAddressModeWrap;
            desc.filterMode = hipFilterModeLinear;
            desc.generateMipmaps = true;

            handle = loader.createTextureFromMemory(data.data(), size, size, 4, desc);
        }

        if (handle.valid) {
            textureIds.push_back(handle.id);
            std::cout << "  Texture " << handle.id << " (" << handle.width << "x" << handle.height << ")\n";
        } else {
            std::cerr << "  Failed to create texture " << i << ": "
                      << hip_demand::getErrorString(handle.error) << "\n";
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

    int maxPasses = 8;
    size_t totalLoaded = 0;

    for (int pass = 0; pass < maxPasses; ++pass) {
        loader.launchPrepare(stream);
        auto ctx = loader.getDeviceContext();

        float time = pass * 0.15f;
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

        auto ticket = loader.processRequestsAsync(stream);
        ticket.wait();
        size_t loaded = loader.getRequestCount();
        totalLoaded += loaded;
        std::cout << "Pass " << (pass + 1) << ": " << loaded << " requests processed, resident="
                  << loader.getResidentTextureCount() << " mem="
                  << (loader.getTotalTextureMemory() / (1024*1024)) << "MB";
        if (loader.hadRequestOverflow()) std::cout << " (overflow)";
        std::cout << "\n";

        if (loaded == 0 && pass > 0) {
            break;
        }
    }

    std::vector<float4> h_output(width * height);
    hipMemcpy(h_output.data(), d_output, width * height * sizeof(float4), hipMemcpyDeviceToHost);

    std::vector<uint8_t> output_rgb(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        output_rgb[i*3 + 0] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, h_output[i].x * 255.0f)));
        output_rgb[i*3 + 1] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, h_output[i].y * 255.0f)));
        output_rgb[i*3 + 2] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, h_output[i].z * 255.0f)));
    }
    stbi_write_png("output_tiled.png", width, height, 3, output_rgb.data(), width * 3);
    std::cout << "Saved output_tiled.png\n";

    hipFree(d_output);
    hipFree(d_textureIds);
    hipStreamDestroy(stream);
    return 0;
}
