#include "DemandLoading/DemandTextureLoader.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <filesystem>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Demo goal: churn many small textures in a mosaic to stress paging and eviction under a low memory budget.
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

static void generateSmallTexture(const char* filename, int size, int id) {
    std::vector<uint8_t> data(size * size * 4);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int idx = (y * size + x) * 4;
            float fx = static_cast<float>(x) / static_cast<float>(size);
            float fy = static_cast<float>(y) / static_cast<float>(size);
            float stripe = 0.5f + 0.5f * std::sin(6.28318f * (fx * 4.0f + fy * 3.0f + id * 0.37f));
            data[idx + 0] = static_cast<uint8_t>(255.0f * stripe);
            data[idx + 1] = static_cast<uint8_t>((id * 53) % 256);
            data[idx + 2] = static_cast<uint8_t>(255.0f * fy);
            data[idx + 3] = 255;
        }
    }
    stbi_write_png(filename, size, size, 4, data.data(), size * 4);
    std::cout << "  Generated texture: " << filename << " (" << size << "x" << size << ")\n";
}

int main() {
    namespace fs = std::filesystem;
    std::cout << "Atlas churn example\n";

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
    if (!module.load("atlas_render_kernel.co", "renderKernelWrapper")) {
        return 1;
    }

    hip_demand::LoaderOptions options;
    options.maxTextureMemory = 96 * 1024 * 1024; // low budget to force eviction
    options.maxTextures = 512;
    options.maxRequestsPerLaunch = 1920 * 1080;
    options.enableEviction = true;

    hip_demand::DemandTextureLoader loader(options);

    const int texSize = 512;
    const int numTextures = 128; // ~128 MB raw; with mipmaps exceeds budget to force churn

    std::vector<uint32_t> textureIds;
    hip_demand::TextureDesc desc;
    desc.addressMode = hipAddressModeWrap;
    desc.filterMode = hipFilterModeLinear;
    desc.generateMipmaps = true;

    std::cout << "Preparing textures...\n";
    for (int i = 0; i < numTextures; ++i) {
        char filename[64];
        sprintf(filename, "atlas_tex_%03d.png", i);

        hip_demand::TextureHandle handle;
        if (fs::exists(filename)) {
            handle = loader.createTexture(filename, desc);
            if (handle.valid) {
                std::cout << "  Loaded existing: " << filename << " (" << handle.width << "x" << handle.height << ")\n";
            }
        } else {
            generateSmallTexture(filename, texSize, i);
            handle = loader.createTexture(filename, desc);
        }

        if (!handle.valid) {
            std::cerr << "Failed to create texture: " << filename << ": "
                      << hip_demand::getErrorString(handle.error) << "\n";
            return 1;
        }
        textureIds.push_back(handle.id);
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

    int maxPasses = 16;

    for (int pass = 0; pass < maxPasses; ++pass) {
        loader.launchPrepare(stream);
        auto ctx = loader.getDeviceContext();

        int numTex = static_cast<int>(textureIds.size());
        int k_width = width;
        int k_height = height;
        int k_pass = pass;

        void* args[] = { &ctx, &d_output, &k_width, &k_height, &d_textureIds, &numTex, &k_pass };

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

        size_t loaded = loader.processRequests(stream);
        std::cout << "Pass " << (pass + 1) << ": " << loaded << " loaded, resident="
                  << loader.getResidentTextureCount() << " mem="
                  << (loader.getTotalTextureMemory() / (1024*1024)) << "MB";
        if (loader.hadRequestOverflow()) std::cout << " (overflow)";
        std::cout << "\n";
    }

    std::vector<float4> h_output(width * height);
    hipMemcpy(h_output.data(), d_output, width * height * sizeof(float4), hipMemcpyDeviceToHost);

    std::vector<uint8_t> output_rgb(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        output_rgb[i*3 + 0] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, h_output[i].x * 255.0f)));
        output_rgb[i*3 + 1] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, h_output[i].y * 255.0f)));
        output_rgb[i*3 + 2] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, h_output[i].z * 255.0f)));
    }
    stbi_write_png("output_atlas.png", width, height, 3, output_rgb.data(), width * 3);
    std::cout << "Saved output_atlas.png\n";

    hipFree(d_output);
    hipFree(d_textureIds);
    hipStreamDestroy(stream);
    return 0;
}
