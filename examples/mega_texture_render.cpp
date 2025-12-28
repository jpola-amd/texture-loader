#include "DemandLoading/DemandTextureLoader.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <filesystem>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Demo goal: stress demand loading by flying over multiple 8K textures under a tight memory budget,
// forcing paging and eviction while cycling textures each pass.
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

static void generateMegaTexture(const char* filename, int size, float hueShift) {
    std::vector<uint8_t> data(size * size * 4);
    for (int y = 0; y < size; ++y) {
        float fy = static_cast<float>(y) / static_cast<float>(size);
        for (int x = 0; x < size; ++x) {
            float fx = static_cast<float>(x) / static_cast<float>(size);
            int idx = (y * size + x) * 4;
            // Rotate and phase-shift per texture so A/B/C look different
            float angle = hueShift * 2.3f;
            float ca = std::cos(angle);
            float sa = std::sin(angle);
            float rx = fx * ca - fy * sa;
            float ry = fx * sa + fy * ca;

            float swirl = 0.5f + 0.5f * std::sin(6.28318f * (rx * 3.0f + ry * 1.5f) + hueShift * 3.7f);
            float grad = 0.4f + 0.6f * fx;
            float band = 0.5f + 0.5f * std::cos(6.28318f * (ry * 2.4f + rx * 1.1f) + hueShift * 5.1f);
            float radial = std::sqrt((fx - 0.5f) * (fx - 0.5f) + (fy - 0.5f) * (fy - 0.5f));
            float vignette = 1.0f - std::min(radial * 1.4f, 1.0f);

            float r = 0.55f * swirl + 0.45f * vignette;
            float g = 0.5f * band + 0.5f * grad;
            float b = 0.6f * (1.0f - band * 0.5f) + 0.4f * std::sin(6.28318f * (radial * 3.0f + hueShift));

            r = fminf(fmaxf(r, 0.0f), 1.0f);
            g = fminf(fmaxf(g, 0.0f), 1.0f);
            b = fminf(fmaxf(b, 0.0f), 1.0f);

            data[idx + 0] = static_cast<uint8_t>(255.0f * r);
            data[idx + 1] = static_cast<uint8_t>(255.0f * g);
            data[idx + 2] = static_cast<uint8_t>(255.0f * b);
            data[idx + 3] = 255;
        }
    }
    stbi_write_png(filename, size, size, 4, data.data(), size * 4);
    std::cout << "  Generated mega texture: " << filename << " (" << size << "x" << size << ")\n";
}

int main() {
    namespace fs = std::filesystem;
    std::cout << "Mega-texture flythrough example\n";

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
    if (!module.load("mega_texture_kernel.co", "renderKernelWrapper")) {
        return 1;
    }

    hip_demand::LoaderOptions options;
    options.maxTextureMemory = 256 * 1024 * 1024; // tighter budget to force paging
    options.maxTextures = 64;
    options.maxRequestsPerLaunch = 1920 * 1080;
    options.enableEviction = true;

    hip_demand::DemandTextureLoader loader(options);

    const int texSize = 8192;
    std::vector<std::string> filenames = {
        "mega_texture_A.png",
        "mega_texture_B.png",
        "mega_texture_C.png"
    };

    std::vector<hip_demand::TextureHandle> handles;
    hip_demand::TextureDesc desc;
    desc.addressMode[0] = hipAddressModeWrap;
    desc.addressMode[1] = hipAddressModeWrap;
    desc.filterMode = hipFilterModeLinear;
    desc.generateMipmaps = true;

    for (size_t i = 0; i < filenames.size(); ++i) {
        const char* filename = filenames[i].c_str();
        hip_demand::TextureHandle handle;
        if (fs::exists(filename)) {
            handle = loader.createTexture(filename, desc);
            if (handle.valid) {
                std::cout << "  Loaded existing mega texture: " << filename << " (" << handle.width << "x" << handle.height << ")\n";
            }
        } else {
            std::cout << "Generating mega texture (" << filename << ")...\n";
            // Use non-integer phase offsets so textures A/B/C are visually distinct
            float hueShift = static_cast<float>(i) * 0.37f;
            generateMegaTexture(filename, texSize, hueShift);
            handle = loader.createTexture(filename, desc);
        }

        if (!handle.valid) {
            std::cerr << "Failed to create mega texture: " << filename << ": "
                      << hip_demand::getErrorString(handle.error) << "\n";
            return 1;
        }
        handles.push_back(handle);
    }

    const int width = 1920;
    const int height = 1080;
    float4* d_output = nullptr;
    hipMalloc(&d_output, width * height * sizeof(float4));

    hipStream_t stream;
    hipStreamCreate(&stream);

    int maxPasses = 12;
    size_t totalLoaded = 0;

    for (int pass = 0; pass < maxPasses; ++pass) {
        loader.launchPrepare(stream);
        auto ctx = loader.getDeviceContext();
        
        float time = pass * 0.3f;
        float zoom = 1.6f;
        uint32_t texId = handles[pass % handles.size()].id;
        int k_width = width;
        int k_height = height;

        void* args[] = { &ctx, &d_output, &k_width, &k_height, &texId, &time, &zoom };

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

        auto ticket = loader.processRequestsAsync(stream, ctx);
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
    stbi_write_png("output_mega.png", width, height, 3, output_rgb.data(), width * 3);
    std::cout << "Saved output_mega.png\n";

    hipFree(d_output);
    hipStreamDestroy(stream);
    return 0;
}
