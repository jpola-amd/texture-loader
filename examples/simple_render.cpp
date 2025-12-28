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

// Load HIP kernel module and get kernel function
struct KernelModule {
    hipModule_t module;
    hipFunction_t kernel;
    
    KernelModule() : module(nullptr), kernel(nullptr) {}
    
    ~KernelModule() {
        if (module) {
            hipModuleUnload(module);
        }
    }
    
    bool load(const char* modulePath, const char* kernelName) {
        // Load the compiled kernel code object
        hipError_t err = hipModuleLoad(&module, modulePath);
        if (err != hipSuccess) {
            std::cerr << "Failed to load HIP module: " << hipGetErrorString(err) << "\n";
            return false;
        }
        
        // Get the kernel function
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

// Generate a procedural checkerboard pattern
void generateCheckerboard(const char* filename, int size) {
    std::vector<uint8_t> data(size * size * 4);
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int checker = ((x / 64) + (y / 64)) % 2;
            uint8_t val = checker ? 255 : 64;
            int idx = (y * size + x) * 4;
            data[idx + 0] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
            data[idx + 3] = 255;
        }
    }
    // In real code, save to file. For now, just demonstrate in-memory
    printf("Generated checkerboard: %s (%dx%d)\n", filename, size, size);
}

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    std::cout << "HIP Demand Texture Loader Example\n";
    std::cout << "==================================\n\n";
    
    // Initialize HIP
    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No HIP devices found!\n";
        return 1;
    }
    
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << "\n\n";
    
    // Load kernel module
    std::cout << "Loading HIP kernel module...\n";
    KernelModule kernelModule;
    if (!kernelModule.load("render_kernel.co", "renderKernelWrapper")) {
        std::cerr << "Failed to load kernel module. Make sure render_kernel.co is in the current directory.\n";
        return 1;
    }
    std::cout << "Kernel module loaded successfully\n\n";
    
    // Create demand texture loader
    hip_demand::LoaderOptions options;
    options.maxTextureMemory = 512 * 1024 * 1024;  // 512 MB
    options.maxTextures = 256;
    options.maxRequestsPerLaunch = 1920 * 1080;  // One request per pixel max
    options.enableEviction = true;
    
    hip_demand::DemandTextureLoader loader(options);
    
    // Create multiple textures
    std::cout << "Creating textures...\n";
    std::vector<uint32_t> textureIds;
    
    // Generate test textures in memory
    for (int i = 0; i < 16; i++) {
        int size = 512 + i * 128;
        char filename[256];
        sprintf(filename, "texture_%02d.png", i);
        hip_demand::TextureHandle handle;

        // Reuse existing textures if present in the current directory
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
            // Generate unique pattern for each texture
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    int idx = (y * size + x) * 4;
                    data[idx + 0] = (i * 16) % 256;          // R varies by texture
                    data[idx + 1] = (x * 255) / size;        // G varies by x
                    data[idx + 2] = (y * 255) / size;        // B varies by y
                    data[idx + 3] = 255;
                }
            }

            stbi_write_png(filename, size, size, 4, data.data(), size * 4);
            std::cout << "  Saved: " << filename << " (" << size << "x" << size << ")\n";

            hip_demand::TextureDesc desc;
            desc.addressMode[0] = hipAddressModeWrap;
            desc.addressMode[1] = hipAddressModeWrap;
            desc.filterMode = hipFilterModeLinear;
            desc.generateMipmaps = true;  // Enable mipmaps for better quality

            handle = loader.createTextureFromMemory(
                data.data(), size, size, 4, desc);
        }
        
        if (handle.valid) {
            textureIds.push_back(handle.id);
            std::cout << "  Texture " << handle.id << ": " 
                     << handle.width << "x" << handle.height;
            if (handle.error != hip_demand::LoaderError::Success) {
                std::cout << " (Warning: " << hip_demand::getErrorString(handle.error) << ")";
            }
            std::cout << "\n";
        } else {
            std::cerr << "  Failed to create texture " << i << ": " 
                     << hip_demand::getErrorString(handle.error) << "\n";
        }
    }
    
    std::cout << "\nCreated " << textureIds.size() << " textures\n\n";
    
    // Allocate output buffer
    const int width = 1920;
    const int height = 1080;
    float4* d_output = nullptr;
    hipMalloc(&d_output, width * height * sizeof(float4));
    
    // Copy texture IDs to device
    uint32_t* d_textureIds = nullptr;
    hipMalloc(&d_textureIds, textureIds.size() * sizeof(uint32_t));
    hipMemcpy(d_textureIds, textureIds.data(), 
             textureIds.size() * sizeof(uint32_t),
             hipMemcpyHostToDevice);
    
    // Multi-pass rendering loop
    std::cout << "Starting multi-pass rendering...\n";
    
    hipStream_t stream;
    hipStreamCreate(&stream);
    
    int maxPasses = 10;
    int pass = 0;
    size_t totalLoaded = 0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    while (pass < maxPasses) {
        pass++;
        
        // Prepare for launch
        loader.launchPrepare(stream);
        
        // Get device context
        auto ctx = loader.getDeviceContext();
        
        // Launch render kernel using module API
        float time = pass * 0.1f;
        int numTextures = static_cast<int>(textureIds.size());
        
        int k_width = width;
        int k_height = height;
        // Set up kernel arguments
        void* args[] = {
            &ctx,
            &d_output,
            &k_width,
            &k_height,
            &d_textureIds,
            &numTextures,
            &time
        };
        
        // Calculate grid and block dimensions
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
        
        // Launch kernel via module API
        hipError_t err = hipModuleLaunchKernel(
            kernelModule.kernel,
            gridSize.x, gridSize.y, gridSize.z,
            blockSize.x, blockSize.y, blockSize.z,
            0,  // shared memory
            stream,
            args,
            nullptr  // extra
        );
        
        if (err != hipSuccess) {
            std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << "\n";
            break;
        }
        
        // Wait for kernel to complete
        hipStreamSynchronize(stream);
        
        // Process texture requests
        size_t loaded = loader.processRequests(stream);
        totalLoaded += loaded;
        
        std::cout << "Pass " << pass << ": " 
                 << loaded << " textures loaded, "
                 << loader.getResidentTextureCount() << " resident, "
                 << (loader.getTotalTextureMemory() / (1024*1024)) << " MB used";
        
        if (loader.hadRequestOverflow()) {
            std::cout << " (WARNING: Request buffer overflow!)";
        }
        std::cout << "\n";
        
        // If no new textures were loaded, we're done
        if (loaded == 0 && pass > 1) {
            std::cout << "\nAll required textures resident. Rendering complete.\n";
            break;
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    std::cout << "\nStatistics:\n";
    std::cout << "  Total passes: " << pass << "\n";
    std::cout << "  Total textures loaded: " << totalLoaded << "\n";
    std::cout << "  Resident textures: " << loader.getResidentTextureCount() << "\n";
    std::cout << "  Memory used: " << (loader.getTotalTextureMemory() / (1024*1024)) << " MB\n";
    std::cout << "  Memory limit: " << (loader.getMaxTextureMemory() / (1024*1024)) << " MB\n";
    std::cout << "  Total time: " << duration.count() << " ms\n";
    
    if (loader.getLastError() != hip_demand::LoaderError::Success) {
        std::cout << "  Last error: " << hip_demand::getErrorString(loader.getLastError()) << "\n";
    }
    
    // Optional: Download and save output
    std::vector<float4> h_output(width * height);
    hipMemcpy(h_output.data(), d_output, 
             width * height * sizeof(float4),
             hipMemcpyDeviceToHost);
    
    std::cout << "\nSample output pixel (center): ("
             << h_output[height/2 * width + width/2].x << ", "
             << h_output[height/2 * width + width/2].y << ", "
             << h_output[height/2 * width + width/2].z << ", "
             << h_output[height/2 * width + width/2].w << ")\n";
    
    // Convert float4 to uint8_t and save final render
    std::cout << "\nSaving final render to output.png...\n";
    std::vector<uint8_t> output_rgb(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        output_rgb[i*3 + 0] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, h_output[i].x * 255.0f)));
        output_rgb[i*3 + 1] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, h_output[i].y * 255.0f)));
        output_rgb[i*3 + 2] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, h_output[i].z * 255.0f)));
    }
    stbi_write_png("output.png", width, height, 3, output_rgb.data(), width * 3);
    std::cout << "Saved output.png (" << width << "x" << height << ")\n";
    
    // Cleanup
    hipFree(d_output);
    hipFree(d_textureIds);
    hipStreamDestroy(stream);
    
    std::cout << "\nDone!\n";
    
    return 0;
}
