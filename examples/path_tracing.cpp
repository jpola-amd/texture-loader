// Path Tracing Example for HIP Demand Texture Loading
// Demonstrates non-coherent texture access patterns typical of ray tracing.
// Rays bounce through a Cornell box-style scene, accessing textures based on
// where they hit surfaces - stressing the demand loading system with scattered access.

#include "DemandLoading/DemandTextureLoader.h"
#include "DemandLoading/Logging.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <cstring>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

struct KernelModule {
    hipModule_t module{nullptr};
    hipFunction_t kernel{nullptr};
    
    ~KernelModule() { 
        if (module) hipModuleUnload(module); 
    }

    bool load(const char* modulePath, const char* kernelName) {
        hipError_t err = hipModuleLoad(&module, modulePath);
        if (err != hipSuccess) {
            std::cerr << "Failed to load HIP module '" << modulePath << "': " << hipGetErrorString(err) << "\n";
            return false;
        }
        err = hipModuleGetFunction(&kernel, module, kernelName);
        if (err != hipSuccess) {
            std::cerr << "Failed to get kernel function '" << kernelName << "': " << hipGetErrorString(err) << "\n";
            hipModuleUnload(module);
            module = nullptr;
            return false;
        }
        return true;
    }
};

// Generate procedural textures for materials
static void generateMaterialTexture(const std::string& filename, int size, int materialType) {
    std::vector<uint8_t> data(size * size * 4);
    
    for (int y = 0; y < size; ++y) {
        float fy = static_cast<float>(y) / static_cast<float>(size);
        for (int x = 0; x < size; ++x) {
            float fx = static_cast<float>(x) / static_cast<float>(size);
            int idx = (y * size + x) * 4;
            
            float r, g, b;
            
            switch (materialType % 9) {
                case 0: // Marble-like
                    {
                        float noise = 0.5f + 0.5f * sinf(fx * 20.0f + sinf(fy * 15.0f) * 2.0f);
                        r = 0.9f * noise + 0.1f;
                        g = 0.85f * noise + 0.1f;
                        b = 0.8f * noise + 0.15f;
                    }
                    break;
                case 1: // Wood grain
                    {
                        float ring = sinf(sqrtf((fx - 0.5f) * (fx - 0.5f) + (fy - 0.5f) * (fy - 0.5f)) * 40.0f);
                        r = 0.6f + 0.2f * ring;
                        g = 0.4f + 0.15f * ring;
                        b = 0.2f + 0.1f * ring;
                    }
                    break;
                case 2: // Blue tiles
                    {
                        int tx = static_cast<int>(fx * 8.0f) % 2;
                        int ty = static_cast<int>(fy * 8.0f) % 2;
                        float tile = (tx ^ ty) ? 0.8f : 0.6f;
                        r = 0.2f * tile;
                        g = 0.4f * tile;
                        b = 0.9f * tile;
                    }
                    break;
                case 3: // Orange emissive (for light sphere)
                    {
                        float glow = 1.0f - sqrtf((fx - 0.5f) * (fx - 0.5f) + (fy - 0.5f) * (fy - 0.5f)) * 1.5f;
                        glow = fmaxf(glow, 0.3f);
                        r = 1.0f * glow;
                        g = 0.8f * glow;
                        b = 0.5f * glow;
                    }
                    break;
                case 4: // Checkerboard floor
                    {
                        int cx = static_cast<int>(fx * 16.0f) % 2;
                        int cy = static_cast<int>(fy * 16.0f) % 2;
                        float check = (cx ^ cy) ? 0.85f : 0.15f;
                        r = check;
                        g = check;
                        b = check;
                    }
                    break;
                case 5: // Brick wall
                    {
                        float bx = fmodf(fx * 4.0f, 1.0f);
                        float by = fmodf(fy * 8.0f, 1.0f);
                        int row = static_cast<int>(fy * 8.0f);
                        if (row % 2 == 1) bx = fmodf(bx + 0.5f, 1.0f);
                        
                        float mortar = (bx < 0.05f || bx > 0.95f || by < 0.08f) ? 0.7f : 0.0f;
                        r = 0.6f + mortar * 0.3f + 0.1f * sinf(fx * 50.0f + fy * 30.0f);
                        g = 0.25f + mortar * 0.5f;
                        b = 0.2f + mortar * 0.5f;
                    }
                    break;
                case 6: // Red wall (Cornell box left)
                    {
                        r = 0.7f + 0.1f * sinf(fx * 30.0f) * sinf(fy * 30.0f);
                        g = 0.15f;
                        b = 0.15f;
                    }
                    break;
                case 7: // Green wall (Cornell box right)
                    {
                        r = 0.15f;
                        g = 0.7f + 0.1f * sinf(fx * 30.0f) * sinf(fy * 30.0f);
                        b = 0.15f;
                    }
                    break;
                case 8: // Ceiling (mostly white, light area in center)
                    {
                        float dist = sqrtf((fx - 0.5f) * (fx - 0.5f) + (fy - 0.5f) * (fy - 0.5f));
                        float light = (dist < 0.25f) ? 1.0f : 0.85f;
                        r = light;
                        g = light * 0.95f;
                        b = light * 0.9f;
                    }
                    break;
            }
            
            data[idx + 0] = static_cast<uint8_t>(fminf(r, 1.0f) * 255.0f);
            data[idx + 1] = static_cast<uint8_t>(fminf(g, 1.0f) * 255.0f);
            data[idx + 2] = static_cast<uint8_t>(fminf(b, 1.0f) * 255.0f);
            data[idx + 3] = 255;
        }
    }
    
    stbi_write_png(filename.c_str(), size, size, 4, data.data(), size * 4);
    std::cout << "  Generated material texture: " << filename << " (" << size << "x" << size << ")\n";
}

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -w <width>      Output width (default: 800)\n";
    std::cout << "  -h <height>     Output height (default: 600)\n";
    std::cout << "  -s <samples>    Samples per pixel per frame (default: 4)\n";
    std::cout << "  -f <frames>     Number of frames to render (default: 64)\n";
    std::cout << "  -b <bounces>    Max ray bounces (default: 4)\n";
    std::cout << "  -i <interval>   Save interval in frames (default: 8)\n";
    std::cout << "  --help          Show this help\n";
}

int main(int argc, char** argv) {
    // Create output subfolder
    const std::string outputDir = "path_tracing_output";
    fs::create_directories(outputDir);
    
    std::cout << "Path Tracing Example - HIP Demand Texture Loading\n";
    std::cout << "==================================================\n";
    std::cout << "Demonstrates non-coherent texture access from ray bounces\n\n";

    // Parse command line arguments
    int width = 800;
    int height = 600;
    int samplesPerPixel = 4;
    int numFrames = 64;
    int maxBounces = 4;
    int saveInterval = 8;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 && i + 1 < argc) {
            height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            samplesPerPixel = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            numFrames = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            maxBounces = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            saveInterval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    std::cout << "Settings:\n";
    std::cout << "  Resolution: " << width << "x" << height << "\n";
    std::cout << "  Samples/pixel/frame: " << samplesPerPixel << "\n";
    std::cout << "  Total frames: " << numFrames << "\n";
    std::cout << "  Max bounces: " << maxBounces << "\n";
    std::cout << "  Save interval: " << saveInterval << " frames\n\n";

    // Check for HIP devices
    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No HIP devices found!\n";
        return 1;
    }

    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << "\n\n";

    // Load kernel module
    KernelModule module;
    if (!module.load("path_tracing_kernel.co", "pathTraceKernelWrapper")) {
        return 1;
    }

    // Configure demand texture loader
    // Use a smaller memory budget to force paging and demonstrate demand loading
    hip_demand::LoaderOptions options;
    options.maxTextureMemory = 64 * 1024 * 1024;  // 64 MB - tight budget
    options.maxTextures = 64;
    options.maxRequestsPerLaunch = width * height;  // One request per pixel max
    options.enableEviction = true;

    hip_demand::setLogLevel(hip_demand::LogLevel::Info);
    hip_demand::DemandTextureLoader loader(options);

    // Generate material textures (9 different materials)
    const int numMaterials = 9;
    const int texSize = 1024;  // 1K textures
    
    std::cout << "Preparing material textures...\n";
    std::vector<uint32_t> textureIds;
    
    hip_demand::TextureDesc desc;
    desc.addressMode[0] = hipAddressModeWrap;
    desc.addressMode[1] = hipAddressModeWrap;
    desc.filterMode = hipFilterModeLinear;
    desc.generateMipmaps = true;
    
    for (int i = 0; i < numMaterials; ++i) {
        char filename[256];
        sprintf(filename, "%s/material_%02d.png", outputDir.c_str(), i);
        
        hip_demand::TextureHandle handle;
        
        if (fs::exists(filename)) {
            handle = loader.createTexture(filename, desc);
            if (handle.valid) {
                std::cout << "  Loaded existing: " << filename << "\n";
            }
        } else {
            generateMaterialTexture(filename, texSize, i);
            handle = loader.createTexture(filename, desc);
        }
        
        if (!handle.valid) {
            std::cerr << "Failed to create texture: " << filename << ": "
                      << hip_demand::getErrorString(handle.error) << "\n";
            return 1;
        }
        
        textureIds.push_back(handle.id);
    }
    
    std::cout << "\nCreated " << textureIds.size() << " material textures\n\n";

    // Create HIP stream
    hipStream_t stream;
    hipStreamCreate(&stream);

    // Allocate GPU buffers
    float4* d_output = nullptr;
    float4* d_accumulator = nullptr;
    uint32_t* d_textureIds = nullptr;
    
    hipMalloc(&d_output, width * height * sizeof(float4));
    hipMalloc(&d_accumulator, width * height * sizeof(float4));
    hipMalloc(&d_textureIds, textureIds.size() * sizeof(uint32_t));
    
    // Initialize accumulator to zero
    hipMemset(d_accumulator, 0, width * height * sizeof(float4));
    
    // Copy texture IDs to device
    hipMemcpy(d_textureIds, textureIds.data(), textureIds.size() * sizeof(uint32_t), hipMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    std::cout << "Starting path tracing with progressive refinement...\n";
    std::cout << "Camera orbits around Cornell box scene\n\n";

    auto startTime = std::chrono::high_resolution_clock::now();
    int totalSamples = 0;
    int totalRequests = 0;

    // Progressive rendering loop
    for (int frame = 0; frame < numFrames; ++frame) {
        // Camera angle - slowly orbit around the scene
        float cameraAngle = static_cast<float>(frame) * 0.05f;
        
        // Get device context for this launch
        hip_demand::DeviceContext ctx = loader.getDeviceContext();
        
        // Launch path tracing kernel
        int numTexturesArg = static_cast<int>(textureIds.size());
        void* args[] = {
            &d_output,
            &d_accumulator,
            &width, &height,
            &frame,
            &samplesPerPixel,
            &d_textureIds,
            &numTexturesArg,
            &maxBounces,
            &cameraAngle,
            &ctx
        };
        
        hipError_t err = hipModuleLaunchKernel(
            module.kernel,
            gridSize.x, gridSize.y, 1,
            blockSize.x, blockSize.y, 1,
            0, stream,
            args, nullptr
        );
        
        if (err != hipSuccess) {
            std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << "\n";
            break;
        }
        
        hipStreamSynchronize(stream);
        
        // Process texture requests
        size_t requestCount = loader.processRequests(stream, ctx);
        totalRequests += static_cast<int>(requestCount);
        totalSamples += width * height * samplesPerPixel;
        
        // Progress update
        if ((frame + 1) % 4 == 0 || frame == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float>(now - startTime).count();
            float sps = static_cast<float>(totalSamples) / elapsed / 1e6f;
            
            std::cout << "Frame " << (frame + 1) << "/" << numFrames
                      << " | Samples: " << totalSamples / 1000000 << "M"
                      << " | Requests: " << requestCount
                      << " | " << std::fixed << std::setprecision(2) << sps << " MS/s\n";
        }
        
        // Save intermediate result
        if ((frame + 1) % saveInterval == 0 || frame == numFrames - 1) {
            std::vector<float4> h_output(width * height);
            hipMemcpy(h_output.data(), d_output, width * height * sizeof(float4), hipMemcpyDeviceToHost);
            
            std::vector<uint8_t> output_rgb(width * height * 3);
            for (int i = 0; i < width * height; ++i) {
                // Gamma correction
                float r = powf(fmaxf(0.0f, fminf(1.0f, h_output[i].x)), 1.0f / 2.2f);
                float g = powf(fmaxf(0.0f, fminf(1.0f, h_output[i].y)), 1.0f / 2.2f);
                float b = powf(fmaxf(0.0f, fminf(1.0f, h_output[i].z)), 1.0f / 2.2f);
                
                output_rgb[i * 3 + 0] = static_cast<uint8_t>(r * 255.0f);
                output_rgb[i * 3 + 1] = static_cast<uint8_t>(g * 255.0f);
                output_rgb[i * 3 + 2] = static_cast<uint8_t>(b * 255.0f);
            }
            
            char filename[256];
            sprintf(filename, "%s/render_%04d.png", outputDir.c_str(), frame + 1);
            stbi_write_png(filename, width, height, 3, output_rgb.data(), width * 3);
            
            int accumulatedSamples = (frame + 1) * samplesPerPixel;
            std::cout << "  -> Saved " << filename << " (" << accumulatedSamples << " spp)\n";
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    float totalTime = std::chrono::duration<float>(endTime - startTime).count();

    std::cout << "\n";
    std::cout << "=== Path Tracing Complete ===\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << totalTime << " seconds\n";
    std::cout << "Total samples: " << totalSamples / 1000000 << " million\n";
    std::cout << "Average: " << std::fixed << std::setprecision(2) 
              << (static_cast<float>(totalSamples) / totalTime / 1e6f) << " MS/s\n";
    std::cout << "Total texture requests: " << totalRequests << "\n";
    std::cout << "\nOutput saved to: " << outputDir << "/\n";

    // Save final render with higher quality filename
    {
        std::vector<float4> h_output(width * height);
        hipMemcpy(h_output.data(), d_output, width * height * sizeof(float4), hipMemcpyDeviceToHost);
        
        std::vector<uint8_t> output_rgb(width * height * 3);
        for (int i = 0; i < width * height; ++i) {
            float r = powf(fmaxf(0.0f, fminf(1.0f, h_output[i].x)), 1.0f / 2.2f);
            float g = powf(fmaxf(0.0f, fminf(1.0f, h_output[i].y)), 1.0f / 2.2f);
            float b = powf(fmaxf(0.0f, fminf(1.0f, h_output[i].z)), 1.0f / 2.2f);
            
            output_rgb[i * 3 + 0] = static_cast<uint8_t>(r * 255.0f);
            output_rgb[i * 3 + 1] = static_cast<uint8_t>(g * 255.0f);
            output_rgb[i * 3 + 2] = static_cast<uint8_t>(b * 255.0f);
        }
        
        std::string finalPath = outputDir + "/final_render.png";
        stbi_write_png(finalPath.c_str(), width, height, 3, output_rgb.data(), width * 3);
        std::cout << "Final render: " << finalPath << " (" << numFrames * samplesPerPixel << " spp)\n";
    }

    // Cleanup
    hipFree(d_output);
    hipFree(d_accumulator);
    hipFree(d_textureIds);
    hipStreamDestroy(stream);

    std::cout << "\nDone!\n";
    return 0;
}
