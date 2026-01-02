// SPDX-License-Identifier: MIT
// Streaming Scene Example
// 
// Demonstrates demand texture loading with a camera flying over a large tile grid.
// Shows:
// - Texture streaming based on camera visibility
// - Memory budget and eviction in action
// - Eviction priorities (center tiles = High, edge tiles = Low)
// - Thrashing prevention
//
// Output: Series of frames showing which textures are resident/loading/evicted

#include <hip/hip_runtime.h>

#include "DemandLoading/DemandTextureLoader.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Must match kernel definitions - use int for visible for cross-compiler compatibility
struct TileInfo {
    uint32_t textureId;
    float worldX;
    float worldZ;
    int visible;
};

struct CameraInfo {
    float posX, posY, posZ;
    float targetX, targetZ;
    float fov;
    float nearPlane, farPlane;
};

struct KernelModule {
    hipModule_t module = nullptr;
    hipFunction_t kernel = nullptr;

    ~KernelModule() {
        if (module) hipModuleUnload(module);
    }

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

// Simple 5x7 bitmap font for digits 0-9
const uint8_t DIGIT_FONT[10][7] = {
    {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}, // 0
    {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}, // 1
    {0x0E, 0x11, 0x01, 0x0E, 0x10, 0x10, 0x1F}, // 2
    {0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E}, // 3
    {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}, // 4
    {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}, // 5
    {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}, // 6
    {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}, // 7
    {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}, // 8
    {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}, // 9
};

// Draw a digit at position (px, py) with given scale
void drawDigit(std::vector<uint8_t>& data, int size, int digit, int px, int py, int scale, 
               uint8_t r, uint8_t g, uint8_t b) {
    if (digit < 0 || digit > 9) return;
    
    for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 5; ++col) {
            if (DIGIT_FONT[digit][row] & (0x10 >> col)) {
                // Draw scaled pixel
                for (int sy = 0; sy < scale; ++sy) {
                    for (int sx = 0; sx < scale; ++sx) {
                        int x = px + col * scale + sx;
                        int y = py + row * scale + sy;
                        if (x >= 0 && x < size && y >= 0 && y < size) {
                            int idx = (y * size + x) * 4;
                            data[idx + 0] = r;
                            data[idx + 1] = g;
                            data[idx + 2] = b;
                        }
                    }
                }
            }
        }
    }
}

// Draw a number (up to 2 digits) centered at (cx, cy)
void drawNumber(std::vector<uint8_t>& data, int size, int number, int cx, int cy, int scale,
                uint8_t r, uint8_t g, uint8_t b) {
    number = std::max(0, std::min(99, number));  // Clamp to 0-99
    
    int digitWidth = 5 * scale + scale;  // 5 pixels + 1 spacing
    
    if (number < 10) {
        // Single digit - center it
        drawDigit(data, size, number, cx - (5 * scale) / 2, cy - (7 * scale) / 2, scale, r, g, b);
    } else {
        // Two digits
        int tens = number / 10;
        int ones = number % 10;
        int totalWidth = digitWidth * 2 - scale;  // Two digits with spacing
        int startX = cx - totalWidth / 2;
        drawDigit(data, size, tens, startX, cy - (7 * scale) / 2, scale, r, g, b);
        drawDigit(data, size, ones, startX + digitWidth, cy - (7 * scale) / 2, scale, r, g, b);
    }
}

// Generate a simple colored tile texture with coordinate overlay
std::vector<uint8_t> generateTileTexture(int tileX, int tileZ, int numTilesX, int size = 256) {
    std::vector<uint8_t> data(size * size * 4);
    
    // Color based on position (creates a gradient across the grid)
    float r = float(tileX) / numTilesX;
    float g = 0.3f;
    float b = float(tileZ) / numTilesX;
    
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int idx = (y * size + x) * 4;
            
            // Checker pattern
            bool checker = ((x / 32) + (y / 32)) % 2 == 0;
            float brightness = checker ? 1.0f : 0.7f;
            
            data[idx + 0] = uint8_t(r * brightness * 255);
            data[idx + 1] = uint8_t(g * brightness * 255);
            data[idx + 2] = uint8_t(b * brightness * 255);
            data[idx + 3] = 255;
        }
    }
    
    // Draw tile coordinates "X,Z" in the center
    int cx = size / 2;
    int cy = size / 2;
    int scale = 4;  // 4x scale for visibility
    
    // Draw background box for better visibility
    int boxW = 60;
    int boxH = 40;
    for (int dy = -boxH/2; dy <= boxH/2; ++dy) {
        for (int dx = -boxW/2; dx <= boxW/2; ++dx) {
            int px = cx + dx;
            int py = cy + dy;
            if (px >= 0 && px < size && py >= 0 && py < size) {
                int idx = (py * size + px) * 4;
                data[idx + 0] = 0;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
                data[idx + 3] = 255;
            }
        }
    }
    
    // Draw X coordinate on top, Z coordinate on bottom
    drawNumber(data, size, tileX, cx, cy - 12, scale, 255, 255, 100);  // Yellow for X
    drawNumber(data, size, tileZ, cx, cy + 12, scale, 100, 255, 255);  // Cyan for Z
    
    return data;
}

float distance2D(float x1, float z1, float x2, float z2) {
    float dx = x2 - x1;
    float dz = z2 - z1;
    return std::sqrt(dx * dx + dz * dz);
}

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    
    // Parse command-line arguments
    int saveInterval = 10;  // Default: save every 10 frames
    for (int i = 1; i < argc; ++i) {
        if ((std::string(argv[i]) == "--save-interval" || std::string(argv[i]) == "-s") && i + 1 < argc) {
            saveInterval = std::atoi(argv[i + 1]);
            if (saveInterval < 1) saveInterval = 1;
            ++i;
        } else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  -s, --save-interval N   Save frame every N iterations (default: 10)\n";
            std::cout << "                          Use 0 to disable frame saving\n";
            std::cout << "  -h, --help              Show this help\n";
            return 0;
        }
    }
    
    std::cout << "===========================================\n";
    std::cout << "   Streaming Scene Demo\n";
    std::cout << "===========================================\n\n";

    // Configuration
    const int GRID_SIZE = 8;          // 8x8 = 64 tiles
    const int TEXTURE_SIZE = 256;     // 256x256 textures
    const float TILE_WORLD_SIZE = 10.0f;  // Each tile is 10 units
    const float VIEW_RADIUS = 25.0f;  // Camera can see 25 units
    const int NUM_FRAMES = 120;       // Animation frames
    const int OUTPUT_WIDTH = 512;
    const int OUTPUT_HEIGHT = 512;
    
    // Memory budget: enough for ~16 tiles (of 64 total)
    const size_t MEMORY_BUDGET = 16 * TEXTURE_SIZE * TEXTURE_SIZE * 4;

    // Initialize HIP
    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No HIP devices found!\n";
        return 1;
    }

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n\n";

    // Load kernel
    KernelModule kernelModule;
    if (!kernelModule.load("streaming_scene_kernel.co", "streamingSceneKernel")) {
        std::cerr << "Failed to load kernel. Ensure streaming_scene_kernel.co exists.\n";
        return 1;
    }

    // Create texture loader with memory budget
    hip_demand::LoaderOptions options;
    options.maxTextureMemory = MEMORY_BUDGET;
    options.maxTextures = GRID_SIZE * GRID_SIZE + 16;
    options.maxRequestsPerLaunch = OUTPUT_WIDTH * OUTPUT_HEIGHT;
    options.enableEviction = true;
    options.minResidentFrames = 3;  // Thrashing prevention
    
    hip_demand::DemandTextureLoader loader(options);

    std::cout << "Configuration:\n";
    std::cout << "  Grid: " << GRID_SIZE << "x" << GRID_SIZE << " = " << (GRID_SIZE * GRID_SIZE) << " tiles\n";
    std::cout << "  Texture size: " << TEXTURE_SIZE << "x" << TEXTURE_SIZE << "\n";
    std::cout << "  Memory budget: " << (MEMORY_BUDGET / 1024 / 1024) << " MB (~" 
              << (MEMORY_BUDGET / (TEXTURE_SIZE * TEXTURE_SIZE * 4)) << " tiles)\n";
    std::cout << "  View radius: " << VIEW_RADIUS << " units\n";
    std::cout << "  Frames: " << NUM_FRAMES << "\n";
    std::cout << "  Save interval: " << (saveInterval > 0 ? std::to_string(saveInterval) : "disabled") << "\n\n";

    // Create output directory
    fs::create_directories("streaming_output");

    // Create tile textures and register with loader
    std::cout << "Creating tile textures...\n";
    std::vector<TileInfo> tiles(GRID_SIZE * GRID_SIZE);
    
    for (int z = 0; z < GRID_SIZE; ++z) {
        for (int x = 0; x < GRID_SIZE; ++x) {
            int tileIdx = z * GRID_SIZE + x;
            
            // Generate texture data
            auto texData = generateTileTexture(x, z, GRID_SIZE, TEXTURE_SIZE);
            
            // Initial priority will be updated dynamically each frame based on camera distance
            hip_demand::TextureDesc desc;
            desc.generateMipmaps = false;
            desc.evictionPriority = hip_demand::EvictionPriority::Normal;
            
            // Create texture from memory
            auto handle = loader.createTextureFromMemory(
                texData.data(), TEXTURE_SIZE, TEXTURE_SIZE, 4, desc);
            
            if (!handle.valid) {
                std::cerr << "Failed to create texture for tile " << tileIdx << "\n";
                return 1;
            }
            
            // Setup tile info
            tiles[tileIdx].textureId = handle.id;
            tiles[tileIdx].worldX = (x + 0.5f) * TILE_WORLD_SIZE;
            tiles[tileIdx].worldZ = (z + 0.5f) * TILE_WORLD_SIZE;
            tiles[tileIdx].visible = 0;
        }
    }
    std::cout << "Created " << tiles.size() << " tile textures\n\n";

    // Allocate GPU resources
    float4* d_output = nullptr;
    TileInfo* d_tiles = nullptr;
    hipMalloc(&d_output, OUTPUT_WIDTH * OUTPUT_HEIGHT * sizeof(float4));
    hipMalloc(&d_tiles, tiles.size() * sizeof(TileInfo));

    std::vector<uint8_t> outputImage(OUTPUT_WIDTH * OUTPUT_HEIGHT * sizeof(float4));

    // Animation: camera moves in a circle around the grid
    float worldExtent = GRID_SIZE * TILE_WORLD_SIZE;
    float circleRadius = worldExtent * 0.35f;
    float circleCenter = worldExtent * 0.5f;

    std::cout << "Rendering " << NUM_FRAMES << " frames...\n";
    std::cout << std::setw(6) << "Frame" << std::setw(10) << "Resident" 
              << std::setw(12) << "Memory MB" << std::setw(10) << "Visible"
              << std::setw(12) << "Requests" << "\n";
    std::cout << std::string(50, '-') << "\n";

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int frame = 0; frame < NUM_FRAMES; ++frame) {
        // Update camera position (circular path)
        float t = float(frame) / NUM_FRAMES * 2.0f * 3.14159f;
        CameraInfo camera;
        camera.posX = circleCenter + std::cos(t) * circleRadius;
        camera.posZ = circleCenter + std::sin(t) * circleRadius;
        camera.posY = 50.0f;  // Height (not used in top-down view)
        camera.targetX = circleCenter;
        camera.targetZ = circleCenter;
        camera.fov = 1.0f;
        camera.nearPlane = 1.0f;
        camera.farPlane = VIEW_RADIUS;

        // Update tile visibility and eviction priority based on distance from camera
        int visibleCount = 0;
        for (auto& tile : tiles) {
            float dist = distance2D(camera.posX, camera.posZ, tile.worldX, tile.worldZ);
            tile.visible = (dist <= VIEW_RADIUS) ? 1 : 0;
            if (tile.visible) visibleCount++;
            
            // Dynamic eviction priority based on camera distance
            // Use tighter zones and KeepResident for tiles right under camera
            hip_demand::EvictionPriority priority;
            if (dist <= TILE_WORLD_SIZE * 1.5f) {
                // Within 1.5 tile distance - NEVER evict these
                priority = hip_demand::EvictionPriority::KeepResident;
            } else if (dist <= VIEW_RADIUS * 0.4f) {
                priority = hip_demand::EvictionPriority::High;      // Close - keep if possible
            } else if (dist <= VIEW_RADIUS * 0.6f) {
                priority = hip_demand::EvictionPriority::Normal;    // Medium distance
            } else {
                priority = hip_demand::EvictionPriority::Low;       // Outer 40% - evict first
            }
            loader.updateEvictionPriority(tile.textureId, priority);
        }

        // Upload tile info to GPU
        hipMemcpy(d_tiles, tiles.data(), tiles.size() * sizeof(TileInfo), hipMemcpyHostToDevice);

        // Prepare for launch and get device context
        loader.launchPrepare();
        hip_demand::DeviceContext ctx = loader.getDeviceContext();

        // Launch kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((OUTPUT_WIDTH + 15) / 16, (OUTPUT_HEIGHT + 15) / 16);

        int outputWidth = OUTPUT_WIDTH;
        int outputHeight = OUTPUT_HEIGHT;
        int gridSizeX = GRID_SIZE;
        int gridSizeY = GRID_SIZE;
        float tileWorldSize = TILE_WORLD_SIZE;

        void* args[] = {
            &ctx, &d_output,
            &outputWidth, &outputHeight,
            &d_tiles,
            &gridSizeX, &gridSizeY,
            &tileWorldSize,
            &camera
        };

        hipError_t err = hipModuleLaunchKernel(kernelModule.kernel,
                              gridSize.x, gridSize.y, 1,
                              blockSize.x, blockSize.y, 1,
                              0, nullptr, args, nullptr);
        if (err != hipSuccess) {
            std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << "\n";
            return 1;
        }

        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::cerr << "hipDeviceSynchronize failed: " << hipGetErrorString(err) << "\n";
            return 1;
        }

        // Process texture requests (loads visible textures, may evict old ones)
        loader.processRequests(nullptr, ctx);

        // Print stats
        size_t residentCount = loader.getResidentTextureCount();
        size_t memoryUsage = loader.getTotalTextureMemory();
        size_t requestCount = loader.getRequestCount();

        std::cout << std::setw(6) << frame 
                  << std::setw(10) << residentCount
                  << std::setw(12) << std::fixed << std::setprecision(2) 
                  << (double(memoryUsage) / 1024.0 / 1024.0)
                  << std::setw(10) << visibleCount
                  << std::setw(12) << requestCount << "\n";

        // Save frame at specified interval
        if (saveInterval > 0 && frame % saveInterval == 0) {
            err = hipMemcpy(outputImage.data(), d_output, 
                     OUTPUT_WIDTH * OUTPUT_HEIGHT * sizeof(float4), 
                     hipMemcpyDeviceToHost);
            if (err != hipSuccess) {
                std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << "\n";
                return 1;
            }

            // Convert float4 to RGBA8
            std::vector<uint8_t> rgba(OUTPUT_WIDTH * OUTPUT_HEIGHT * 4);
            for (int i = 0; i < OUTPUT_WIDTH * OUTPUT_HEIGHT; ++i) {
                float4* pixel = reinterpret_cast<float4*>(outputImage.data()) + i;
                rgba[i * 4 + 0] = uint8_t(std::min(255.0f, pixel->x * 255.0f));
                rgba[i * 4 + 1] = uint8_t(std::min(255.0f, pixel->y * 255.0f));
                rgba[i * 4 + 2] = uint8_t(std::min(255.0f, pixel->z * 255.0f));
                rgba[i * 4 + 3] = 255;
            }

            char filename[256];
            snprintf(filename, sizeof(filename), "streaming_output/frame_%03d.png", frame);
            if (!stbi_write_png(filename, OUTPUT_WIDTH, OUTPUT_HEIGHT, 4, rgba.data(), OUTPUT_WIDTH * 4)) {
                std::cerr << "Failed to write " << filename << "\n";
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "\n";
    std::cout << "===========================================\n";
    std::cout << "   Summary\n";
    std::cout << "===========================================\n";
    std::cout << "Total time: " << duration.count() << " ms\n";
    std::cout << "Avg frame time: " << (duration.count() / NUM_FRAMES) << " ms\n";
    std::cout << "Final resident textures: " << loader.getResidentTextureCount() << "\n";
    std::cout << "Final memory usage: " << std::fixed << std::setprecision(2)
              << (double(loader.getTotalTextureMemory()) / 1024.0 / 1024.0) << " MB\n";
    std::cout << "\nOutput frames saved to streaming_output/\n";
    std::cout << "  - Colored tiles: texture is resident\n";
    std::cout << "  - Pink tiles: texture loading (not yet resident)\n";
    std::cout << "  - Dark gray tiles: outside view radius\n";
    std::cout << "  - Yellow dot: camera position\n";
    std::cout << "  - Green circle: view radius\n";

    // Cleanup
    hipFree(d_output);
    hipFree(d_tiles);

    return 0;
}
