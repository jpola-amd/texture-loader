#include <hip/hip_runtime.h>
#include "DemandLoading/TextureSampling.h"

using namespace hip_demand;

__global__ void renderKernel(
    DeviceContext ctx,
    float4* output,
    int width, int height,
    const uint32_t* textureIds,
    int numTextures,
    float time)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Compute UV coordinates
    float u = (float)x / width;
    float v = (float)y / height;
    
    // Animate UV coordinates
    u += sinf(time + v * 3.14159f) * 0.1f;
    v += cosf(time + u * 3.14159f) * 0.1f;
    
    // Select texture based on screen position (creates a grid)
    int gridX = (x * 4) / width;
    int gridY = (y * 4) / height;
    int texIndex = (gridY * 4 + gridX) % numTextures;
    uint32_t texId = textureIds[texIndex];
    
    // Sample texture with demand loading
    float4 color;
    bool resident = tex2D(ctx, texId, u, v, color);
    
    // If not resident, the color will be magenta by default
    // This creates a visible indication of missing textures
    
    // Darken non-resident pixels slightly for debugging
    if (!resident) {
        color.x *= 0.5f;
        color.y *= 0.5f;
        color.z *= 0.5f;
    }
    
    output[idx] = color;
}

// Export the kernel directly with C linkage to prevent name mangling
// This allows hipModuleGetFunction to find it by name
extern "C" __global__ void renderKernelWrapper(
    DeviceContext ctx,
    float4* output,
    int width, int height,
    const uint32_t* textureIds,
    int numTextures,
    float time)
{
    // This is the actual kernel entry point when called via hipModuleLaunchKernel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Compute UV coordinates
    float u = (float)x / width;
    float v = (float)y / height;
    
    // Animate UV coordinates
    u += sinf(time + v * 3.14159f) * 0.1f;
    v += cosf(time + u * 3.14159f) * 0.1f;
    
    // Select texture based on screen position (creates a grid)
    int gridX = (x * 4) / width;
    int gridY = (y * 4) / height;
    int texIndex = (gridY * 4 + gridX) % numTextures;
    uint32_t texId = textureIds[texIndex];
    
    // Sample texture with demand loading
    float4 color;
    bool resident = tex2D(ctx, texId, u, v, color);
    
    // If not resident, the color will be magenta by default
    // This creates a visible indication of missing textures
    
    // Darken non-resident pixels slightly for debugging
    if (!resident) {
        color.x *= 0.5f;
        color.y *= 0.5f;
        color.z *= 0.5f;
    }
    
    output[idx] = color;
}
