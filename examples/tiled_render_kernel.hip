#include <hip/hip_runtime.h>
#include "DemandLoading/TextureSampling.h"

using namespace hip_demand;

// Tile-aware sampling: each screen tile maps to the full 0..1 UV range of its texture.
__device__ __forceinline__ void renderKernel(
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

    // 4x4 tiling across the screen
    const int tilesX = 4;
    const int tilesY = 4;
    const float tileW = static_cast<float>(width) / tilesX;
    const float tileH = static_cast<float>(height) / tilesY;

    int gridX = min(tilesX - 1, x / static_cast<int>(tileW));
    int gridY = min(tilesY - 1, y / static_cast<int>(tileH));
    int texIndex = (gridY * tilesX + gridX) % numTextures;
    uint32_t texId = textureIds[texIndex];

    // Local UVs inside the tile (0..1)
    float u = (x - gridX * tileW) / tileW;
    float v = (y - gridY * tileH) / tileH;

    // Small animation to keep things moving without crossing tiles
    u = fmodf(u + 0.1f * sinf(time + v * 3.14159f), 1.0f);
    v = fmodf(v + 0.1f * cosf(time + u * 3.14159f), 1.0f);

    float4 color;
    bool resident = tex2D(ctx, texId, u, v, color);

    if (!resident) {
        // Magenta miss indicator, slightly darkened
        color = make_float4(1.0f, 0.0f, 1.0f, 1.0f);
        color.x *= 0.5f; color.y *= 0.5f; color.z *= 0.5f;
    }

    output[y * width + x] = color;
}

// C linkage wrapper for module API
extern "C" __global__ void renderKernelWrapper(
    DeviceContext ctx,
    float4* output,
    int width, int height,
    const uint32_t* textureIds,
    int numTextures,
    float time)
{
    renderKernel(ctx, output, width, height, textureIds, numTextures, time);
}
