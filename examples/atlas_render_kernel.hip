#include "DemandLoading/TextureSampling.h"
#include <hip/hip_runtime.h>

using namespace hip_demand;

// Atlas stress: many tiles, each can switch textures per pass to churn residency.
__device__ __forceinline__ void renderKernel(
    DeviceContext ctx,
    float4* output,
    int width, int height,
    const uint32_t* textureIds,
    int numTextures,
    int pass)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Tile grid across the screen
    const int tilesX = 16;
    const int tilesY = 9;
    float tileW = static_cast<float>(width) / tilesX;
    float tileH = static_cast<float>(height) / tilesY;

    int gridX = min(tilesX - 1, x / static_cast<int>(tileW));
    int gridY = min(tilesY - 1, y / static_cast<int>(tileH));
    int tileId = gridY * tilesX + gridX;

    // Pick texture based on tile id and pass to force churn
    int texIndex = (tileId + pass) % numTextures;
    uint32_t texId = textureIds[texIndex];

    // Local UV inside tile with a slight animated offset
    float u = (x - gridX * tileW + 0.5f) / tileW;
    float v = (y - gridY * tileH + 0.5f) / tileH;
    u = fmodf(u + 0.05f * sinf(0.7f * pass + v * 6.28318f), 1.0f);
    v = fmodf(v + 0.05f * cosf(0.6f * pass + u * 6.28318f), 1.0f);

    float4 color;
    bool resident = tex2D(ctx, texId, u, v, color);
    if (!resident) {
        color = make_float4(1.0f, 0.0f, 1.0f, 1.0f);
    }

    output[y * width + x] = color;
}

extern "C" __global__ void renderKernelWrapper(
    DeviceContext ctx,
    float4* output,
    int width, int height,
    const uint32_t* textureIds,
    int numTextures,
    int pass)
{
    renderKernel(ctx, output, width, height, textureIds, numTextures, pass);
}
