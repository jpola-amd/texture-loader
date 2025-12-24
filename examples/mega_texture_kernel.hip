#include "DemandLoading/TextureSampling.h"
#include <hip/hip_runtime.h>

using namespace hip_demand;

// Pan and zoom across a large texture to stress demand loading.
__device__ __forceinline__ void renderKernel(
    DeviceContext ctx,
    float4* output,
    int width, int height,
    uint32_t textureId,
    float time,
    float zoom)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float fx = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
    float fy = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);

    // Smooth camera pan and gentle zoom oscillation.
    float offsetX = 0.07f * time;
    float offsetY = 0.05f * time;
    float localZoom = zoom * (0.85f + 0.15f * __cosf(time * 0.6f));

    float u = fx * localZoom + offsetX;
    float v = fy * localZoom + offsetY;

    float4 color;
    bool resident = tex2D(ctx, textureId, u, v, color);

    if (!resident) {
        color = make_float4(1.0f, 0.2f, 1.0f, 1.0f);
    }

    output[y * width + x] = color;
}

extern "C" __global__ void renderKernelWrapper(
    DeviceContext ctx,
    float4* output,
    int width, int height,
    uint32_t textureId,
    float time,
    float zoom)
{
    renderKernel(ctx, output, width, height, textureId, time, zoom);
}
