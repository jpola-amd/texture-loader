#include <hip/hip_runtime.h>
#include "DemandLoading/TextureSampling.h"

using namespace hip_demand;

// Simple kernel that requests textures for benchmarking
__device__ void benchmarkKernel(
    DeviceContext ctx,
    const uint32_t* textureIds,
    int numTextureIds,
    int numRequesters,
    int iterations)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // For benchmarking loader overhead, we want deterministic request volume.
    // Limit request generation to the first numRequesters threads so we don't
    // overflow the request buffer when launching large grids.
    if (tid >= numRequesters) {
        return;
    }
    
    // Each thread requests different textures
    for (int i = 0; i < iterations; ++i) {
        int texIndex = (tid + i) % numTextureIds;
        uint32_t texId = textureIds[texIndex];
        
        float u = (tid % 256) / 256.0f;
        float v = (tid / 256) / 256.0f;
        
        float4 color;
        tex2D(ctx, texId, u, v, color);
    }
}

extern "C" __global__ void benchmarkKernelWrapper(
    DeviceContext ctx,
    const uint32_t* textureIds,
    int numTextureIds,
    int numRequesters,
    int iterations)
{
    benchmarkKernel(ctx, textureIds, numTextureIds, numRequesters, iterations);
}
