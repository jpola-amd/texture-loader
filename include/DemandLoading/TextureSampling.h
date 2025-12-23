#pragma once

#include "DemandLoading/DeviceContext.h"
#include <hip/hip_runtime.h>
#include <hip/texture_types.h>

namespace hip_demand {

// Device-side texture sampling functions
// These check residency and record requests if needed

__device__ __forceinline__ bool isTextureResident(const DeviceContext& ctx, uint32_t texId) {
    if (texId >= ctx.maxTextures) return false;
    const uint32_t wordIdx = texId >> 5;   // divide by 32
    const uint32_t bitIdx  = texId & 31u;  // modulo 32
    return (ctx.residentFlags[wordIdx] & (1u << bitIdx)) != 0;
}

// Warp-level deduplication: only the first lane for a given texId appends a request.
__device__ __forceinline__ void recordTextureRequest(const DeviceContext& ctx, uint32_t texId) {
    // If overflow already flagged, skip atomics to reduce contention
    if (atomicAdd(ctx.requestOverflow, 0u) != 0u) return;

    // Deduplicate within a warp/wave using match_any; only the leader appends
    const unsigned active = __ballot_sync(0xFFFFFFFFu, true);
    const unsigned match = __match_any_sync(active, texId);
    const int leader = __ffs(match) - 1;  // lowest set bit index
    const int lane = static_cast<int>(__lane_id());
    if (lane != leader) return;

    const uint32_t idx = atomicAdd(ctx.requestCount, 1u);
    if (idx < ctx.maxRequests) {
        ctx.requests[idx] = texId;
    } else {
        // Set overflow flag
        atomicExch(ctx.requestOverflow, 1u);
    }
}

// Main texture sampling function
// Returns true if texture is resident and sampled successfully
__device__ inline bool tex2D(const DeviceContext& ctx,
                             uint32_t texId,
                             float u, float v,
                             float4& result,
                             float4 defaultColor = make_float4(1.0f, 0.0f, 1.0f, 1.0f)) {
    // Bounds check
    if (texId >= ctx.maxTextures) {
        result = defaultColor;
        return false;
    }
    
    if (!isTextureResident(ctx, texId)) {
        recordTextureRequest(ctx, texId);
        result = defaultColor;
        return false;
    }
    
    result = ::tex2D<float4>(ctx.textures[texId], u, v);
    return true;
}

// Gradient-based sampling (for mipmap LOD control)
__device__ inline bool tex2DGrad(const DeviceContext& ctx,
                                  uint32_t texId,
                                  float u, float v,
                                  float2 ddx, float2 ddy,
                                  float4& result,
                                  float4 defaultColor = make_float4(1.0f, 0.0f, 1.0f, 1.0f)) {
    // Bounds check
    if (texId >= ctx.maxTextures) {
        result = defaultColor;
        return false;
    }
    
    if (!isTextureResident(ctx, texId)) {
        recordTextureRequest(ctx, texId);
        result = defaultColor;
        return false;
    }
    
    result = ::tex2DGrad<float4>(ctx.textures[texId], u, v, ddx, ddy);
    return true;
}

// LOD-based sampling
__device__ inline bool tex2DLod(const DeviceContext& ctx,
                                uint32_t texId,
                                float u, float v,
                                float lod,
                                float4& result,
                                float4 defaultColor = make_float4(1.0f, 0.0f, 1.0f, 1.0f)) {
    // Bounds check
    if (texId >= ctx.maxTextures) {
        result = defaultColor;
        return false;
    }
    
    if (!isTextureResident(ctx, texId)) {
        recordTextureRequest(ctx, texId);
        result = defaultColor;
        return false;
    }
    
    result = ::tex2DLod<float4>(ctx.textures[texId], u, v, lod);
    return true;
}

} // namespace hip_demand
