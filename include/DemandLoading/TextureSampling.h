#pragma once

#include "DemandLoading/DeviceContext.h"
#include <hip/hip_runtime.h>
#include <hip/texture_types.h>

#if !defined(HIP_ENABLE_WARP_SYNC_BUILTINS)
#warning "HIP_ENABLE_WARP_SYNC_BUILTINS is not defined; warp-sync builtins may be unavailable"
#endif

namespace hip_demand {

// Device-side texture sampling functions
// These check residency and record requests if needed

__device__ __forceinline__ bool isTextureResident(const DeviceContext& ctx, uint32_t texId) {
    if (texId >= ctx.maxTextures) return false;
    const uint32_t wordIdx = texId >> 5;   // divide by 32
    const uint32_t bitIdx  = texId & 31u;  // modulo 32
    return (ctx.residentFlags[wordIdx] & (1u << bitIdx)) != 0;
}

// Record a texture request; use warp-level dedup where supported, otherwise per-thread atomics.
__device__ __forceinline__ void recordTextureRequest(const DeviceContext& ctx, uint32_t texId) {
    // If overflow already flagged, skip atomics to reduce contention.
    // This is a best-effort early-out; using a volatile load is sufficient here.
    const volatile uint32_t* overflowPtr = reinterpret_cast<volatile uint32_t*>(ctx.requestOverflow);
    if (*overflowPtr != 0u) return;

#if defined(HIP_ENABLE_WARP_SYNC_BUILTINS)
    // Warp-level deduplication: only the leader for a given texId appends
    const unsigned long long active = __activemask();
    const unsigned long long match = __match_any_sync(active, texId);
    const int leader = __ffsll(match) - 1;  // lowest set bit index
    const int lane = static_cast<int>(__lane_id());
    if (lane != leader) return;

    const uint32_t idx = atomicAdd(ctx.requestCount, 1u);
    if (idx < ctx.maxRequests) {
        ctx.requests[idx] = texId;
    } else {
        atomicExch(ctx.requestOverflow, 1u);
    }
#else
    // Fallback when warp-sync builtins are unavailable
    const uint32_t idx = atomicAdd(ctx.requestCount, 1u);
    if (idx < ctx.maxRequests) {
        ctx.requests[idx] = texId;
    } else {
        atomicExch(ctx.requestOverflow, 1u);
    }
#endif
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
