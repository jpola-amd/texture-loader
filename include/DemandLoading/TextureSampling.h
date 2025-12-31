#pragma once

#include "DemandLoading/DeviceContext.h"
#include <hip/hip_runtime.h>
#include <hip/texture_types.h>

#if !defined(HIP_ENABLE_WARP_SYNC_BUILTINS)
#warning "HIP_ENABLE_WARP_SYNC_BUILTINS is not defined; warp-sync builtins may be unavailable"
#endif

namespace hip_demand {

//=============================================================================
// Device-side Helper Functions
//=============================================================================

// Get the current wave (wavefront) size at runtime
// Note: RDNA (gfx10/11/12) uses wave32, older GCN uses wave64
__device__ __forceinline__ int getWaveSize() {
#if defined(__AMDGCN__)
    return __builtin_amdgcn_wavefrontsize();
#else
    return 64;  // Default for unknown architectures
#endif
}

// Get the current lane ID within the wave
__device__ __forceinline__ int getLaneId() {
#if defined(__AMDGCN__)
    return __lane_id();
#else
    return threadIdx.x % 64;
#endif
}

// Get the active lane mask for the current wave
__device__ __forceinline__ uint64_t getActiveMask() {
#if defined(HIP_ENABLE_WARP_SYNC_BUILTINS)
    return __activemask();
#elif defined(__AMDGCN__)
    return __builtin_amdgcn_read_exec();
#else
    return ~0ULL;  // Assume all lanes active
#endif
}

//=============================================================================
// Texture Residency Check
//=============================================================================

__device__ __forceinline__ bool isTextureResident(const DeviceContext& ctx, uint32_t texId) {
    if (texId >= ctx.maxTextures) return false;
    const uint32_t wordIdx = texId >> 5;   // divide by 32
    const uint32_t bitIdx  = texId & 31u;  // modulo 32
    return (ctx.residentFlags[wordIdx] & (1u << bitIdx)) != 0;
}

//=============================================================================
// Request Recording with Wave-Level Deduplication
//=============================================================================

// Record a texture request with wave-level deduplication to reduce atomic contention.
// On AMD GPUs, this uses native wave intrinsics for efficient deduplication.
__device__ __forceinline__ void recordTextureRequest(const DeviceContext& ctx, uint32_t texId) {
    // Early-out if overflow already flagged to reduce global memory traffic.
    // Use __atomic_load_n for a true atomic load (no read-modify-write overhead).
    if (__atomic_load_n(ctx.requestOverflow, __ATOMIC_RELAXED) != 0u) return;

#if defined(HIP_ENABLE_WARP_SYNC_BUILTINS)
    // Wave-level deduplication: only one lane per unique texId writes to global memory.
    // __match_any_sync returns a mask of lanes that have the same value.
    const uint64_t active = __activemask();
    const uint64_t match = __match_any_sync(active, texId);
    
    // Find the leader lane (lowest active lane with this texId)
    const int leader = __ffsll(static_cast<long long>(match)) - 1;
    const int lane = getLaneId();
    
    // Only the leader lane issues the atomic
    if (lane != leader) return;
#endif

    // Common path: issue atomic (either as leader or when no wave-sync available)
    const uint32_t idx = atomicAdd(ctx.requestCount, 1u);
    if (idx < ctx.maxRequests) {
        ctx.requests[idx] = texId;
    } else {
        atomicExch(ctx.requestOverflow, 1u);
    }
}

//=============================================================================
// Main Texture Sampling Functions
//=============================================================================

// Main texture sampling function
// Returns true if texture is resident and sampled successfully
__device__ inline bool tex2D(const DeviceContext& ctx,
                             uint32_t texId,
                             float u, float v,
                             float4& result,
                             float4 defaultColor = make_float4(1.0f, 0.0f, 1.0f, 1.0f)) {
    // Bounds check first
    if (texId >= ctx.maxTextures) {
        result = defaultColor;
        return false;
    }
    
    // Now we know texId is valid - check residency
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
