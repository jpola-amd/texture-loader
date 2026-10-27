#pragma once

/// @file TextureSampling.h
/// @brief Device-side texture sampling functions for demand loading.
/// @note This header requires HIP types. Include <hip/hip_runtime.h> before this header.  
///       Must be compiled with HIP compiler (hipcc) for device code.

#include "DemandLoading/DeviceContext.h"
#include "ShaderUtil/TextureUtil.h"

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
    if (texId >= ctx.maxTextures) return;
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
    
    result = ::tex2D<float4>((hipTextureObject_t)ctx.textures[texId], u, v);
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
    
    result = ::tex2DGrad<float4>((hipTextureObject_t)ctx.textures[texId], u, v, ddx, ddy);
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
    
    result = ::tex2DLod<float4>((hipTextureObject_t)ctx.textures[texId], u, v, lod);
    return true;
}

//=============================================================================
// OptiX Toolkit-style Entry Points
//=============================================================================
// These mirror the demandLoading:: device API used by OptiX Toolkit clients, so renderer code can
// keep one call shape for both backends. Deliberate differences from the OptiX Toolkit:
// - Only sampler pages (page < maxTextures) exist. Base-color pages are not published, so a
//   page at or above maxTextures returns 0 with *valid == false and records no request.
// - A resident sampler means the whole mip chain is resident (mipTailFirstLevel == 0).
// - Filtering is the hardware sampler's; no software cubic or anisotropic filtering is applied.

/// Return the sampler page for a texture, or record a request and return 0 when it is not resident.
/// The returned value is the device address of a TextureSampler, as in demandLoading::pagingMapOrRequest.
__device__ inline unsigned long long pagingMapOrRequest(const DeviceContext& ctx, unsigned int page, bool* valid) {
    if (page >= ctx.maxTextures || !ctx.samplers) {
        *valid = false;
        return 0;
    }
    if (!isTextureResident(ctx, page)) {
        recordTextureRequest(ctx, page);
        *valid = false;
        return 0;
    }
    *valid = true;
    return reinterpret_cast<unsigned long long>(&ctx.samplers[page]);
}

namespace detail {
__device__ __forceinline__ void convertType(const float4& in, float& out) { out = in.x; }
__device__ __forceinline__ void convertType(const float4& in, float2& out) { out = make_float2(in.x, in.y); }
__device__ __forceinline__ void convertType(const float4& in, float4& out) { out = in; }

// OptiX Toolkit returns this value for a sample that is not resident.
__device__ __forceinline__ float4 nonResidentColor() { return make_float4(1.0f, 0.0f, 1.0f, 0.0f); }
}  // namespace detail

/// Sample with explicit gradients. Sample is float, float2, or float4.
/// Sets *isResident and records a request when the texture is not resident.
template <class Sample>
__device__ inline Sample tex2DGrad(const DeviceContext& ctx, unsigned int textureId, float x, float y,
                                   float2 ddx, float2 ddy, bool* isResident) {
    float4 value = detail::nonResidentColor();
    *isResident = tex2DGrad(ctx, textureId, x, y, ddx, ddy, value, value);
    Sample rval;
    detail::convertType(value, rval);
    return rval;
}

/// Sample at an explicit LOD. Sample is float, float2, or float4.
template <class Sample>
__device__ inline Sample tex2DLod(const DeviceContext& ctx, unsigned int textureId, float x, float y,
                                  float lod, bool* isResident) {
    float4 value = detail::nonResidentColor();
    *isResident = tex2DLod(ctx, textureId, x, y, lod, value, value);
    Sample rval;
    detail::convertType(value, rval);
    return rval;
}

/// Sample with implicit LOD. Sample is float, float2, or float4.
template <class Sample>
__device__ inline Sample tex2D(const DeviceContext& ctx, unsigned int textureId, float x, float y,
                               bool* isResident) {
    float4 value = detail::nonResidentColor();
    *isResident = tex2D(ctx, textureId, x, y, value, value);
    Sample rval;
    detail::convertType(value, rval);
    return rval;
}

namespace detail {
__device__ __forceinline__ float4 cubicWeights(float x) {
    float4 w;
    w.x = -x * x * x + 3.0f * x * x - 3.0f * x + 1.0f;
    w.y = 3.0f * x * x * x - 6.0f * x * x + 4.0f;
    w.z = -3.0f * x * x * x + 3.0f * x * x + 3.0f * x + 1.0f;
    w.w = x * x * x;
    return w * (1.0f / 6.0f);
}

__device__ __forceinline__ float getPixelSpanForCubic(float2 ddx, float2 ddy, float width, float height) {
    const float spanX = fmaxf(fabsf(ddx.x), fabsf(ddy.x)) * width;
    const float spanY = fmaxf(fabsf(ddx.y), fabsf(ddy.y)) * height;
    return fmaxf(spanX, spanY) + 1.0e-8f;
}

// Keep the gradients usable: not zero, not shorter than a texel and not more anisotropic than 16:1.
__device__ __forceinline__ void fixGradients(float2& ddx, float2& ddy, int width, int height) {
    const float invAniso2 = 1.0f / (16.0f * 16.0f);
    const float minVal = 0.99f / fmaxf((float)width, (float)height);

    float ddx2 = ddx.x * ddx.x + ddx.y * ddx.y;
    float ddy2 = ddy.x * ddy.x + ddy.y * ddy.y;
    if (ddx2 > minVal * minVal && ddy2 > minVal * minVal && ddx2 >= invAniso2 * ddy2 &&
        ddy2 >= invAniso2 * ddx2)
        return;

    if (ddx2 == 0.0f)
        ddx = make_float2(minVal, 0.0f);
    if (ddy2 == 0.0f)
        ddy = make_float2(0.0f, minVal);

    ddx2 = ddx.x * ddx.x + ddx.y * ddx.y;
    if (ddx2 < minVal * minVal) {
        const float scale = minVal / sqrtf(ddx2);
        ddx = make_float2(ddx.x * scale, ddx.y * scale);
    }
    ddy2 = ddy.x * ddy.x + ddy.y * ddy.y;
    if (ddy2 < minVal * minVal) {
        const float scale = minVal / sqrtf(ddy2);
        ddy = make_float2(ddy.x * scale, ddy.y * scale);
    }

    ddx2 = ddx.x * ddx.x + ddx.y * ddx.y;
    ddy2 = ddy.x * ddy.x + ddy.y * ddy.y;
    if (ddx2 * invAniso2 > ddy2) {
        const float scale = sqrtf(ddy2 / (ddx2 * invAniso2));
        ddx = make_float2(ddx.x * scale, ddx.y * scale);
    }
    if (ddy2 * invAniso2 > ddx2) {
        const float scale = sqrtf(ddx2 / (ddy2 * invAniso2));
        ddy = make_float2(ddy.x * scale, ddy.y * scale);
    }
}

// Cubic B-spline filter of one mip level from four bilinear fetches.
__device__ __forceinline__ float4 cubicFetch(hipTextureObject_t texture, float s, float t, int mipLevel,
                                             int mipWidth, int mipHeight) {
    const float ts = s * mipWidth - 0.5f;
    const float tt = t * mipHeight - 0.5f;
    const float i = floorf(ts);
    const float j = floorf(tt);
    const float4 wx = cubicWeights(ts - i);
    const float4 wy = cubicWeights(tt - j);

    const float x0 = (i - 0.5f + wx.y / (wx.x + wx.y)) / mipWidth;
    const float x1 = (i + 1.5f + wx.w / (wx.z + wx.w)) / mipWidth;
    const float y0 = (j - 0.5f + wy.y / (wy.x + wy.y)) / mipHeight;
    const float y1 = (j + 1.5f + wy.w / (wy.z + wy.w)) / mipHeight;

    const float4 t0 = ::tex2DLod<float4>(texture, x0, y0, (float)mipLevel);
    const float4 t1 = ::tex2DLod<float4>(texture, x1, y0, (float)mipLevel);
    const float4 t2 = ::tex2DLod<float4>(texture, x0, y1, (float)mipLevel);
    const float4 t3 = ::tex2DLod<float4>(texture, x1, y1, (float)mipLevel);
    return t0 * ((wx.x + wx.y) * (wy.x + wy.y)) + t1 * ((wx.z + wx.w) * (wy.x + wy.y)) +
           t2 * ((wx.x + wx.y) * (wy.z + wy.w)) + t3 * ((wx.z + wx.w) * (wy.z + wy.w));
}

__device__ __forceinline__ float4 lerp4(float4 a, float4 b, float w) {
    return a + (b - a) * w;
}
}  // namespace detail

/// Cubic filtered sample with explicit gradients, as in the OptiX Toolkit's tex2DCubic.
/// filterMode uses the OptiX Toolkit numbering: 2 is bicubic, 3 is smart bicubic (cubic when
/// magnified, bilinear when minified) and anything else is plain bilinear.
/// Sets result and returns true if the texture is resident. Otherwise it records a request.
__device__ inline bool tex2DCubic(const DeviceContext& ctx, unsigned int textureId, float s, float t,
                                  float2 ddx, float2 ddy, unsigned int filterMode, float4& result,
                                  float4 defaultColor = make_float4(1.0f, 0.0f, 1.0f, 0.0f)) {
    bool resident = false;
    const auto* sampler =
        reinterpret_cast<const TextureSampler*>(pagingMapOrRequest(ctx, textureId, &resident));
    if (!resident || !sampler) {
        result = defaultColor;
        return false;
    }

    const hipTextureObject_t texture = (hipTextureObject_t)sampler->texture;
    const int texWidth = (int)sampler->width;
    const int texHeight = (int)sampler->height;
    const int numMipLevels = max((int)sampler->desc.numMipLevels, 1);
    const bool pointMips = sampler->desc.mipmapFilterMode == 0;
    const float invAnisotropy = 1.0f / (float)max((int)sampler->desc.maxAnisotropy, 1);

    const float p1 = (ddx.x * texWidth) * (ddx.x * texWidth) + (ddx.y * texHeight) * (ddx.y * texHeight);
    const float p2 = (ddy.x * texWidth) * (ddy.x * texWidth) + (ddy.y * texHeight) * (ddy.y * texHeight);
    const float maxGradientLengthInPixels = sqrtf(fmaxf(p1, p2));

    detail::fixGradients(ddx, ddy, texWidth, texHeight);

    const bool bicubic = filterMode == 2;
    float cubicBlend = bicubic ? 1.0f : 0.0f;
    if (filterMode == 3 && maxGradientLengthInPixels < 2.0f)
        cubicBlend = fminf(2.0f - maxGradientLengthInPixels, 1.0f);

    // Only bicubic picks a mip level here; smart bicubic cubic-filters the base level.
    float ml = 0.0f;
    int mipLevel = 0;
    if (bicubic) {
        ml = getMipLevel(ddx, ddy, texWidth, texHeight, invAnisotropy);
        if (pointMips)
            ml = fmaxf(0.0f, ceilf(ml - 0.5f));
        ml = fminf(ml, (float)(numMipLevels - 1));
        mipLevel = max((int)floorf(ml), 0);
    }

    result = ::tex2DGrad<float4>(texture, s, t, ddx, ddy);
    if (cubicBlend <= 0.0f)
        return true;

    float4 cubic = detail::cubicFetch(texture, s, t, mipLevel, max(texWidth >> mipLevel, 1),
                                      max(texHeight >> mipLevel, 1));
    result = detail::lerp4(result, cubic, cubicBlend);

    // Blend with the next level for bicubic
    if (!bicubic || pointMips || ml == (float)mipLevel || ml <= 0.0f)
        return true;
    ++mipLevel;
    cubic = detail::cubicFetch(texture, s, t, mipLevel, max(texWidth >> mipLevel, 1),
                               max(texHeight >> mipLevel, 1));
    result = detail::lerp4(cubic, result, (float)mipLevel - ml);
    return true;
}

/// Cubic filtered sample. Sample is float, float2, or float4. See the float4 overload.
template <class Sample>
__device__ inline Sample tex2DCubic(const DeviceContext& ctx, unsigned int textureId, float x, float y,
                                    float2 ddx, float2 ddy, unsigned int filterMode, bool* isResident) {
    float4 value = detail::nonResidentColor();
    *isResident = tex2DCubic(ctx, textureId, x, y, ddx, ddy, filterMode, value, value);
    Sample rval;
    detail::convertType(value, rval);
    return rval;
}

} // namespace hip_demand
