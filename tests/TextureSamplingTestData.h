// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace hip_demand { namespace test {

enum class SamplingPath : uint32_t {
    Implicit, Lod, Gradient, RecordRequest, NativeLod, NativeSamplerWords, NativeImageControlWord,
    // OptiX Toolkit-style entry points. PagingSampler returns (width, height, numMipLevels,
    // mipTailFirstLevel); PagingSamplerDesc returns (maxAnisotropy, wrapMode0, mipmapFilterMode,
    // texture == context.textures[id]); MipLevel returns getMipLevel(ddx, ddy, u, v, lod) in x.
    OtkImplicit, OtkLod, OtkGradient, PagingSampler, PagingSamplerDesc, MipLevel,
    // Cubic samples with the gradients and uses lod as the filter mode (2 bicubic, 3 smart bicubic).
    Cubic,
    // Gradient sample with a software mip blend; lod is the anisotropy of the footprint.
    LinearMip
};

struct SamplingInput {
    uint32_t textureId = UINT32_MAX;
    SamplingPath path = SamplingPath::Implicit;
    float u = .5f;
    float v = .5f;
    float lod = 0;
    float2 ddx{};
    float2 ddy{};
    float4 defaultColor{1, 0, 1, 1};
};

struct SamplingResult {
    float4 value{};
    uint32_t resident = 0;
};

static_assert(std::is_trivially_copyable_v<SamplingInput>);
static_assert(std::is_trivially_copyable_v<SamplingResult>);
static_assert(sizeof(SamplingInput) == 64 && offsetof(SamplingInput, defaultColor) == 48);
static_assert(sizeof(SamplingResult) == 32 && offsetof(SamplingResult, resident) == 16);

} }
