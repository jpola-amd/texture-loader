// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace hip_demand { namespace test {

enum class SamplingPath : uint32_t {
    Implicit, Lod, Gradient, RecordRequest, NativeLod, NativeSamplerWords, NativeImageControlWord,
    DelayedSnapshot
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
