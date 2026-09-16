// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime.h>
#include <DemandLoading/WholeMipContext.h>

namespace hip_demand { namespace test {

enum class WholeMipPath : uint32_t {
    Lod, Gradient, Implicit, RecordRequest, RecordLocalRequest, DelayedSnapshot, Inactive, WaveLeader,
    NativeGradient
};

struct WholeMipInput {
    contract_v1::GpuKey key{};
    uint64_t revision = 1;
    WholeMipPath path = WholeMipPath::Lod;
    uint32_t originalMip = 0;
    float u = .5f, v = .5f, lod = 0;
    float2 ddx{}, ddy{};
    float4 defaultColor{1, 0, 1, 1};
    uint32_t requestReserved = 0;
};

struct WholeMipResult {
    float4 value{};
    contract_v1::SampleDecision decision{};
    uint32_t waveSize = 0;
};

static_assert(std::is_trivially_copyable<WholeMipInput>::value);
static_assert(std::is_trivially_copyable<WholeMipResult>::value);
static_assert(sizeof(WholeMipInput) == 96 && offsetof(WholeMipInput, defaultColor) == 64);
static_assert(sizeof(WholeMipResult) == 48 && offsetof(WholeMipResult, decision) == 16);

} }
