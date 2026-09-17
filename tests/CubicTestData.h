// SPDX-License-Identifier: MIT
#pragma once
#include <DemandLoading/CubicContext.h>
#include <hip/hip_runtime.h>
namespace hip_demand {
namespace test {
struct CubicInput
{
    contract_v1::GpuKey key{};
    uint64_t            revision = 1;
    cubic_v1::Config    config{};
    float               s = .4375f, t = .4375f, lod = 0;
    float2              dx{}, dy{}, jitter{};
    uint32_t            gradient = 0, outputs = 7;
    uint32_t            nativeControl = 0;
};
struct CubicOutput
{
    float4                      value{}, ds{}, dt{};
    contract_v1::SampleDecision decision{};
    double                      lod = 0, weight = 0;
    uint32_t                    low = 0, high = 0;
    float2                      dx{}, dy{};
    float4                      nativeValue{};
};
}  // namespace test
}  // namespace hip_demand
