// SPDX-License-Identifier: MIT
// Types shared by the sampler test kernels and the tests that launch them.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace hip_demand::vmm::test {

struct SamplePoint
{
    float  x   = 0.0f;
    float  y   = 0.0f;
    float  lod = 0.0f;
    float2 ddx = { 0.0f, 0.0f };
    float2 ddy = { 0.0f, 0.0f };
};

struct SampleResult
{
    float4   value    = { 0.0f, 0.0f, 0.0f, 0.0f };
    uint32_t resident = 0;
    uint32_t padding[3]{};
};

}  // namespace hip_demand::vmm::test
