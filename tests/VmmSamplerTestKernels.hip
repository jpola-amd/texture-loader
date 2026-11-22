// SPDX-License-Identifier: MIT
// Kernels that sample one point per thread through the VMM sampler and through a native HIP texture.

#include "VmmSamplerTestTypes.h"
#include <DemandLoading/VmmTextureSampling.h>
#include <hip/hip_runtime.h>

using namespace hip_demand::vmm;
using namespace hip_demand::vmm::test;

extern "C" __global__ void sampleVmmLod( DeviceContext context, uint32_t textureId, const SamplePoint* points, uint32_t count, SampleResult* results )
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= count )
        return;
    const SamplePoint point    = points[i];
    bool              resident = false;
    const float4      value    = tex2DLod<float4>( context, textureId, point.x, point.y, point.lod, resident );
    results[i].value           = value;
    results[i].resident        = resident ? 1u : 0u;
}

extern "C" __global__ void sampleVmmGrad( DeviceContext context, uint32_t textureId, const SamplePoint* points, uint32_t count, SampleResult* results )
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= count )
        return;
    const SamplePoint point    = points[i];
    bool              resident = false;
    const float4      value    = tex2DGrad<float4>( context, textureId, point.x, point.y, point.ddx, point.ddy, resident );
    results[i].value           = value;
    results[i].resident        = resident ? 1u : 0u;
}

extern "C" __global__ void sampleVmmUdim( DeviceContext context, uint32_t textureId, uint32_t channel, const SamplePoint* points, uint32_t count, SampleResult* results )
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= count )
        return;
    const SamplePoint point    = points[i];
    bool              resident = false;
    const float4      value    = tex2DGradUdim<float4>( context, textureId, channel, point.x, point.y, point.ddx, point.ddy, resident );
    results[i].value           = value;
    results[i].resident        = resident ? 1u : 0u;
}

extern "C" __global__ void sampleNativeLod( hipTextureObject_t texture, const SamplePoint* points, uint32_t count, SampleResult* results )
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= count )
        return;
    const SamplePoint point = points[i];
    results[i].value        = tex2DLod<float4>( texture, point.x, point.y, point.lod );
    results[i].resident     = 1u;
}

extern "C" __global__ void sampleNativeGrad( hipTextureObject_t texture, const SamplePoint* points, uint32_t count, SampleResult* results )
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= count )
        return;
    const SamplePoint point = points[i];
    results[i].value        = tex2DGrad<float4>( texture, point.x, point.y, point.ddx, point.ddy );
    results[i].resident     = 1u;
}
