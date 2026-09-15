// SPDX-License-Identifier: MIT
#include "TextureSamplingTestData.h"
#include <DemandLoading/TextureSampling.h>

extern "C" __global__ void sampleLegacyTextures(
    hip_demand::DeviceContext context, const hip_demand::test::SamplingInput* inputs,
    hip_demand::test::SamplingResult* outputs, uint32_t count) {
    using namespace hip_demand::test;
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const auto& input = inputs[index];
    float4 value = input.defaultColor;
    bool resident = false;
    switch (input.path) {
    case SamplingPath::Implicit:
        resident = hip_demand::tex2D(context, input.textureId, input.u, input.v, value, input.defaultColor);
        break;
    case SamplingPath::Lod:
        resident = hip_demand::tex2DLod(context, input.textureId, input.u, input.v, input.lod,
                                       value, input.defaultColor);
        break;
    case SamplingPath::Gradient:
        resident = hip_demand::tex2DGrad(context, input.textureId, input.u, input.v, input.ddx, input.ddy,
                                        value, input.defaultColor);
        break;
    case SamplingPath::RecordRequest:
        hip_demand::recordTextureRequest(context, input.textureId);
        break;
    case SamplingPath::NativeLod:
        value = ::tex2DLod<float4>((hipTextureObject_t)context.textures[input.textureId],
                                  input.u, input.v, input.lod);
        resident = true;
        break;
    case SamplingPath::NativeSamplerWords: {
        // Read the same SDK-defined sampler words consumed by the intrinsic.
        // This is diagnostic-only; no descriptor bits are modified.
        const auto* words = reinterpret_cast<const unsigned int*>(context.textures[input.textureId]) +
                            HIP_SAMPLER_OBJECT_OFFSET_DWORD;
        value = make_float4(__uint_as_float(words[0]), __uint_as_float(words[1]),
                            __uint_as_float(words[2]), __uint_as_float(words[3]));
        break;
    }
    case SamplingPath::NativeImageControlWord: {
        const auto* words = reinterpret_cast<const unsigned int*>(context.textures[input.textureId]);
        value = make_float4(__uint_as_float(words[5]), 0, 0, 0);
        break;
    }
    case SamplingPath::DelayedSnapshot: {
        resident = hip_demand::isTextureResident(context, input.textureId);
        const hipTextureObject_t object = (hipTextureObject_t)(resident ? context.textures[input.textureId] : 0);
        const auto start = clock64();
        while (clock64() - start < 100000000ULL) {}
        if (resident)
            value = ::tex2DLod<float4>(object, input.u, input.v, input.lod);
        break;
    }
    }
    outputs[index].value = value;
    outputs[index].resident = resident ? 1u : 0u;
}
