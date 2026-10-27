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
    case SamplingPath::OtkImplicit:
        value = hip_demand::tex2D<float4>(context, input.textureId, input.u, input.v, &resident);
        break;
    case SamplingPath::OtkLod:
        value = hip_demand::tex2DLod<float4>(context, input.textureId, input.u, input.v, input.lod, &resident);
        break;
    case SamplingPath::OtkGradient:
        value = hip_demand::tex2DGrad<float4>(context, input.textureId, input.u, input.v, input.ddx, input.ddy,
                                              &resident);
        break;
    case SamplingPath::PagingSampler:
    case SamplingPath::PagingSamplerDesc: {
        const auto* sampler = reinterpret_cast<const hip_demand::TextureSampler*>(
            hip_demand::pagingMapOrRequest(context, input.textureId, &resident));
        if (resident && sampler) {
            if (input.path == SamplingPath::PagingSampler)
                value = make_float4(sampler->width, sampler->height, sampler->desc.numMipLevels,
                                    sampler->mipTailFirstLevel);
            else
                value = make_float4(sampler->desc.maxAnisotropy, sampler->desc.wrapMode0,
                                    sampler->desc.mipmapFilterMode,
                                    sampler->texture == context.textures[input.textureId] ? 1.f : 0.f);
        }
        break;
    }
    case SamplingPath::MipLevel:
        value = make_float4(getMipLevel(input.ddx, input.ddy, static_cast<int>(input.u),
                                        static_cast<int>(input.v), input.lod), 0, 0, 0);
        resident = true;
        break;
    case SamplingPath::Cubic:
        resident = hip_demand::tex2DCubic(context, input.textureId, input.u, input.v, input.ddx, input.ddy,
                                          static_cast<unsigned int>(input.lod), value, input.defaultColor);
        break;
    }
    outputs[index].value = value;
    outputs[index].resident = resident ? 1u : 0u;
}
