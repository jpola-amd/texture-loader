// SPDX-License-Identifier: MIT
#include "WholeMipTestData.h"
#include <DemandLoading/WholeMipSampling.h>

extern "C" __global__ void sampleWholeMipTextures(
    hip_demand::whole_mip_v1::DeviceContext context, const hip_demand::test::WholeMipInput* inputs,
    hip_demand::test::WholeMipResult* outputs, uint32_t count) {
    using namespace hip_demand;
    using namespace hip_demand::test;
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const WholeMipInput input = inputs[index];
    WholeMipResult result;
    result.value = input.defaultColor;
    result.waveSize = getWaveSize();
    switch (input.path) {
    case WholeMipPath::Lod:
        result.decision = whole_mip_v1::tex2DLod(context, input.key, input.revision,
            input.u, input.v, input.lod, result.value, input.defaultColor);
        break;
    case WholeMipPath::Gradient:
        result.decision = whole_mip_v1::tex2DGrad(context, input.key, input.revision,
            input.u, input.v, input.ddx, input.ddy, result.value, input.defaultColor);
        break;
    case WholeMipPath::Implicit:
        result.decision = whole_mip_v1::tex2D(context, input.key, input.revision,
            input.u, input.v, result.value, input.defaultColor);
        break;
    case WholeMipPath::RecordRequest:
        result.decision.outcome = whole_mip_v1::recordRequest(context,
            {input.key, input.revision, input.originalMip, input.requestReserved});
        break;
    case WholeMipPath::RecordLocalRequest: {
        // Synthetic per-lane snapshots permit valid requests differing in every
        // identity field even though this prototype never recycles real slots.
        if (input.key.slot >= 2) {
            result.decision.outcome = contract_v1::Outcome::InvalidKey;
            break;
        }
        whole_mip_v1::Entry local[2];
        local[input.key.slot].texture.key = input.key;
        local[input.key.slot].texture.revision = input.revision;
        local[input.key.slot].texture.state = contract_v1::RegistrationState::Live;
        local[input.key.slot].texture.mips = {16, 16, 5};
        context.incarnation = input.key.incarnation;
        context.entries = local;
        context.numSamplers = 2;
        result.decision.outcome = whole_mip_v1::recordRequest(context,
            {input.key, input.revision, input.originalMip, input.requestReserved});
        break;
    }
    case WholeMipPath::DelayedSnapshot: {
        const whole_mip_v1::Entry* entry = nullptr;
        const auto outcome = whole_mip_v1::detail::lookup(context, input.key, input.revision, entry);
        result.decision.outcome = outcome;
        if (outcome != contract_v1::Outcome::Success)
            break;
        const auto snapshot = *entry;
        result.decision = contract_v1::evaluate(snapshot.texture.mips,
            contract_v1::requiredLevels(input.lod, snapshot.texture.mips.originalLevels,
                                       snapshot.descriptor.mipFilter), snapshot.descriptor.samplingPolicy);
        const auto start = clock64();
        while (clock64() - start < 100000000ULL) {}
        if (result.decision.contributesStrictSample())
            result.value = ::tex2DLod<float4>(reinterpret_cast<hipTextureObject_t>(snapshot.texture.textureObject),
                input.u, input.v, result.decision.resourceLod);
        break;
    }
    case WholeMipPath::Inactive:
        break;
    case WholeMipPath::WaveLeader: {
        const auto mask = getActiveMask();
        const auto match = __match_any_sync(mask, input.key.slot);
        result.value = make_float4(float(getLaneId()), float(__popcll(mask)),
            whole_mip_v1::detail::requestLeader({input.key, input.revision, input.originalMip, 0}) ? 1.f : 0.f,
            float(__popcll(match)));
        result.decision.outcome = whole_mip_v1::detail::append(context,
            {input.key, input.revision, input.originalMip, 0});
        const auto pointer = reinterpret_cast<uint64_t>(context.requestCount);
        result.decision.required = {uint32_t(pointer), uint32_t(pointer >> 32)};
        break;
    }
    case WholeMipPath::NativeGradient:
        result.value = ::tex2DGrad<float4>(
            reinterpret_cast<hipTextureObject_t>(context.entries[input.key.slot].texture.textureObject),
            input.u, input.v, input.ddx, input.ddy);
        result.decision.outcome = contract_v1::Outcome::Success;
        result.decision.validity = contract_v1::SampleValidity::Complete;
        break;
    }
    outputs[index] = result;
}
