// SPDX-License-Identifier: MIT
#pragma once

// Include hip/hip_runtime.h first; compile consumers with the HIP compiler.
#include "DemandLoading/WholeMipContext.h"
#include "DemandLoading/TextureSampling.h"

namespace hip_demand { namespace whole_mip_v1 {
namespace detail {

__device__ __forceinline__ bool finite(float value) {
    return value >= -3.402823466e+38F && value <= 3.402823466e+38F;
}

__device__ inline contract_v1::Outcome lookup(
    const DeviceContext& context, contract_v1::GpuKey key, uint64_t revision, const Entry*& entry) {
    using namespace contract_v1;
    entry = nullptr;
    if (!compatible(context.abi, sizeof(whole_mip_v1::DeviceContext)))
        return Outcome::AbiMismatch;
    // Validate the complete wire header before following any pointer, including
    // the request buffers on a resident hit. These are the fixed v1 bounds.
    if (!context.incarnation || !context.entries || !context.requests ||
        !context.requestCount || !context.requestOverflow ||
        context.numSamplers > 4096 || !context.maxRequests || context.maxRequests > 1048576)
        return Outcome::InvalidInput;
    if (!valid(key) || key.incarnation != context.incarnation || key.slot >= context.numSamplers)
        return Outcome::InvalidKey;
    const Entry& candidate = context.entries[key.slot];
    const PublishedTexture& texture = candidate.texture;
    if (!(key == texture.key) || !revision || revision != texture.revision ||
        texture.state != RegistrationState::Live)
        return Outcome::InvalidKey;
    const Outcome descriptor = validate(candidate.descriptor);
    if (descriptor != Outcome::Success)
        return descriptor;
    if (texture.reserved || validate(texture.mips) != Outcome::Success)
        return Outcome::InvalidInput;
    if (texture.residency != Outcome::Success && texture.residency != Outcome::Pending &&
        texture.residency != Outcome::Deferred)
        return texture.residency;
    if (candidate.descriptor.maxAnisotropy != 1)
        return Outcome::Unsupported;
    uint32_t levels = candidate.descriptor.mipPolicy == MipPolicy::Disabled
        ? 1 : fullMipCount(texture.mips.originalWidth, texture.mips.originalHeight);
    if (candidate.descriptor.maxMipLevels && candidate.descriptor.maxMipLevels < levels)
        levels = candidate.descriptor.maxMipLevels;
    if (texture.mips.originalLevels != levels)
        return Outcome::Unsupported;
    const bool backing = texture.mips.firstResidentMip != InvalidSlot;
    if ((texture.residency == Outcome::Success && (!backing || !texture.textureObject)) ||
        (texture.residency != Outcome::Success && (backing || texture.textureObject)))
        return Outcome::InvalidTransition;
    entry = &candidate;
    return Outcome::Success;
}

__device__ __forceinline__ bool requestLeader(const contract_v1::RequestKey& request) {
    const uint64_t active = getActiveMask();
#if defined(HIP_ENABLE_WARP_SYNC_BUILTINS)
    uint64_t matches = __match_any_sync(active, request.texture.slot);
    matches &= __match_any_sync(active, request.texture.generation);
    matches &= __match_any_sync(active, uint32_t(request.texture.incarnation));
    matches &= __match_any_sync(active, uint32_t(request.texture.incarnation >> 32));
    matches &= __match_any_sync(active, uint32_t(request.revision));
    matches &= __match_any_sync(active, uint32_t(request.revision >> 32));
    matches &= __match_any_sync(active, request.originalMip);
    return getLaneId() == __ffsll(static_cast<long long>(matches)) - 1;
#else
    // All active lanes participate even when their own leader was already found.
    uint64_t remaining = active;
    int leader = getLaneId();
    while (remaining) {
        const int lane = __ffsll(static_cast<long long>(remaining)) - 1;
        const int width = getWaveSize();
        const uint32_t slot = __shfl(request.texture.slot, lane, width);
        const uint32_t generation = __shfl(request.texture.generation, lane, width);
        const uint32_t incarnationLow = __shfl(uint32_t(request.texture.incarnation), lane, width);
        const uint32_t incarnationHigh = __shfl(uint32_t(request.texture.incarnation >> 32), lane, width);
        const uint32_t revisionLow = __shfl(uint32_t(request.revision), lane, width);
        const uint32_t revisionHigh = __shfl(uint32_t(request.revision >> 32), lane, width);
        const uint32_t mip = __shfl(request.originalMip, lane, width);
        const bool same = request.texture.slot == slot && request.texture.generation == generation &&
            uint32_t(request.texture.incarnation) == incarnationLow &&
            uint32_t(request.texture.incarnation >> 32) == incarnationHigh &&
            uint32_t(request.revision) == revisionLow && uint32_t(request.revision >> 32) == revisionHigh &&
            request.originalMip == mip;
        if (same && lane < leader)
            leader = lane;
        remaining &= remaining - 1;
    }
    return getLaneId() == leader;
#endif
}

__device__ inline contract_v1::Outcome append(const DeviceContext& context,
                                              const contract_v1::RequestKey& request) {
    using contract_v1::Outcome;
    if (!requestLeader(request))
        return Outcome::Success;
    uint32_t count = __atomic_load_n(context.requestCount, __ATOMIC_RELAXED);
    for (;;) {
        if (count >= context.maxRequests) {
            atomicExch(context.requestOverflow, 1u);
            return Outcome::RequestOverflow;
        }
        const uint32_t previous = atomicCAS(context.requestCount, count, count + 1);
        if (previous == count) {
            context.requests[count] = request;
            // Host readback is ordered after the consumer-stream fence, not
            // merely an observation of the reservation counter.
            return Outcome::Success;
        }
        count = previous;
    }
}

__device__ inline void demand(const DeviceContext& context, contract_v1::GpuKey key,
                              uint64_t revision, contract_v1::SampleDecision& decision) {
    if (!decision.needsRequest())
        return;
    for (uint32_t level = decision.required.first; level <= decision.required.last; ++level)
        if (append(context, {key, revision, level, 0}) == contract_v1::Outcome::RequestOverflow)
            decision.outcome = contract_v1::Outcome::RequestOverflow;
}

} // namespace detail

__device__ inline contract_v1::Outcome recordRequest(
    const DeviceContext& context, const contract_v1::RequestKey& request) {
    using namespace contract_v1;
    const Entry* entry = nullptr;
    const Outcome outcome = detail::lookup(context, request.texture, request.revision, entry);
    if (outcome != Outcome::Success)
        return outcome;
    if (request.reserved || request.originalMip >= entry->texture.mips.originalLevels)
        return Outcome::InvalidInput;
    return detail::append(context, request);
}

__device__ inline contract_v1::SampleDecision tex2DLod(
    const DeviceContext& context, contract_v1::GpuKey key, uint64_t revision,
    float u, float v, float originalLod, float4& result,
    float4 defaultColor = make_float4(1, 0, 1, 1)) {
    using namespace contract_v1;
    result = defaultColor;
    const Entry* entry = nullptr;
    const Outcome outcome = detail::lookup(context, key, revision, entry);
    if (outcome != Outcome::Success)
        return {outcome};
    if (!detail::finite(u) || !detail::finite(v))
        return {Outcome::InvalidInput};
    SampleDecision decision = evaluate(entry->texture.mips,
        requiredLevels(originalLod, entry->texture.mips.originalLevels, entry->descriptor.mipFilter,
                       entry->descriptor.maxAnisotropy), entry->descriptor.samplingPolicy);
    detail::demand(context, key, revision, decision);
    if (decision.validity == SampleValidity::Complete || decision.validity == SampleValidity::CoarsePreview)
        result = ::tex2DLod<float4>(reinterpret_cast<hipTextureObject_t>(entry->texture.textureObject),
                                   u, v, decision.resourceLod);
    return decision;
}

__device__ inline contract_v1::SampleDecision tex2DGrad(
    const DeviceContext& context, contract_v1::GpuKey key, uint64_t revision,
    float u, float v, float2 ddx, float2 ddy, float4& result,
    float4 defaultColor = make_float4(1, 0, 1, 1)) {
    using namespace contract_v1;
    result = defaultColor;
    const Entry* entry = nullptr;
    const Outcome outcome = detail::lookup(context, key, revision, entry);
    if (outcome != Outcome::Success)
        return {outcome};
    if (!detail::finite(u) || !detail::finite(v) || !detail::finite(ddx.x) ||
        !detail::finite(ddx.y) || !detail::finite(ddy.x) || !detail::finite(ddy.y))
        return {Outcome::InvalidInput};
    const MipLayout& mips = entry->texture.mips;
    const float width = entry->descriptor.normalizedCoords ? float(mips.originalWidth) : 1.f;
    const float height = entry->descriptor.normalizedCoords ? float(mips.originalHeight) : 1.f;
    // Preserve exact ordinary integer footprints. Log-domain scaling is only
    // needed when finite extreme derivatives overflow the texel-space length.
    const float scale = fmaxf(fmaxf(fabsf(ddx.x), fabsf(ddx.y)),
                              fmaxf(fabsf(ddy.x), fabsf(ddy.y)));
    float lod = 0;
    if (scale > 0) {
        const float footprint = fmaxf(hypotf(ddx.x * width, ddx.y * height),
                                      hypotf(ddy.x * width, ddy.y * height));
        if (detail::finite(footprint) && footprint > 0)
            lod = log2f(footprint);
        else {
            const float x = hypotf(ddx.x / scale * width, ddx.y / scale * height);
            const float y = hypotf(ddy.x / scale * width, ddy.y / scale * height);
            lod = log2f(scale) + log2f(fmaxf(x, y));
        }
    }
    SampleDecision decision = evaluate(mips,
        requiredLevels(lod, mips.originalLevels, entry->descriptor.mipFilter,
                       entry->descriptor.maxAnisotropy), entry->descriptor.samplingPolicy);
    detail::demand(context, key, revision, decision);
    if (decision.validity == SampleValidity::Complete || decision.validity == SampleValidity::CoarsePreview) {
        if (entry->descriptor.normalizedCoords) {
            // Floor-rounded NPOT and saturated thin dimensions do not shrink
            // by exactly 2^firstMip. Correct just that discrepancy; native
            // gradients already account for the actual backing dimensions.
            const float mipScale = exp2f(float(mips.firstResidentMip));
            const float x = float(mips.originalWidth) / (float(mips.resourceWidth) * mipScale);
            const float y = float(mips.originalHeight) / (float(mips.resourceHeight) * mipScale);
            ddx.x *= x;
            ddy.x *= x;
            ddx.y *= y;
            ddy.y *= y;
        }
        result = ::tex2DGrad<float4>(reinterpret_cast<hipTextureObject_t>(entry->texture.textureObject),
                                    u, v, ddx, ddy);
    }
    return decision;
}

__device__ inline contract_v1::SampleDecision tex2D(
    const DeviceContext& context, contract_v1::GpuKey key, uint64_t revision,
    float u, float v, float4& result, float4 defaultColor = make_float4(1, 0, 1, 1)) {
    return tex2DLod(context, key, revision, u, v, 0, result, defaultColor);
}

} }
