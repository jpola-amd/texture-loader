// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime.h>
#include <DemandLoading/DemandTextureLoader.h>

namespace hip_demand { namespace internal {

inline bool validAnisotropy(const anisotropy_v1::Request& request) {
    namespace aniso = anisotropy_v1;
    if (request.abi.version != aniso::Version || request.abi.byteSize != sizeof(request))
        return false;
    if (request.profile == aniso::Profile::LegacyCompatibility)
        return request.maxAnisotropy == 0 && request.requirement == aniso::Requirement::AllowUnqualified;
    return (request.profile == aniso::Profile::Explicit || request.profile == aniso::Profile::Parity16) &&
           request.maxAnisotropy >= 1 && request.maxAnisotropy <= 16 &&
           (request.profile != aniso::Profile::Parity16 || request.maxAnisotropy == 16) &&
           request.requirement <= aniso::Requirement::RequireQualified;
}

inline bool validDescriptor(const TextureDesc& desc) {
    const auto validAddress = [](hipTextureAddressMode mode) {
        return mode == hipAddressModeWrap || mode == hipAddressModeClamp ||
               mode == hipAddressModeMirror || mode == hipAddressModeBorder;
    };
    const auto validFilter = [](hipTextureFilterMode mode) {
        return mode == hipFilterModePoint || mode == hipFilterModeLinear;
    };
    return validAddress(desc.addressMode[0]) && validAddress(desc.addressMode[1]) &&
           validFilter(desc.filterMode) && validFilter(desc.mipmapFilterMode) &&
           (desc.evictionPriority == EvictionPriority::Normal || desc.evictionPriority == EvictionPriority::Low ||
            desc.evictionPriority == EvictionPriority::High || desc.evictionPriority == EvictionPriority::KeepResident);
}

inline capability_v1::Failure hipFailure(capability_v1::Operation operation, hipError_t error) {
    using contract_v1::Outcome;
    Outcome outcome = Outcome::RuntimeFailure;
    if (error == hipSuccess) outcome = Outcome::Success;
    else if (error == hipErrorNotSupported) outcome = Outcome::Unsupported;
    else if (error == hipErrorOutOfMemory) outcome = Outcome::DeviceOutOfMemory;
    else if (error == hipErrorInvalidValue) outcome = Outcome::InvalidInput;
    return {outcome, operation, static_cast<int32_t>(error)};
}

inline hipTextureDesc makeSampler(const TextureDesc& desc, bool floatPixels, bool mipmapped, int levels,
                                 uint32_t maxAnisotropy) {
    hipTextureDesc sampler{};
    sampler.addressMode[0] = desc.addressMode[0];
    sampler.addressMode[1] = desc.addressMode[1];
    sampler.filterMode = desc.filterMode;
    sampler.readMode = floatPixels ? hipReadModeElementType : hipReadModeNormalizedFloat;
    sampler.normalizedCoords = desc.normalizedCoords ? 1 : 0;
    sampler.sRGB = desc.sRGB && !floatPixels ? 1 : 0;
    sampler.maxAnisotropy = maxAnisotropy;
    if (mipmapped) {
        sampler.mipmapFilterMode = desc.mipmapFilterMode;
        sampler.maxMipmapLevelClamp = static_cast<float>(levels - 1);
    }
    return sampler;
}

} }
