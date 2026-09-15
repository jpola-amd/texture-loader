// SPDX-License-Identifier: MIT
#pragma once
#include <DemandLoading/Contracts.h>

struct ContractDeviceCase {
    hip_demand::contract_v1::PublishedTexture texture{};
    hip_demand::contract_v1::GpuKey key{};
    uint64_t revision = 1;
    float lod = 0;
    hip_demand::contract_v1::SamplerDesc descriptor{};
    uint64_t incarnation = 1;
    uint32_t contextVersion = hip_demand::contract_v1::Version;
    uint32_t contextBytes = sizeof(hip_demand::contract_v1::DeviceContext);
};

struct ContractDeviceResult {
    hip_demand::contract_v1::SampleDecision decision{};
    hip_demand::contract_v1::AbiInfo abi{};
    uint32_t needsRequest = 0;
    uint32_t contributes = 0;
};
