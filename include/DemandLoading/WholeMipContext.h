// SPDX-License-Identifier: MIT
#pragma once

#include "DemandLoading/Contracts.h"

namespace hip_demand { namespace whole_mip_v1 {

constexpr uint32_t Version = 1;

struct Entry {
    contract_v1::PublishedTexture texture{};
    contract_v1::SamplerDesc descriptor{};
};

// A launch borrows this immutable snapshot until its consuming stream is fenced.
// Do not retain/reuse a context across prepare, resize, unload, cancel or destruction.
struct DeviceContext {
    contract_v1::AbiHeader abi{Version, sizeof(DeviceContext)};
    uint64_t incarnation = 0;
    const Entry* entries = nullptr;
    contract_v1::RequestKey* requests = nullptr;
    uint32_t* requestCount = nullptr;
    uint32_t* requestOverflow = nullptr;
    uint32_t numSamplers = 0;
    uint32_t maxRequests = 0;
};

static_assert(sizeof(Entry) == 128 && offsetof(Entry, descriptor) == 72);
static_assert(sizeof(DeviceContext) == 56 && offsetof(DeviceContext, entries) == 16);
static_assert(std::is_trivially_copyable<Entry>::value &&
              std::is_trivially_copyable<DeviceContext>::value);

} }
