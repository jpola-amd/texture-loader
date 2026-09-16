// SPDX-License-Identifier: MIT
#pragma once

#include "DemandLoading/DemandTextureLoader.h"
#include "DemandLoading/WholeMipContext.h"

namespace hip_demand { namespace whole_mip_v1 {

struct Options {
    contract_v1::AbiHeader abi{Version, sizeof(Options)};
    uint32_t maxSamplers = 16;  // 1..4096; fixed prototype metadata capacity
    uint32_t maxRequests = 128; // 1..1048576; overflow is reported, never retried implicitly
    uint64_t maxManagedBytes = 512ULL * 1024 * 1024;
    uint64_t maxDecodedBytes = 256ULL * 1024 * 1024;
    uint64_t maxPinnedBytes = 64ULL * 1024 * 1024;
    capability_v1::MipPolicy mipPolicy = capability_v1::MipPolicy::Required;
    uint32_t reserved = 0;
};

struct Status {
    contract_v1::AbiHeader abi{Version, sizeof(Status)};
    contract_v1::Outcome initialization = contract_v1::Outcome::Pending;
    contract_v1::Outcome operation = contract_v1::Outcome::Pending;
    capability_v1::Failure primary{}, cleanup{};
    contract_v1::MipLayout mips{};
    uint32_t desiredFirstMip = UINT32_MAX;
    uint32_t numSamplers = 0;
    uint64_t residentBytes = 0, pendingBytes = 0, retiringBytes = 0;
    uint64_t overheadBytes = 0, managedPeakBytes = 0;
    uint64_t decodedPeakBytes = 0, pinnedBytes = 0, pinnedPeakBytes = 0;
    uint64_t sourceBytes = 0, uploadedBytes = 0;
    uint64_t authoredReads = 0, generatedLevels = 0, replacements = 0;
    uint32_t requestCount = 0, requestOverflow = 0, rejectedRequests = 0;
    // Native sampler zero is an explicitly identified legacy setting, not
    // qualification of anisotropy=1 or of native linear mip interpolation.
    uint32_t submittedMaxAnisotropy = 0;
    capability_v1::Status device{};
};

// Isolated item-07 prototype: one immutable source/storage policy, several
// compatible samplers, one consuming stream, serialized host operations.
// Register variants before loading. There are no leases, ID reuse, eviction,
// automatic retries, base fallback, multistream scheduling or Arnold integration.
// Caller-owned source/decoder caches and opaque HIP overhead are excluded from
// the hard managed-payload/decoded/pinned limits and must be bounded externally.
class Texture {
public:
    Texture(std::shared_ptr<ImageSource> source, const TextureDesc& storageDescriptor = {},
            const Options& options = {});
    ~Texture();
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    contract_v1::RegistrationResult addSampler(const TextureDesc& descriptor,
        contract_v1::SamplingPolicy sampling = contract_v1::SamplingPolicy::Strict,
        const anisotropy_v1::Request& anisotropy = anisotropy_v1::Request::legacy());

    // Ends at the policy-limited last original mip. Zero retains the full-chain
    // small-texture path; a one-level suffix uses an ordinary array.
    // A prepared epoch with no requests is fenced/closed. Recorded requests
    // must first be drained with processRequests; resize never drops them.
    contract_v1::Outcome resize(uint32_t firstOriginalMip);
    contract_v1::Outcome unload();
    // prepare publishes/reset requests only after the previous batch was drained.
    // Submit consumers on stream before processRequests (or resize/unload).
    // No concurrent consumer submission is allowed during any host operation.
    contract_v1::Outcome prepare(hipStream_t stream, DeviceContext& context);
    contract_v1::Outcome processRequests();
    // Cancels blocked/active reads cooperatively, prevents later publication,
    // then waits outside metadata locks for work and consumers before teardown.
    contract_v1::Outcome cancel();
    contract_v1::Outcome collectRetired();
    contract_v1::Outcome getStatus(Status& status) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} }
