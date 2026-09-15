// SPDX-License-Identifier: MIT
#include "TestUtils.h"
#include "ImageDataTestUtils.h"
#include "TextureSamplingHarness.h"
#include <DemandLoading/Internal/HipCalls.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>
#include <tuple>

#ifdef USE_OIIO
#include <OpenImageIO/imageio.h>
#endif

namespace hip_demand { namespace test {
namespace {
namespace cap = capability_v1;
using contract_v1::Outcome;
using internal::HipFaultState;
using internal::HipOperation;

// Authored constants, point mip selection and forced-base sampling have the
// same fixed absolute tolerance, independent of the platform under test.
constexpr float CapabilityPixelTolerance = 1e-6f;

struct CapabilityFaults {
    std::shared_ptr<HipFaultState> state = std::make_shared<HipFaultState>();
    CapabilityFaults() { internal::setHipFaultState(state); }
    ~CapabilityFaults() { internal::setHipFaultState(nullptr); }
    size_t calls(HipOperation operation, bool successful = false) const {
        size_t count = 0;
        for (const auto& record : state->records())
            if (record.operation == operation && (!successful || record.error == hipSuccess))
                ++count;
        return count;
    }
};

LoaderOptions capabilityOptions() {
    LoaderOptions options;
    options.maxTextures = 32;
    options.maxRequestsPerLaunch = 64;
    options.maxThreads = 4;
    options.enableEviction = false;
    return options;
}

TextureDesc capabilityDesc(bool generate = true) {
    TextureDesc desc;
    desc.filterMode = hipFilterModePoint;
    desc.mipmapFilterMode = hipFilterModePoint;
    desc.generateMipmaps = generate;
    return desc;
}

cap::Policy policyFor(cap::MipPolicy policy) {
    cap::Policy result;
    result.mipPolicy = policy;
    return result;
}

Outcome classified(hipError_t error) {
    if (error == hipErrorNotSupported) return Outcome::Unsupported;
    if (error == hipErrorOutOfMemory) return Outcome::DeviceOutOfMemory;
    if (error == hipErrorInvalidValue) return Outcome::InvalidInput;
    return Outcome::RuntimeFailure;
}

void expectFailure(const cap::Failure& failure, cap::Operation operation, hipError_t error) {
    EXPECT_EQ(failure.operation, operation);
    EXPECT_EQ(failure.rawHipError, static_cast<int32_t>(error));
    EXPECT_EQ(failure.outcome, classified(error));
}

void expectPixel(const SamplingResult& sample, const std::array<float, 4>& expected) {
    EXPECT_EQ(sample.resident, 1u);
    const std::array<float, 4> values{sample.value.x, sample.value.y, sample.value.z, sample.value.w};
    for (size_t channel = 0; channel < 4; ++channel) {
        EXPECT_TRUE(std::isfinite(values[channel]));
        EXPECT_NEAR(values[channel], expected[channel], CapabilityPixelTolerance) << "channel=" << channel;
    }
}

class Capability : public HipTestFixture {
protected:
    void queue(DemandTextureLoader& loader, const std::vector<uint32_t>& ids) {
        loader.launchPrepare();
        const auto context = loader.getDeviceContext();
        const uint32_t count = static_cast<uint32_t>(ids.size());
        ASSERT_LE(count, context.maxRequests);
        ASSERT_EQ(hipMemcpy(context.requests, ids.data(), ids.size() * sizeof(uint32_t),
                            hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(context.requestCount, &count, sizeof(count), hipMemcpyHostToDevice), hipSuccess);
    }
    void request(DemandTextureLoader& loader, const std::vector<uint32_t>& ids, size_t expected) {
        queue(loader, ids);
        ASSERT_EQ(loader.processRequests(nullptr, loader.getDeviceContext()), expected);
        loader.launchPrepare();
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    }
    cap::Status status(DemandTextureLoader& loader, uint32_t id) {
        cap::Status result;
        EXPECT_EQ(loader.getTextureStatusV1(id, result), Outcome::Success);
        return result;
    }
    hipTextureObject_t object(DemandTextureLoader& loader, uint32_t id) {
        hipTextureObject_t result = 0;
        EXPECT_EQ(hipMemcpy(&result, loader.getDeviceContext().textures + id, sizeof(result),
                            hipMemcpyDeviceToHost), hipSuccess);
        return result;
    }
    hipResourceDesc resource(DemandTextureLoader& loader, uint32_t id) {
        hipResourceDesc result{};
        const auto sampler = object(loader, id);
        EXPECT_NE(sampler, hipTextureObject_t{});
        if (sampler)
            EXPECT_EQ(hipGetTextureObjectResourceDesc(&result, sampler), hipSuccess);
        return result;
    }
    void expectResident(DemandTextureLoader& loader, uint32_t id, unsigned int levels = 4) {
        const auto result = status(loader, id);
        EXPECT_EQ(result.state, cap::State::Resident);
        EXPECT_EQ(result.primary.outcome, Outcome::Success);
        EXPECT_EQ(result.cleanup.outcome, Outcome::Success);
        EXPECT_EQ(result.resource, levels > 1 ? cap::Resource::MipmappedArray : cap::Resource::Array);
        EXPECT_EQ(result.resourceLevels, levels);
        EXPECT_EQ(result.firstResidentMip, 0u);
        EXPECT_EQ(result.lastResidentMip, levels - 1);
        EXPECT_EQ(result.submitted, 1u);
        EXPECT_EQ(result.returned, 1u);
        EXPECT_EQ(result.published, 1u);
        EXPECT_NE(result.capability, cap::Support::BehaviorQualified);
        EXPECT_NE(object(loader, id), hipTextureObject_t{});
    }
    void expectUnpublished(DemandTextureLoader& loader, uint32_t id) {
        EXPECT_EQ(object(loader, id), hipTextureObject_t{});
        EXPECT_EQ(status(loader, id).published, 0u);
    }
};

TEST_F(Capability, DefaultPoliciesRegistrationStatusAndSamplerIdentity) {
    CapabilityFaults faults;
    DemandTextureLoader loader(capabilityOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
    const auto desc = capabilityDesc();
    const auto legacy = loader.createTexture(source, desc);
    const auto required = loader.createTextureV1(source, desc);
    const auto allowed = loader.createTextureV1(source, desc, policyFor(cap::MipPolicy::AllowBaseLevelFallback));
    const auto disabled = loader.createTextureV1(source, desc, policyFor(cap::MipPolicy::Disabled));
    for (const auto& handle : {legacy, required, allowed, disabled}) ASSERT_TRUE(handle.valid);
    EXPECT_NE(legacy.id, required.id);
    EXPECT_NE(required.id, allowed.id);
    EXPECT_NE(allowed.id, disabled.id);
    EXPECT_EQ(loader.createTextureV1(source, desc).id, required.id);
    EXPECT_EQ(loader.createTextureV1(source, desc, policyFor(cap::MipPolicy::LegacyCompatibility)).id, legacy.id);
    const auto before = status(loader, required.id);
    EXPECT_EQ(before.state, cap::State::Registered);
    EXPECT_EQ(before.policy, cap::MipPolicy::Required);
    EXPECT_EQ(status(loader, legacy.id).policy, cap::MipPolicy::LegacyCompatibility);
    EXPECT_EQ(before.capability, cap::Support::Unknown);
    EXPECT_EQ(before.resource, cap::Resource::None);
    EXPECT_EQ(before.attempts, 0u);
    EXPECT_EQ(before.submitted, 0u);
    EXPECT_EQ(before.returned, 0u);
    EXPECT_EQ(before.published, 0u);
    EXPECT_TRUE(before.requested == desc);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 0u);
    EXPECT_EQ(source->reads, 0u);
    request(loader, {required.id, allowed.id}, 2);
    expectResident(loader, required.id);
    expectResident(loader, allowed.id);
    EXPECT_EQ(resource(loader, required.id).res.mipmap.mipmap, resource(loader, allowed.id).res.mipmap.mipmap);
    EXPECT_NE(object(loader, required.id), object(loader, allowed.id));
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), 1u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 340u);
}

TEST_F(Capability, StatusRejectsBadAbiAndUnknownIdsWithoutWritingAnyBytes) {
    DemandTextureLoader loader(capabilityOptions());
    const auto handle = loader.createTextureV1(std::make_shared<TypedImageSource>(makeAuthoredMipSource(false)),
                                               capabilityDesc());
    ASSERT_TRUE(handle.valid);
    for (unsigned int invalid = 0; invalid < 5; ++invalid) {
        cap::Status output;
        output.textureId = 12345;
        output.attempts = UINT64_MAX;
        output.requested.maxMipLevel = 17;
        uint32_t id = handle.id;
        Outcome expected = Outcome::AbiMismatch;
        if (invalid == 0) output.abi.version = 0;
        if (invalid == 1) output.abi.byteSize = sizeof(output) - 1;
        if (invalid == 2) output.abi.byteSize = sizeof(output) + 1;
        if (invalid >= 3) {
            id = invalid == 3 ? InvalidTextureId : handle.id + 1;
            expected = Outcome::InvalidKey;
        }
        std::array<unsigned char, sizeof(output)> saved{};
        std::memcpy(saved.data(), &output, sizeof(output));
        EXPECT_EQ(loader.getTextureStatusV1(id, output), expected);
        EXPECT_EQ(std::memcmp(saved.data(), &output, sizeof(output)), 0);
    }
}

TEST_F(Capability, MalformedPolicyAndMemoryInputsNeverConsumeTextureZero) {
    DemandTextureLoader loader(capabilityOptions());
    const auto desc = capabilityDesc();
    std::array<unsigned char, 4> pixel{17, 53, 101, 239};
    for (unsigned int invalid = 0; invalid < 4; ++invalid) {
        cap::Policy policy;
        if (invalid == 0) policy.abi.version = 2;
        if (invalid == 1) policy.abi.byteSize = sizeof(policy) - 1;
        if (invalid == 2) policy.abi.byteSize = sizeof(policy) + 1;
        if (invalid == 3) policy.mipPolicy = static_cast<cap::MipPolicy>(UINT32_MAX);
        for (const auto& handle : {
                 loader.createTextureV1("capability-invalid-policy.png", desc, policy),
                 loader.createTextureV1(std::make_shared<TypedImageSource>(makeAuthoredMipSource(false)), desc, policy),
                 loader.createTextureFromMemoryV1(pixel.data(), 1, 1, 4, desc, policy)}) {
            EXPECT_FALSE(handle.valid);
            EXPECT_EQ(handle.id, InvalidTextureId);
            EXPECT_EQ(handle.error, LoaderError::InvalidParameter);
        }
    }
    for (const auto& handle : {
             loader.createTextureFromMemoryV1(nullptr, 1, 1, 4, desc),
             loader.createTextureFromMemoryV1(pixel.data(), 0, 1, 4, desc),
             loader.createTextureFromMemoryV1(pixel.data(), 1, -1, 4, desc),
             loader.createTextureFromMemoryV1(pixel.data(), 1, 1, 0, desc),
             loader.createTextureFromMemoryV1(pixel.data(), 1, 1, 5, desc)}) {
        EXPECT_FALSE(handle.valid);
        EXPECT_EQ(handle.id, InvalidTextureId);
        EXPECT_EQ(handle.error, LoaderError::InvalidParameter);
    }
    const auto valid = loader.createTextureFromMemoryV1(pixel.data(), 1, 1, 4, desc);
    ASSERT_TRUE(valid.valid);
    EXPECT_EQ(valid.id, 0u);
    EXPECT_EQ(status(loader, valid.id).policy, cap::MipPolicy::Required);
    request(loader, {valid.id}, 1);
    expectResident(loader, valid.id, 1);
    EXPECT_EQ(status(loader, valid.id).reason, cap::Reason::Singleton);
}

class CapabilityIntentionalBase : public Capability, public testing::WithParamInterface<int> {};

TEST_P(CapabilityIntentionalBase, IntentionalSingleLevelDoesNotProbeOrReportDegradation) {
    CapabilityFaults faults;
    faults.state->fail(HipOperation::ProbeAllocate, hipErrorNotSupported);
    DemandTextureLoader loader(capabilityOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
    auto desc = capabilityDesc();
    auto policy = policyFor(cap::MipPolicy::Required);
    cap::Reason reason = cap::Reason::None;
    switch (GetParam()) {
    case 0: policy.mipPolicy = cap::MipPolicy::Disabled; reason = cap::Reason::Disabled; break;
    case 1: desc.maxMipLevel = 1; reason = cap::Reason::LevelLimit; break;
    case 2:
        desc.generateMipmaps = false;
        policy.mipPolicy = cap::MipPolicy::LegacyCompatibility;
        reason = cap::Reason::LegacyBaseOnly;
        break;
    case 3:
        source->info.width = source->info.height = source->info.numMipLevels = 1;
        source->pixels.resize(4);
        source->mipPixels = {source->pixels};
        reason = cap::Reason::Singleton;
        break;
    case 4:
        desc.maxMipLevel = 1;
        policy.mipPolicy = cap::MipPolicy::LegacyCompatibility;
        reason = cap::Reason::LevelLimit;
        break;
    }
    const auto handle = loader.createTextureV1(source, desc, policy);
    ASSERT_TRUE(handle.valid);
    request(loader, {handle.id}, 1);
    expectResident(loader, handle.id, 1);
    const auto result = status(loader, handle.id);
    EXPECT_EQ(result.reason, reason);
    EXPECT_EQ(result.fallback.outcome, Outcome::Success);
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{0}));
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 0u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), 0u);
}

INSTANTIATE_TEST_SUITE_P(DisabledLimitLegacySingleton, CapabilityIntentionalBase, testing::Range(0, 5));

class CapabilityMissingAuthored : public Capability,
    public testing::WithParamInterface<std::tuple<cap::MipPolicy, unsigned int>> {};

TEST_P(CapabilityMissingAuthored, MissingRequiredAuthoredLevelsNeverGenerateOrFallback) {
    const auto [policy, authoredLevels] = GetParam();
    CapabilityFaults faults;
    DemandTextureLoader loader(capabilityOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
    source->info.numMipLevels = authoredLevels;
    source->mipPixels.resize(authoredLevels);
    const auto handle = loader.createTextureV1(source, capabilityDesc(false), policyFor(policy));
    ASSERT_TRUE(handle.valid);
    request(loader, {handle.id}, 0);
    const auto result = status(loader, handle.id);
    EXPECT_EQ(result.state, cap::State::Failed);
    EXPECT_EQ(result.primary.outcome, Outcome::Unsupported);
    EXPECT_EQ(result.primary.operation, cap::Operation::SourceRead);
    EXPECT_EQ(result.primary.rawHipError, 0);
    EXPECT_EQ(loader.getLastError(), LoaderError::Unsupported);
    EXPECT_EQ(result.fallback.outcome, Outcome::Success);
    EXPECT_EQ(result.capability, cap::Support::Unknown);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 0u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateArray), 0u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), 0u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    expectUnpublished(loader, handle.id);
}

INSTANTIATE_TEST_SUITE_P(RequiredAndAllowed, CapabilityMissingAuthored,
    testing::Combine(testing::Values(cap::MipPolicy::Required, cap::MipPolicy::AllowBaseLevelFallback),
                     testing::Values(1u, 3u)));

struct CapabilityFaultCase {
    HipOperation seam;
    cap::Operation operation;
    const char* name;
};

const std::array<CapabilityFaultCase, 4> probeCases{{
    {HipOperation::ProbeAllocate, cap::Operation::ProbeAllocate, "Allocate"},
    {HipOperation::ProbeGetLevel, cap::Operation::ProbeGetLevel, "GetLevel"},
    {HipOperation::ProbeUpload, cap::Operation::ProbeUpload, "Upload"},
    {HipOperation::ProbeCreateSampler, cap::Operation::ProbeCreateSampler, "Sampler"}
}};

const std::array<hipError_t, 4> capabilityErrors{
    hipErrorNotSupported, hipErrorOutOfMemory, hipErrorInvalidValue, hipErrorInvalidContext
};

class CapabilityProbeFailure : public Capability,
    public testing::WithParamInterface<std::tuple<CapabilityFaultCase, hipError_t, cap::MipPolicy>> {};

TEST_P(CapabilityProbeFailure, ClassifiesCachesOnlyUnsupportedAndRetriesTransientFailures) {
    const auto [fault, error, policy] = GetParam();
    CapabilityFaults faults;
    faults.state->fail(fault.seam, error);
    DemandTextureLoader loader(capabilityOptions());
    const auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
    const auto handle = policy == cap::MipPolicy::LegacyCompatibility
        ? loader.createTexture(source, capabilityDesc())
        : loader.createTextureV1(source, capabilityDesc(), policyFor(policy));
    ASSERT_TRUE(handle.valid);
    const bool fallback = error == hipErrorNotSupported && fault.seam != HipOperation::ProbeCreateSampler &&
                          policy != cap::MipPolicy::Required;
    request(loader, {handle.id}, fallback ? 1 : 0);
    const auto failed = status(loader, handle.id);
    EXPECT_EQ(failed.state, fallback ? cap::State::Degraded : cap::State::Failed);
    EXPECT_EQ(failed.capability, error == hipErrorNotSupported ? cap::Support::Unsupported : cap::Support::Unknown);
    expectFailure(fallback ? failed.fallback : failed.primary, fault.operation, error);
    EXPECT_EQ(failed.cleanup.outcome, Outcome::Success);
    EXPECT_EQ(failed.attempts, 1u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), 0u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateArray), fallback ? 1u : 0u);
    if (fallback) {
        EXPECT_EQ(failed.primary.outcome, Outcome::Success);
        EXPECT_EQ(failed.reason, cap::Reason::CapabilityFallback);
        EXPECT_EQ(failed.originalWidth, 8u);
        EXPECT_EQ(failed.originalHeight, 8u);
        EXPECT_EQ(failed.originalLevels, 4u);
        EXPECT_EQ(failed.resourceWidth, 8u);
        EXPECT_EQ(failed.resourceHeight, 8u);
        EXPECT_EQ(failed.resourceLevels, 1u);
        EXPECT_EQ(failed.firstResidentMip, 0u);
        EXPECT_EQ(failed.lastResidentMip, 0u);
        EXPECT_EQ(failed.payloadBytes, 256u);
        loader.unloadTexture(handle.id);
    } else {
        EXPECT_EQ(loader.getLastError(), error == hipErrorOutOfMemory ? LoaderError::OutOfMemory : LoaderError::HipError);
        expectUnpublished(loader, handle.id);
        EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    }
    const size_t probeAllocations = faults.calls(HipOperation::ProbeAllocate);
    request(loader, {handle.id}, error != hipErrorNotSupported || fallback ? 1 : 0);
    if (error == hipErrorNotSupported) {
        EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), probeAllocations);
        EXPECT_EQ(status(loader, handle.id).state, fallback ? cap::State::Degraded : cap::State::Failed);
    } else {
        expectResident(loader, handle.id);
        EXPECT_EQ(status(loader, handle.id).capability, cap::Support::OperationSupported);
        EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), probeAllocations + 1);
    }
    EXPECT_EQ(status(loader, handle.id).attempts, 2u);
}

INSTANTIATE_TEST_SUITE_P(OperationErrorPolicy, CapabilityProbeFailure,
    testing::Combine(testing::ValuesIn(probeCases), testing::ValuesIn(capabilityErrors),
                     testing::Values(cap::MipPolicy::LegacyCompatibility, cap::MipPolicy::Required,
                                     cap::MipPolicy::AllowBaseLevelFallback)));

class CapabilityAllocationFailure : public Capability,
    public testing::WithParamInterface<std::tuple<hipError_t, cap::MipPolicy>> {};

TEST_P(CapabilityAllocationFailure, ActualFailureNeverPoisonsProbeAndOnlyExplicitUnsupportedMayFallback) {
    const auto [error, policy] = GetParam();
    CapabilityFaults faults;
    faults.state->fail(HipOperation::AllocateMipmapped, error);
    DemandTextureLoader loader(capabilityOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
    const auto handle = policy == cap::MipPolicy::LegacyCompatibility
        ? loader.createTexture(source, capabilityDesc())
        : loader.createTextureV1(source, capabilityDesc(), policyFor(policy));
    ASSERT_TRUE(handle.valid);
    const bool fallback = error == hipErrorNotSupported && policy == cap::MipPolicy::AllowBaseLevelFallback;
    request(loader, {handle.id}, fallback ? 1 : 0);
    const auto result = status(loader, handle.id);
    EXPECT_EQ(result.capability, cap::Support::OperationSupported);
    EXPECT_EQ(result.state, fallback ? cap::State::Degraded : cap::State::Failed);
    expectFailure(fallback ? result.fallback : result.primary, cap::Operation::AllocateMipmapped, error);
    EXPECT_EQ(faults.calls(HipOperation::AllocateArray), fallback ? 1u : 0u);
    if (!fallback) {
        EXPECT_EQ(loader.getLastError(), error == hipErrorOutOfMemory ? LoaderError::OutOfMemory : LoaderError::HipError);
        expectUnpublished(loader, handle.id);
        EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    }
    const auto independent = loader.createTextureV1(
        std::make_shared<TypedImageSource>(makeAuthoredMipSource(false)), capabilityDesc());
    ASSERT_TRUE(independent.valid);
    request(loader, {independent.id}, 1);
    expectResident(loader, independent.id);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 1u);
    if (!fallback) {
        request(loader, {handle.id}, 1);
        expectResident(loader, handle.id);
        EXPECT_EQ(status(loader, handle.id).attempts, 2u);
    }
}

INSTANTIATE_TEST_SUITE_P(ErrorAndPolicy, CapabilityAllocationFailure,
    testing::Combine(testing::ValuesIn(capabilityErrors),
                     testing::Values(cap::MipPolicy::LegacyCompatibility, cap::MipPolicy::Required,
                                     cap::MipPolicy::AllowBaseLevelFallback)));

const std::array<CapabilityFaultCase, 4> actualCases{{
    {HipOperation::GetLevel, cap::Operation::GetLevel, "GetLevel"},
    {HipOperation::Upload, cap::Operation::Upload, "Upload"},
    {HipOperation::CreateSampler, cap::Operation::CreateSampler, "CreateSampler"},
    {HipOperation::ReadSampler, cap::Operation::ReadSampler, "ReadSampler"}
}};

class CapabilityActualFailure : public Capability,
    public testing::WithParamInterface<std::tuple<CapabilityFaultCase, hipError_t>> {};

TEST_P(CapabilityActualFailure, NoUploadOrSamplerErrorMayPublishFallback) {
    const auto [fault, error] = GetParam();
    CapabilityFaults faults;
    faults.state->fail(fault.seam, error);
    DemandTextureLoader loader(capabilityOptions());
    const auto handle = loader.createTextureV1(std::make_shared<TypedImageSource>(makeAuthoredMipSource(false)),
        capabilityDesc(), policyFor(cap::MipPolicy::AllowBaseLevelFallback));
    ASSERT_TRUE(handle.valid);
    request(loader, {handle.id}, 0);
    const auto result = status(loader, handle.id);
    EXPECT_EQ(result.state, cap::State::Failed);
    expectFailure(result.primary, fault.operation, error);
    EXPECT_EQ(loader.getLastError(), error == hipErrorOutOfMemory ? LoaderError::OutOfMemory : LoaderError::HipError);
    EXPECT_EQ(result.fallback.outcome, Outcome::Success);
    EXPECT_EQ(result.capability, cap::Support::OperationSupported);
    EXPECT_EQ(result.submitted, fault.seam == HipOperation::CreateSampler || fault.seam == HipOperation::ReadSampler ? 1u : 0u);
    EXPECT_EQ(result.returned, 0u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateArray), 0u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    expectUnpublished(loader, handle.id);
    request(loader, {handle.id}, 1);
    expectResident(loader, handle.id);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 1u);
}

INSTANTIATE_TEST_SUITE_P(OperationAndError, CapabilityActualFailure,
    testing::Combine(testing::ValuesIn(actualCases), testing::ValuesIn(capabilityErrors)));

TEST_F(Capability, ProbeUsesThreeLevelsAndReportsOwningDeviceAndExactSamplerReadback) {
    CapabilityFaults faults;
    DemandTextureLoader loader(capabilityOptions());
    auto desc = capabilityDesc();
    desc.addressMode[0] = hipAddressModeClamp;
    const auto handle = loader.createTextureV1(std::make_shared<TypedImageSource>(makeAuthoredMipSource(true)), desc);
    ASSERT_TRUE(handle.valid);
    request(loader, {handle.id}, 1);
    expectResident(loader, handle.id);
    const auto result = status(loader, handle.id);
    EXPECT_EQ(result.capability, cap::Support::OperationSupported);
    EXPECT_EQ(result.device, device_);
    int runtime = 0, driver = 0;
    hipCtx_t context = nullptr;
    hipDeviceProp_t properties{};
    ASSERT_EQ(hipRuntimeGetVersion(&runtime), hipSuccess);
    ASSERT_EQ(hipDriverGetVersion(&driver), hipSuccess);
    ASSERT_EQ(hipCtxGetCurrent(&context), hipSuccess);
    ASSERT_EQ(hipGetDeviceProperties(&properties, device_), hipSuccess);
    EXPECT_EQ(result.runtimeVersion, runtime);
    EXPECT_EQ(result.driverVersion, driver);
    EXPECT_EQ(result.ownerContext, reinterpret_cast<uintptr_t>(context));
    EXPECT_NE(result.ownerContext, 0u);
    EXPECT_STREQ(result.deviceName, properties.name);
    EXPECT_STREQ(result.architecture, properties.gcnArchName);
    EXPECT_TRUE(result.requested == desc);
    EXPECT_EQ(result.submittedSampler.addressMode[0], desc.addressMode[0]);
    EXPECT_EQ(result.submittedSampler.addressMode[1], desc.addressMode[1]);
    EXPECT_EQ(result.submittedSampler.filterMode, desc.filterMode);
    EXPECT_EQ(result.submittedSampler.mipmapFilterMode, desc.mipmapFilterMode);
    EXPECT_EQ(result.submittedSampler.readMode, hipReadModeElementType);
    EXPECT_EQ(result.submittedSampler.maxMipmapLevelClamp, 3.f);
    hipTextureDesc returned{};
    ASSERT_EQ(hipGetTextureObjectTextureDesc(&returned, object(loader, handle.id)), hipSuccess);
    EXPECT_EQ(result.returnedSampler.addressMode[0], returned.addressMode[0]);
    EXPECT_EQ(result.returnedSampler.addressMode[1], returned.addressMode[1]);
    EXPECT_EQ(result.returnedSampler.filterMode, returned.filterMode);
    EXPECT_EQ(result.returnedSampler.mipmapFilterMode, returned.mipmapFilterMode);
    EXPECT_EQ(result.returnedSampler.readMode, returned.readMode);
    EXPECT_EQ(result.returnedSampler.normalizedCoords, returned.normalizedCoords);
    EXPECT_EQ(result.returnedSampler.sRGB, returned.sRGB);
    EXPECT_EQ(result.returnedSampler.minMipmapLevelClamp, returned.minMipmapLevelClamp);
    EXPECT_EQ(result.returnedSampler.maxMipmapLevelClamp, returned.maxMipmapLevelClamp);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate, true), 1u);
    EXPECT_EQ(faults.calls(HipOperation::ProbeGetLevel, true), 3u);
    EXPECT_EQ(faults.calls(HipOperation::ProbeUpload, true), 3u);
    EXPECT_EQ(faults.calls(HipOperation::ProbeCreateSampler, true), 1u);
    EXPECT_EQ(faults.calls(HipOperation::ProbeDestroySampler, true), 1u);
    EXPECT_EQ(faults.calls(HipOperation::ProbeFree, true), 1u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 1360u);
}

class CapabilityProbeIsolation : public Capability, public testing::WithParamInterface<int> {};

TEST_P(CapabilityProbeIsolation, SamplerRejectionDoesNotFallbackOrPoisonDifferentConfiguration) {
    CapabilityFaults faults;
    faults.state->fail(HipOperation::ProbeCreateSampler, hipErrorNotSupported);
    DemandTextureLoader loader(capabilityOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
    const auto supportedDesc = capabilityDesc();
    auto rejectedDesc = supportedDesc;
    switch (GetParam()) {
    case 0: rejectedDesc.addressMode[0] = hipAddressModeClamp; break;
    case 1: rejectedDesc.addressMode[1] = hipAddressModeClamp; break;
    case 2: rejectedDesc.filterMode = hipFilterModeLinear; break;
    case 3: rejectedDesc.mipmapFilterMode = hipFilterModeLinear; break;
    case 4: rejectedDesc.normalizedCoords = false; break;
    case 5: rejectedDesc.sRGB = true; break;
    }
    const auto rejected = loader.createTextureV1(source, rejectedDesc,
        policyFor(cap::MipPolicy::AllowBaseLevelFallback));
    const auto supported = loader.createTextureV1(source, supportedDesc);
    ASSERT_TRUE(rejected.valid);
    ASSERT_TRUE(supported.valid);
    request(loader, {rejected.id, supported.id}, 1);
    EXPECT_EQ(status(loader, rejected.id).state, cap::State::Failed);
    expectFailure(status(loader, rejected.id).primary, cap::Operation::ProbeCreateSampler, hipErrorNotSupported);
    expectUnpublished(loader, rejected.id);
    expectResident(loader, supported.id);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 2u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateArray), 0u);
    const auto objectBefore = object(loader, supported.id);
    const auto bytesBefore = loader.getTotalTextureMemory();
    loader.unloadTexture(rejected.id);
    EXPECT_EQ(object(loader, supported.id), objectBefore);
    EXPECT_EQ(loader.getTotalTextureMemory(), bytesBefore);
}

INSTANTIATE_TEST_SUITE_P(SamplerFields, CapabilityProbeIsolation, testing::Range(0, 6));

TEST_F(Capability, UnsupportedProbeIsIsolatedByUploadFormatAndLoaderLifetime) {
    CapabilityFaults faults;
    faults.state->fail(HipOperation::ProbeAllocate, hipErrorNotSupported);
    auto bytes = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
    auto floats = std::make_shared<TypedImageSource>(makeAuthoredMipSource(true));
    {
        DemandTextureLoader loader(capabilityOptions());
        const auto a = loader.createTextureV1(bytes, capabilityDesc());
        const auto b = loader.createTextureV1(floats, capabilityDesc());
        ASSERT_TRUE(a.valid);
        ASSERT_TRUE(b.valid);
        request(loader, {a.id, b.id}, 1);
        EXPECT_EQ(status(loader, a.id).capability, cap::Support::Unsupported);
        expectUnpublished(loader, a.id);
        expectResident(loader, b.id);
        EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 2u);
    }
    {
        DemandTextureLoader loader(capabilityOptions());
        const auto a = loader.createTextureV1(bytes, capabilityDesc());
        ASSERT_TRUE(a.valid);
        request(loader, {a.id}, 1);
        expectResident(loader, a.id);
        EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 3u);
    }
    EXPECT_EQ(faults.calls(HipOperation::FreeMipmapped, true), 2u);
}

TEST_F(Capability, ConcurrentRequestsCoalesceProbeAcrossEquivalentLimitsAndPriorities) {
    CapabilityFaults faults;
    DemandTextureLoader loader(capabilityOptions());
    std::vector<uint32_t> ids;
    for (unsigned int index = 0; index < 8; ++index) {
        auto desc = capabilityDesc(index % 2 == 0);
        desc.maxMipLevel = index % 3 == 0 ? 2 : 0;
        desc.evictionPriority = index % 2 == 0 ? EvictionPriority::High : EvictionPriority::Normal;
        const auto handle = loader.createTextureV1(
            std::make_shared<TypedImageSource>(makeAuthoredMipSource(false)), desc);
        ASSERT_TRUE(handle.valid);
        ids.push_back(handle.id);
    }
    queue(loader, ids);
    const auto context = loader.getDeviceContext();
    std::promise<void> start;
    const auto ready = start.get_future().share();
    std::vector<std::future<size_t>> work;
    for (unsigned int index = 0; index < 4; ++index)
        work.push_back(std::async(std::launch::async, [&loader, context, ready] {
            ready.wait();
            return loader.processRequests(nullptr, context);
        }));
    start.set_value();
    size_t loaded = 0;
    for (auto& future : work) loaded += future.get();
    EXPECT_EQ(loaded, ids.size());
    loader.launchPrepare();
    for (unsigned int index = 0; index < ids.size(); ++index)
        expectResident(loader, ids[index], index % 3 == 0 ? 2 : 4);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 1u);
    EXPECT_EQ(faults.calls(HipOperation::ProbeGetLevel), 3u);
    EXPECT_EQ(faults.calls(HipOperation::ProbeUpload), 3u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), ids.size());
}

class CapabilityPolicyOrdering : public Capability, public testing::WithParamInterface<bool> {};

TEST_P(CapabilityPolicyOrdering, RequiredAndPermissiveShareFullStorageInEitherRequestOrder) {
    CapabilityFaults faults;
    DemandTextureLoader loader(capabilityOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
    const auto a = loader.createTextureV1(source, capabilityDesc());
    const auto b = loader.createTextureV1(source, capabilityDesc(), policyFor(cap::MipPolicy::AllowBaseLevelFallback));
    ASSERT_TRUE(a.valid);
    ASSERT_TRUE(b.valid);
    request(loader, {GetParam() ? b.id : a.id}, 1);
    const auto bytes = loader.getTotalTextureMemory();
    request(loader, {GetParam() ? a.id : b.id}, 1);
    expectResident(loader, a.id);
    expectResident(loader, b.id);
    EXPECT_EQ(resource(loader, a.id).res.mipmap.mipmap, resource(loader, b.id).res.mipmap.mipmap);
    EXPECT_EQ(loader.getTotalTextureMemory(), bytes);
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), 1u);
    EXPECT_EQ(source->reads, 4u);
    loader.unloadTexture(b.id);
    expectResident(loader, a.id);
    EXPECT_EQ(loader.getTotalTextureMemory(), bytes);
    loader.unloadTexture(a.id);
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
}

TEST_P(CapabilityPolicyOrdering, RequiredFailureDoesNotDamagePermissiveFallbackInEitherRequestOrder) {
    CapabilityFaults faults;
    faults.state->fail(HipOperation::ProbeAllocate, hipErrorNotSupported);
    DemandTextureLoader loader(capabilityOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
    const auto strict = loader.createTextureV1(source, capabilityDesc());
    const auto allowed = loader.createTextureV1(source, capabilityDesc(),
                                               policyFor(cap::MipPolicy::AllowBaseLevelFallback));
    ASSERT_TRUE(strict.valid);
    ASSERT_TRUE(allowed.valid);
    if (GetParam()) {
        request(loader, {allowed.id}, 1);
        const auto preserved = object(loader, allowed.id);
        request(loader, {strict.id}, 0);
        EXPECT_EQ(object(loader, allowed.id), preserved);
    } else {
        request(loader, {strict.id}, 0);
        request(loader, {allowed.id}, 1);
    }
    const auto strictStatus = status(loader, strict.id), allowedStatus = status(loader, allowed.id);
    EXPECT_EQ(strictStatus.state, cap::State::Failed);
    EXPECT_EQ(strictStatus.primary.outcome, Outcome::Unsupported);
    expectUnpublished(loader, strict.id);
    EXPECT_EQ(allowedStatus.state, cap::State::Degraded);
    EXPECT_EQ(allowedStatus.resource, cap::Resource::Array);
    EXPECT_EQ(allowedStatus.firstResidentMip, 0u);
    EXPECT_EQ(allowedStatus.lastResidentMip, 0u);
    EXPECT_EQ(allowedStatus.originalLevels, 4u);
    EXPECT_EQ(loader.getResidentTextureCount(), 1u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 256u);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 1u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateArray), 1u);
    const auto preserved = object(loader, allowed.id);
    loader.unloadTexture(strict.id);
    EXPECT_EQ(object(loader, allowed.id), preserved);
    EXPECT_EQ(status(loader, allowed.id).state, cap::State::Degraded);
    EXPECT_EQ(loader.getTotalTextureMemory(), 256u);
}

INSTANTIATE_TEST_SUITE_P(RequiredOrPermissiveFirst, CapabilityPolicyOrdering, testing::Bool());

class CapabilityProbeCleanup : public Capability, public testing::WithParamInterface<bool> {};

TEST_P(CapabilityProbeCleanup, RetainsProbeHandlesAndPayloadUntilExplicitRetry) {
    const auto seam = GetParam() ? HipOperation::ProbeDestroySampler : HipOperation::ProbeFree;
    const auto operation = GetParam() ? cap::Operation::ProbeDestroySampler : cap::Operation::ProbeFree;
    CapabilityFaults faults;
    faults.state->fail(seam, hipErrorInvalidContext);
    DemandTextureLoader loader(capabilityOptions());
    const auto handle = loader.createTextureV1(
        std::make_shared<TypedImageSource>(makeAuthoredMipSource(false)), capabilityDesc());
    ASSERT_TRUE(handle.valid);
    request(loader, {handle.id}, 0);
    const auto result = status(loader, handle.id);
    EXPECT_EQ(result.state, cap::State::Failed);
    EXPECT_EQ(result.primary.outcome, Outcome::Success);
    expectFailure(result.cleanup, operation, hipErrorInvalidContext);
    EXPECT_EQ(result.capability, cap::Support::OperationSupported);
    EXPECT_EQ(loader.getTotalTextureMemory(), 84u);
    EXPECT_EQ(faults.calls(HipOperation::ProbeFree, true), 0u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), 0u);
    expectUnpublished(loader, handle.id);
    request(loader, {handle.id}, 1);
    expectResident(loader, handle.id);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 1u);
    EXPECT_EQ(faults.calls(HipOperation::ProbeFree, true), 1u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 340u);
}

INSTANTIATE_TEST_SUITE_P(FreeAndDestroySampler, CapabilityProbeCleanup, testing::Bool());

TEST_F(Capability, PrimaryProbeErrorSurvivesIndependentRollbackFailure) {
    CapabilityFaults faults;
    faults.state->fail(HipOperation::ProbeUpload, hipErrorOutOfMemory);
    faults.state->fail(HipOperation::ProbeFree, hipErrorInvalidContext);
    DemandTextureLoader loader(capabilityOptions());
    const auto handle = loader.createTextureV1(
        std::make_shared<TypedImageSource>(makeAuthoredMipSource(false)), capabilityDesc());
    ASSERT_TRUE(handle.valid);
    request(loader, {handle.id}, 0);
    const auto result = status(loader, handle.id);
    expectFailure(result.primary, cap::Operation::ProbeUpload, hipErrorOutOfMemory);
    expectFailure(result.cleanup, cap::Operation::ProbeFree, hipErrorInvalidContext);
    EXPECT_EQ(result.capability, cap::Support::Unknown);
    EXPECT_EQ(loader.getTotalTextureMemory(), 84u);
    request(loader, {handle.id}, 1);
    expectResident(loader, handle.id);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 2u);
    EXPECT_EQ(faults.calls(HipOperation::ProbeFree, true), 2u);
}

class CapabilityRollback : public Capability, public testing::WithParamInterface<bool> {};

TEST_P(CapabilityRollback, PrimaryUploadOrReadbackErrorSurvivesFailedResourceCleanup) {
    CapabilityFaults faults;
    const bool sampler = GetParam();
    faults.state->fail(sampler ? HipOperation::ReadSampler : HipOperation::Upload, hipErrorOutOfMemory);
    faults.state->fail(sampler ? HipOperation::DestroySampler : HipOperation::FreeMipmapped, hipErrorInvalidContext);
    DemandTextureLoader loader(capabilityOptions());
    const auto handle = loader.createTextureV1(
        std::make_shared<TypedImageSource>(makeAuthoredMipSource(false)), capabilityDesc());
    ASSERT_TRUE(handle.valid);
    request(loader, {handle.id}, 0);
    const auto result = status(loader, handle.id);
    expectFailure(result.primary, sampler ? cap::Operation::ReadSampler : cap::Operation::Upload, hipErrorOutOfMemory);
    expectFailure(result.cleanup, sampler ? cap::Operation::DestroySampler : cap::Operation::FreeMipmapped,
                  hipErrorInvalidContext);
    EXPECT_EQ(loader.getTotalTextureMemory(), 340u);
    expectUnpublished(loader, handle.id);
    request(loader, {handle.id}, 1);
    expectResident(loader, handle.id);
    EXPECT_EQ(loader.getTotalTextureMemory(), 340u);
    loader.unloadAll();
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
}

INSTANTIATE_TEST_SUITE_P(StorageAndSampler, CapabilityRollback, testing::Bool());

class CapabilityRetirement : public Capability, public testing::WithParamInterface<bool> {};

TEST_P(CapabilityRetirement, FailedFirstSamplerAndFinalBackingCleanupRetainSiblingAndBytes) {
    CapabilityFaults faults;
    const bool fallback = GetParam();
    if (fallback) faults.state->fail(HipOperation::ProbeAllocate, hipErrorNotSupported);
    DemandTextureLoader loader(capabilityOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
    auto firstDesc = capabilityDesc(), secondDesc = firstDesc;
    secondDesc.addressMode[0] = hipAddressModeClamp;
    const auto policy = policyFor(cap::MipPolicy::AllowBaseLevelFallback);
    const auto a = loader.createTextureV1(source, firstDesc, policy);
    const auto b = loader.createTextureV1(source, secondDesc, policy);
    ASSERT_TRUE(a.valid);
    ASSERT_TRUE(b.valid);
    request(loader, {a.id, b.id}, 2);
    const auto bytes = loader.getTotalTextureMemory();
    const auto siblingObject = object(loader, b.id);
    const auto freeSeam = fallback ? HipOperation::FreeArray : HipOperation::FreeMipmapped;
    const auto freeOperation = fallback ? cap::Operation::FreeArray : cap::Operation::FreeMipmapped;
    faults.state->fail(HipOperation::DestroySampler, hipErrorInvalidContext);
    loader.unloadTexture(a.id);
    expectFailure(status(loader, a.id).cleanup, cap::Operation::DestroySampler, hipErrorInvalidContext);
    EXPECT_EQ(loader.getTotalTextureMemory(), bytes);
    EXPECT_EQ(object(loader, b.id), siblingObject);
    EXPECT_EQ(faults.calls(freeSeam), 0u);
    loader.unloadTexture(a.id);
    EXPECT_EQ(loader.getTotalTextureMemory(), bytes);
    EXPECT_EQ(object(loader, b.id), siblingObject);
    faults.state->fail(freeSeam, hipErrorInvalidContext);
    loader.unloadTexture(b.id);
    expectFailure(status(loader, b.id).cleanup, freeOperation, hipErrorInvalidContext);
    EXPECT_EQ(status(loader, b.id).state, cap::State::Unloaded);
    expectUnpublished(loader, b.id);
    EXPECT_EQ(loader.getTotalTextureMemory(), bytes);
    loader.unloadTexture(b.id);
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    EXPECT_EQ(faults.calls(freeSeam, true), 1u);
}

TEST_P(CapabilityRetirement, DestructorRetriesFinalCleanupWithoutLeakingProbeOrBacking) {
    const bool fallback = GetParam();
    for (bool samplerFailure : {false, true}) {
        SCOPED_TRACE(samplerFailure);
        CapabilityFaults faults;
        if (fallback) faults.state->fail(HipOperation::ProbeAllocate, hipErrorNotSupported);
        {
            DemandTextureLoader loader(capabilityOptions());
            const auto handle = loader.createTextureV1(
                std::make_shared<TypedImageSource>(makeAuthoredMipSource(false)), capabilityDesc(),
                policyFor(cap::MipPolicy::AllowBaseLevelFallback));
            ASSERT_TRUE(handle.valid);
            request(loader, {handle.id}, 1);
            const auto seam = samplerFailure ? HipOperation::DestroySampler :
                              fallback ? HipOperation::FreeArray : HipOperation::FreeMipmapped;
            faults.state->fail(seam, hipErrorInvalidContext);
        }
        EXPECT_EQ(faults.calls(HipOperation::DestroySampler, true), 1u);
        EXPECT_EQ(faults.calls(fallback ? HipOperation::FreeArray : HipOperation::FreeMipmapped, true), 1u);
        EXPECT_EQ(faults.calls(HipOperation::ProbeFree, true), fallback ? 0u : 1u);
    }
}

INSTANTIATE_TEST_SUITE_P(NativeAndFallback, CapabilityRetirement, testing::Bool());

class CapabilityBlockedSource : public TypedImageSource {
public:
    CapabilityBlockedSource() : TypedImageSource(makeAuthoredMipSource(false)) {}
    bool readMipLevel(char* dest, unsigned int level, unsigned int width, unsigned int height,
                      hipStream_t stream) override {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            entered_ = true;
            cv_.notify_all();
            cv_.wait(lock, [&] { return released_; });
        }
        return TypedImageSource::readMipLevel(dest, level, width, height, stream);
    }
    bool waitForRead() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(10), [&] { return entered_; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool entered_ = false, released_ = false;
};

TEST_F(Capability, PendingStatusIsPollableWhileSourceReadBlocksAndTicketIsIncomplete) {
    DemandTextureLoader loader(capabilityOptions());
    auto source = std::make_shared<CapabilityBlockedSource>();
    const auto handle = loader.createTextureV1(source, capabilityDesc());
    ASSERT_TRUE(handle.valid);
    queue(loader, {handle.id});
    auto ticket = loader.processRequestsAsync(nullptr, loader.getDeviceContext());
    const bool entered = source->waitForRead();
    if (!entered) source->release();
    ASSERT_TRUE(entered);
    auto query = std::async(std::launch::async, [&] {
        cap::Status output;
        const auto outcome = loader.getTextureStatusV1(handle.id, output);
        return std::make_pair(outcome, output);
    });
    const auto ready = query.wait_for(std::chrono::seconds(2));
    EXPECT_EQ(ready, std::future_status::ready) << "Status must not wait for the blocked source or operation lock";
    EXPECT_EQ(ticket.numTasksRemaining(), 1);
    source->release();
    const auto result = query.get();
    EXPECT_EQ(result.first, Outcome::Success);
    if (ready == std::future_status::ready) {
        EXPECT_EQ(result.second.state, cap::State::Pending);
        EXPECT_EQ(result.second.attempts, 1u);
        EXPECT_EQ(result.second.published, 0u);
        EXPECT_EQ(result.second.capability, cap::Support::Unknown);
    }
    ticket.wait();
    EXPECT_EQ(ticket.numTasksRemaining(), 0);
    loader.launchPrepare();
    expectResident(loader, handle.id);
}

class CapabilityCancellation : public Capability, public testing::WithParamInterface<bool> {};

TEST_P(CapabilityCancellation, CancellationPreservesSourceLifetimeAndNeverRepublishes) {
    CapabilityFaults faults;
    auto source = std::make_shared<CapabilityBlockedSource>();
    std::weak_ptr<CapabilityBlockedSource> weak = source;
    {
        DemandTextureLoader loader(capabilityOptions());
        const auto handle = loader.createTextureV1(source, capabilityDesc());
        ASSERT_TRUE(handle.valid);
        queue(loader, {handle.id});
        auto ticket = loader.processRequestsAsync(nullptr, loader.getDeviceContext());
        const bool entered = source->waitForRead();
        if (!entered) source->release();
        ASSERT_TRUE(entered);
        std::promise<void> started;
        const auto signal = started.get_future();
        auto cancel = std::async(std::launch::async, [&] {
            started.set_value();
            if (GetParam()) loader.abort();
            else loader.unloadTexture(handle.id);
        });
        signal.wait();
        EXPECT_EQ(cancel.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
        EXPECT_FALSE(weak.expired());
        source->release();
        source.reset();
        ticket.wait();
        cancel.get();
        const auto result = status(loader, handle.id);
        EXPECT_EQ(result.state, GetParam() ? cap::State::Cancelled : cap::State::Unloaded);
        EXPECT_EQ(result.published, 0u);
        EXPECT_EQ(loader.getResidentTextureCount(), 0u);
        EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
        EXPECT_FALSE(weak.expired()) << "Unload is not registration release";
        expectUnpublished(loader, handle.id);
    }
    EXPECT_TRUE(weak.expired());
}

INSTANTIATE_TEST_SUITE_P(UnloadAndAbort, CapabilityCancellation, testing::Bool());

class CapabilityDeviceFailure : public Capability, public testing::WithParamInterface<HipOperation> {};

TEST_P(CapabilityDeviceFailure, WorkerDeviceOrContextFailureIsTransientAndNeverFallsBack) {
    CapabilityFaults faults;
    DemandTextureLoader loader(capabilityOptions());
    const auto handle = loader.createTextureV1(
        std::make_shared<TypedImageSource>(makeAuthoredMipSource(false)), capabilityDesc(),
        policyFor(cap::MipPolicy::AllowBaseLevelFallback));
    ASSERT_TRUE(handle.valid);
    queue(loader, {handle.id});
    // readRequests selects once; fail the following selection on the upload worker.
    faults.state->fail(GetParam(), hipErrorInvalidContext, 2);
    auto ticket = loader.processRequestsAsync(nullptr, loader.getDeviceContext());
    ticket.wait();
    loader.launchPrepare();
    const auto result = status(loader, handle.id);
    EXPECT_EQ(result.state, cap::State::Failed);
    expectFailure(result.primary, cap::Operation::SelectDevice, hipErrorInvalidContext);
    EXPECT_EQ(result.capability, cap::Support::Unknown);
    EXPECT_EQ(result.fallback.outcome, Outcome::Success);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 0u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    expectUnpublished(loader, handle.id);
    request(loader, {handle.id}, 1);
    expectResident(loader, handle.id);
}

INSTANTIATE_TEST_SUITE_P(DeviceAndContext, CapabilityDeviceFailure,
    testing::Values(HipOperation::SelectDevice, HipOperation::GetContext));

const std::array<std::array<uint16_t, 4>, 4> capabilityHalfColors{{
    {0xc000, 0x3400, 0x4200, 0x3800}, {0x4400, 0xbc00, 0x3000, 0x3c00},
    {0x3800, 0x4000, 0xc200, 0x3400}, {0x4800, 0x3a00, 0x3e00, 0x0000}
}};

TypedImageSource capabilityPixels(unsigned int format) {
    auto result = makeAuthoredMipSource(format != 0);
    if (format == 2) {
        result.info.format = HIP_AD_FORMAT_HALF;
        for (unsigned int level = 0; level < 4; ++level) {
            const auto color = packed(capabilityHalfColors[level]);
            const size_t texels = size_t(8u >> level) * (8u >> level);
            result.mipPixels[level].resize(texels * color.size());
            for (size_t pixel = 0; pixel < texels; ++pixel)
                std::memcpy(result.mipPixels[level].data() + pixel * color.size(), color.data(), color.size());
        }
        result.pixels = result.mipPixels[0];
    }
    return result;
}

std::array<float, 4> capabilityColor(unsigned int format, unsigned int level) {
    if (format != 0) return authoredFloatMipColors[level];
    std::array<float, 4> result{};
    for (size_t channel = 0; channel < 4; ++channel)
        result[channel] = authoredByteMipColors[level][channel] / 255.f;
    return result;
}

class CapabilityPixels : public Capability,
    public testing::WithParamInterface<std::tuple<unsigned int, bool, bool, unsigned int>> {};

TEST_P(CapabilityPixels, AuthoredStorageAndPointMipLodGradientClampsPreservePixels) {
    const auto [format, fallback, generate, limit] = GetParam();
    CapabilityFaults faults;
    if (fallback) faults.state->fail(HipOperation::ProbeAllocate, hipErrorNotSupported);
    DemandTextureLoader loader(capabilityOptions());
    auto source = std::make_shared<TypedImageSource>(capabilityPixels(format));
    auto desc = capabilityDesc(generate);
    desc.maxMipLevel = limit;
    desc.addressMode[0] = desc.addressMode[1] = hipAddressModeClamp;
    const auto handle = loader.createTextureV1(source, desc,
        policyFor(fallback ? cap::MipPolicy::AllowBaseLevelFallback : cap::MipPolicy::Required));
    ASSERT_TRUE(handle.valid);
    request(loader, {handle.id}, 1);
    const unsigned int requestedLevels = limit ? limit : 4;
    const unsigned int levels = fallback ? 1 : requestedLevels;
    const auto result = status(loader, handle.id);
    EXPECT_EQ(result.originalWidth, 8u);
    EXPECT_EQ(result.originalHeight, 8u);
    EXPECT_EQ(result.originalLevels, requestedLevels);
    EXPECT_EQ(result.resourceWidth, 8u);
    EXPECT_EQ(result.resourceHeight, 8u);
    EXPECT_EQ(result.resourceLevels, levels);
    EXPECT_EQ(result.firstResidentMip, 0u);
    EXPECT_EQ(result.lastResidentMip, levels - 1);
    EXPECT_EQ(result.state, fallback ? cap::State::Degraded : cap::State::Resident);
    EXPECT_EQ(result.reason, fallback ? cap::Reason::CapabilityFallback : cap::Reason::None);
    EXPECT_NE(result.capability, cap::Support::BehaviorQualified);
    EXPECT_EQ(result.submittedSampler.readMode, format == 0 ? hipReadModeNormalizedFloat : hipReadModeElementType);
    const auto backing = resource(loader, handle.id);
    ASSERT_EQ(backing.resType, fallback ? hipResourceTypeArray : hipResourceTypeMipmappedArray);
    size_t bytes = 0;
    std::vector<unsigned int> expectedReads;
    for (unsigned int level = 0; level < levels; ++level) {
        SCOPED_TRACE(level);
        expectedReads.push_back(level);
        hipArray_t array = fallback ? backing.res.array.array : nullptr;
        if (!fallback)
            ASSERT_EQ(hipGetMipmappedArrayLevel(&array, backing.res.mipmap.mipmap, level), hipSuccess);
        hipChannelFormatDesc channel{};
        hipExtent extent{};
        unsigned int flags = 0;
        ASSERT_EQ(hipArrayGetInfo(&channel, &extent, &flags, array), hipSuccess);
        const size_t side = 8u >> level;
        EXPECT_EQ(extent.width, side);
        EXPECT_EQ(extent.height, side);
        EXPECT_EQ(channel.x, format == 0 ? 8 : 32);
        EXPECT_EQ(channel.y, channel.x);
        EXPECT_EQ(channel.z, channel.x);
        EXPECT_EQ(channel.w, channel.x);
        const size_t rowBytes = side * (format == 0 ? 4 : 16);
        std::vector<unsigned char> actual(rowBytes * side);
        ASSERT_EQ(hipMemcpy2DFromArray(actual.data(), rowBytes, array, 0, 0, rowBytes, side,
                                      hipMemcpyDeviceToHost), hipSuccess);
        std::vector<unsigned char> expected;
        const auto pixel = format == 0 ? packed(authoredByteMipColors[level]) : packed(authoredFloatMipColors[level]);
        for (size_t index = 0; index < side * side; ++index)
            expected.insert(expected.end(), pixel.begin(), pixel.end());
        EXPECT_EQ(actual, expected) << "Readback must preserve authored pixels, including HALF-to-FLOAT conversion";
        bytes += actual.size();
    }
    EXPECT_EQ(source->readLevels, expectedReads);
    EXPECT_EQ(loader.getTotalTextureMemory(), bytes);
    EXPECT_EQ(result.payloadBytes, bytes);
    TextureSamplingHarness harness;
    ASSERT_EQ(harness.open(samplingModulePath()), hipSuccess);
    const std::array<float, 12> lods{-2.f, 0.f, .25f, .75f, 1.f, 1.25f, 1.75f,
                                    2.f, 2.25f, 2.75f, 3.f, 8.f};
    std::vector<SamplingInput> inputs;
    std::vector<unsigned int> expectedLevels;
    for (const auto path : {SamplingPath::Lod, SamplingPath::Gradient}) {
        for (float lod : lods) {
            SamplingInput input;
            input.textureId = handle.id;
            input.path = path;
            input.lod = lod;
            input.ddx = make_float2(std::exp2(lod) / 8.f, 0);
            input.ddy = make_float2(0, std::exp2(lod) / 8.f);
            inputs.push_back(input);
            expectedLevels.push_back(fallback ? 0u : static_cast<unsigned int>(
                std::clamp(std::floor(lod + .5f), 0.f, float(levels - 1))));
        }
    }
    std::vector<SamplingResult> samples;
    ASSERT_EQ(harness.sample(loader.getDeviceContext(), inputs, samples), hipSuccess);
    ASSERT_EQ(samples.size(), inputs.size());
    for (size_t index = 0; index < samples.size(); ++index) {
        SCOPED_TRACE(index);
        SCOPED_TRACE(inputs[index].lod);
        expectPixel(samples[index], capabilityColor(format, expectedLevels[index]));
    }
}

INSTANTIATE_TEST_SUITE_P(FormatResourceGenerationLimit, CapabilityPixels,
    testing::Combine(testing::Values(0u, 1u, 2u), testing::Bool(), testing::Bool(), testing::Values(0u, 2u)));

class CapabilitySingleLevelPixels : public Capability,
    public testing::WithParamInterface<std::tuple<unsigned int, bool>> {};

TEST_P(CapabilitySingleLevelPixels, DisabledAndSingletonPreserveOriginalBaseWithoutCapabilityLoss) {
    const auto [format, singleton] = GetParam();
    CapabilityFaults faults;
    faults.state->fail(HipOperation::ProbeAllocate, hipErrorNotSupported);
    DemandTextureLoader loader(capabilityOptions());
    auto source = std::make_shared<TypedImageSource>(capabilityPixels(format));
    if (singleton) {
        const size_t nativeBytes = format == 0 ? 4 : format == 1 ? 16 : 8;
        source->info.width = source->info.height = source->info.numMipLevels = 1;
        source->pixels.resize(nativeBytes);
        source->mipPixels = {source->pixels};
    }
    const auto handle = loader.createTextureV1(source, capabilityDesc(false),
        policyFor(singleton ? cap::MipPolicy::Required : cap::MipPolicy::Disabled));
    ASSERT_TRUE(handle.valid);
    request(loader, {handle.id}, 1);
    expectResident(loader, handle.id, 1);
    const auto result = status(loader, handle.id);
    EXPECT_EQ(result.reason, singleton ? cap::Reason::Singleton : cap::Reason::Disabled);
    EXPECT_EQ(result.fallback.outcome, Outcome::Success);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 0u);
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{0}));
    TextureSamplingHarness harness;
    ASSERT_EQ(harness.open(samplingModulePath()), hipSuccess);
    std::vector<SamplingInput> inputs;
    for (const auto path : {SamplingPath::Lod, SamplingPath::Gradient}) {
        for (float lod : {-1.f, 0.f, .25f, .75f, 1.f, 4.f}) {
            SamplingInput input;
            input.textureId = handle.id;
            input.path = path;
            input.lod = lod;
            input.ddx = make_float2(std::exp2(lod) / float(source->info.width), 0);
            input.ddy = make_float2(0, std::exp2(lod) / float(source->info.height));
            inputs.push_back(input);
        }
    }
    std::vector<SamplingResult> samples;
    ASSERT_EQ(harness.sample(loader.getDeviceContext(), inputs, samples), hipSuccess);
    ASSERT_EQ(samples.size(), inputs.size());
    for (const auto& sample : samples) expectPixel(sample, capabilityColor(format, 0));
}

INSTANTIATE_TEST_SUITE_P(FormatDisabledOrSingleton, CapabilitySingleLevelPixels,
    testing::Combine(testing::Values(0u, 1u, 2u), testing::Bool()));

class CapabilityBasePixels : public Capability, public testing::WithParamInterface<bool> {};

TEST_P(CapabilityBasePixels, ForcedArrayPreservesOriginalSpatialPixelsAtPositiveAndFractionalLod) {
    CapabilityFaults faults;
    faults.state->fail(HipOperation::AllocateMipmapped, hipErrorNotSupported);
    DemandTextureLoader loader(capabilityOptions());
    auto source = std::make_shared<TypedImageSource>(makeBoundaryPatternSource(true));
    auto desc = capabilityDesc();
    desc.filterMode = GetParam() ? hipFilterModeLinear : hipFilterModePoint;
    desc.mipmapFilterMode = hipFilterModeLinear;
    desc.addressMode[0] = desc.addressMode[1] = hipAddressModeClamp;
    const auto handle = loader.createTextureV1(source, desc, policyFor(cap::MipPolicy::AllowBaseLevelFallback));
    ASSERT_TRUE(handle.valid);
    request(loader, {handle.id}, 1);
    EXPECT_EQ(status(loader, handle.id).state, cap::State::Degraded);
    EXPECT_EQ(resource(loader, handle.id).resType, hipResourceTypeArray);
    TextureSamplingHarness harness;
    ASSERT_EQ(harness.open(samplingModulePath()), hipSuccess);
    std::vector<SamplingInput> inputs;
    std::vector<std::array<float, 4>> expected;
    for (const auto path : {SamplingPath::Lod, SamplingPath::Gradient}) {
        for (float lod : {-2.f, 0.f, .25f, .5f, .75f, 1.f, 1.5f, 2.f, 8.f}) {
            for (unsigned int pixel : {0u, 5u, 10u, 15u}) {
                SamplingInput input;
                input.textureId = handle.id;
                input.path = path;
                input.u = (float(pixel % 4) + .5f) / 4.f;
                input.v = (float(pixel / 4) + .5f) / 4.f;
                input.lod = lod;
                input.ddx = make_float2(std::exp2(lod) / 4.f, 0);
                input.ddy = make_float2(0, std::exp2(lod) / 4.f);
                inputs.push_back(input);
                std::array<float, 4> color{};
                std::memcpy(color.data(), source->pixels.data() + pixel * sizeof(color), sizeof(color));
                expected.push_back(color);
            }
        }
    }
    std::vector<SamplingResult> samples;
    ASSERT_EQ(harness.sample(loader.getDeviceContext(), inputs, samples), hipSuccess);
    ASSERT_EQ(samples.size(), expected.size());
    for (size_t index = 0; index < samples.size(); ++index) {
        SCOPED_TRACE(index);
        expectPixel(samples[index], expected[index]);
    }
}

INSTANTIATE_TEST_SUITE_P(PointAndLinearSpatial, CapabilityBasePixels, testing::Bool());

class CapabilityArrayFailure : public Capability, public testing::WithParamInterface<hipError_t> {};

TEST_P(CapabilityArrayFailure, FailedFallbackAllocationRetainsBothPrimaryAndFallbackProvenance) {
    CapabilityFaults faults;
    faults.state->fail(HipOperation::ProbeAllocate, hipErrorNotSupported);
    faults.state->fail(HipOperation::AllocateArray, GetParam());
    DemandTextureLoader loader(capabilityOptions());
    const auto handle = loader.createTextureV1(
        std::make_shared<TypedImageSource>(makeAuthoredMipSource(false)), capabilityDesc(),
        policyFor(cap::MipPolicy::AllowBaseLevelFallback));
    ASSERT_TRUE(handle.valid);
    request(loader, {handle.id}, 0);
    const auto result = status(loader, handle.id);
    EXPECT_EQ(result.state, cap::State::Failed);
    expectFailure(result.primary, cap::Operation::AllocateArray, GetParam());
    expectFailure(result.fallback, cap::Operation::ProbeAllocate, hipErrorNotSupported);
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    expectUnpublished(loader, handle.id);
    request(loader, {handle.id}, 1);
    EXPECT_EQ(status(loader, handle.id).state, cap::State::Degraded);
    EXPECT_EQ(status(loader, handle.id).resourceLevels, 1u);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 1u);
}

INSTANTIATE_TEST_SUITE_P(EveryError, CapabilityArrayFailure, testing::ValuesIn(capabilityErrors));

class CapabilitySourceFailure : public Capability, public testing::WithParamInterface<unsigned int> {};

TEST_P(CapabilitySourceFailure, AuthoredReadFailureIsExplicitAndDoesNotDamageResidentTexture) {
    DemandTextureLoader loader(capabilityOptions());
    auto healthy = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
    auto broken = std::make_shared<TypedImageSource>(makeAuthoredMipSource(true));
    broken->failReadLevel = GetParam();
    const auto resident = loader.createTextureV1(healthy, capabilityDesc());
    const auto failed = loader.createTextureV1(broken, capabilityDesc(false),
        policyFor(cap::MipPolicy::AllowBaseLevelFallback));
    ASSERT_TRUE(resident.valid);
    ASSERT_TRUE(failed.valid);
    request(loader, {resident.id}, 1);
    const auto original = object(loader, resident.id);
    request(loader, {failed.id}, 0);
    const auto result = status(loader, failed.id);
    EXPECT_EQ(result.state, cap::State::Failed);
    EXPECT_EQ(result.primary.operation, cap::Operation::SourceRead);
    EXPECT_EQ(result.primary.outcome, Outcome::SourceFailure);
    EXPECT_EQ(result.primary.rawHipError, 0);
    EXPECT_EQ(result.fallback.outcome, Outcome::Success);
    expectUnpublished(loader, failed.id);
    EXPECT_EQ(object(loader, resident.id), original);
    EXPECT_EQ(loader.getTotalTextureMemory(), 340u);
    broken->failReadLevel = UINT32_MAX;
    request(loader, {failed.id}, 1);
    expectResident(loader, failed.id);
    EXPECT_EQ(object(loader, resident.id), original);
}

INSTANTIATE_TEST_SUITE_P(BaseAndAuthoredMip, CapabilitySourceFailure, testing::Values(0u, 2u));

class CapabilityThrowingSource : public TypedImageSource {
public:
    CapabilityThrowingSource(unsigned int level, bool allocation)
        : TypedImageSource(makeAuthoredMipSource(false)), failLevel(level), allocationFailure(allocation) {}
    bool readMipLevel(char* dest, unsigned int level, unsigned int width, unsigned int height,
                      hipStream_t stream) override {
        if (fail && level == failLevel) {
            if (allocationFailure) throw std::bad_alloc();
            throw std::runtime_error("Injected capability source exception");
        }
        return TypedImageSource::readMipLevel(dest, level, width, height, stream);
    }
    unsigned int failLevel;
    bool allocationFailure;
    bool fail = true;
};

class CapabilitySourceException : public Capability,
    public testing::WithParamInterface<std::tuple<unsigned int, bool>> {};

TEST_P(CapabilitySourceException, HostAllocationFailureIsNotDeviceOutOfMemoryAndCanRetry) {
    const auto [level, allocation] = GetParam();
    CapabilityFaults faults;
    DemandTextureLoader loader(capabilityOptions());
    const bool cleanupFailure = allocation && level != 0;
    if (cleanupFailure) faults.state->fail(HipOperation::FreeMipmapped, hipErrorInvalidContext);
    auto source = std::make_shared<CapabilityThrowingSource>(level, allocation);
    const auto handle = loader.createTextureV1(source, capabilityDesc(false),
        policyFor(cap::MipPolicy::AllowBaseLevelFallback));
    ASSERT_TRUE(handle.valid);
    request(loader, {handle.id}, 0);
    const auto result = status(loader, handle.id);
    EXPECT_EQ(result.state, cap::State::Failed);
    EXPECT_EQ(result.primary.operation, cap::Operation::SourceRead);
    EXPECT_EQ(result.primary.outcome, allocation ? Outcome::HostOutOfMemory : Outcome::SourceFailure);
    EXPECT_EQ(result.primary.rawHipError, 0);
    if (cleanupFailure)
        expectFailure(result.cleanup, cap::Operation::FreeMipmapped, hipErrorInvalidContext);
    else
        EXPECT_EQ(result.cleanup.outcome, Outcome::Success);
    EXPECT_EQ(result.fallback.outcome, Outcome::Success);
    EXPECT_EQ(loader.getLastError(), allocation ? LoaderError::OutOfMemory : LoaderError::ImageLoadFailed);
    EXPECT_EQ(loader.getTotalTextureMemory(), cleanupFailure ? 340u : 0u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateArray), 0u);
    expectUnpublished(loader, handle.id);
    source->fail = false;
    request(loader, {handle.id}, 1);
    expectResident(loader, handle.id);
    EXPECT_EQ(status(loader, handle.id).attempts, 2u);
}

INSTANTIATE_TEST_SUITE_P(BaseOrMipHostOrSourceError, CapabilitySourceException,
    testing::Combine(testing::Values(0u, 2u), testing::Bool()));

TEST_F(Capability, PublicationFailureReportsUnpublishedAndRetriesWithoutReallocatingStorage) {
    CapabilityFaults faults;
    DemandTextureLoader loader(capabilityOptions());
    const auto handle = loader.createTextureV1(
        std::make_shared<TypedImageSource>(makeAuthoredMipSource(false)), capabilityDesc());
    ASSERT_TRUE(handle.valid);
    queue(loader, {handle.id});
    ASSERT_EQ(loader.processRequests(nullptr, loader.getDeviceContext()), 1u);
    EXPECT_EQ(status(loader, handle.id).published, 0u);
    faults.state->fail(HipOperation::PublishMappings, hipErrorInvalidContext);
    loader.launchPrepare();
    const auto failed = status(loader, handle.id);
    EXPECT_EQ(failed.state, cap::State::Failed);
    expectFailure(failed.primary, cap::Operation::Publish, hipErrorInvalidContext);
    expectUnpublished(loader, handle.id);
    EXPECT_EQ(loader.getTotalTextureMemory(), 340u);
    loader.launchPrepare();
    expectResident(loader, handle.id);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 1u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), 1u);
    EXPECT_EQ(faults.calls(HipOperation::CreateSampler), 1u);
}

#ifdef USE_OIIO
TEST_F(Capability, FilenameApiPreservesAuthoredFloatMipsWithGenerationDisabled) {
    struct File {
        std::filesystem::path path;
        ~File() { std::error_code error; std::filesystem::remove(path, error); }
    } file{std::filesystem::current_path() / ("capability-authored-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".exr")};
    auto output = OIIO::ImageOutput::create(file.path.string());
    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(output->supports("mipmap"));
    for (unsigned int level = 0; level < 4; ++level) {
        const int side = 8 >> level;
        OIIO::ImageSpec spec(side, side, 4, OIIO::TypeDesc::FLOAT);
        spec.tile_width = spec.tile_height = 16;
        spec.attribute("textureformat", "Plain Texture");
        spec.attribute("openexr:levelmode", 1);
        spec.attribute("openexr:roundingmode", 0);
        ASSERT_TRUE(output->open(file.path.string(), spec,
            level == 0 ? OIIO::ImageOutput::Create : OIIO::ImageOutput::AppendMIPLevel)) << output->geterror();
        std::vector<float> pixels;
        for (int pixel = 0; pixel < side * side; ++pixel)
            pixels.insert(pixels.end(), authoredFloatMipColors[level].begin(), authoredFloatMipColors[level].end());
        ASSERT_TRUE(output->write_image(OIIO::TypeDesc::FLOAT, pixels.data())) << output->geterror();
    }
    ASSERT_TRUE(output->close()) << output->geterror();
    output.reset();
    DemandTextureLoader loader(capabilityOptions());
    const auto handle = loader.createTextureV1(file.path.string(), capabilityDesc(false));
    ASSERT_TRUE(handle.valid);
    EXPECT_EQ(status(loader, handle.id).policy, cap::MipPolicy::Required);
    EXPECT_EQ(loader.createTextureV1(file.path.string(), capabilityDesc(false)).id, handle.id);
    request(loader, {handle.id}, 1);
    expectResident(loader, handle.id);
    TextureSamplingHarness harness;
    ASSERT_EQ(harness.open(samplingModulePath()), hipSuccess);
    std::vector<SamplingInput> inputs(4);
    for (unsigned int level = 0; level < 4; ++level) {
        inputs[level].textureId = handle.id;
        inputs[level].path = SamplingPath::Lod;
        inputs[level].lod = static_cast<float>(level);
    }
    std::vector<SamplingResult> samples;
    ASSERT_EQ(harness.sample(loader.getDeviceContext(), inputs, samples), hipSuccess);
    ASSERT_EQ(samples.size(), 4u);
    for (unsigned int level = 0; level < 4; ++level)
        expectPixel(samples[level], authoredFloatMipColors[level]);
}
#endif

} // namespace
} }
