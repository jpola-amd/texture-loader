// SPDX-License-Identifier: MIT
#include "TestUtils.h"
#include "ImageDataTestUtils.h"
#include "TextureSamplingHarness.h"
#include <DemandLoading/Internal/HipCalls.h>
#include <DemandLoading/Internal/TextureIdentity.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <future>
#include <set>
#include <tuple>
#ifdef USE_OIIO
#include <OpenImageIO/imageio.h>
#endif

namespace hip_demand { namespace test {
namespace {
namespace aniso = anisotropy_v1;
namespace cap = capability_v1;
using contract_v1::Outcome;
using internal::HipOperation;

aniso::Request anisotropy(unsigned int ratio) {
    aniso::Request request;
    request.maxAnisotropy = ratio;
    return request;
}

struct AnisotropyFaults {
    std::shared_ptr<internal::HipFaultState> state = std::make_shared<internal::HipFaultState>();
    AnisotropyFaults() { internal::setHipFaultState(state); }
    ~AnisotropyFaults() { internal::setHipFaultState(nullptr); }
    size_t calls(HipOperation operation) const {
        const auto records = state->records();
        return std::count_if(records.begin(), records.end(), [=](const auto& r) { return r.operation == operation; });
    }
};

LoaderOptions anisotropyOptions() {
    LoaderOptions options;
    options.maxTextures = 32;
    options.maxRequestsPerLaunch = 64;
    options.maxThreads = 2;
    options.enableEviction = false;
    return options;
}

TEST(AnisotropyContract, ProfilesAndFieldwiseKeysPreserveLegacyLayouts) {
    static_assert(sizeof(TextureDesc) == 28 && offsetof(TextureDesc, maxMipLevel) == 20);
    static_assert(sizeof(cap::Policy) == 12);
    EXPECT_EQ(aniso::Request{}.maxAnisotropy, 1u);
    EXPECT_EQ(aniso::Request::legacy().maxAnisotropy, 0u);
    EXPECT_EQ(aniso::Request::parity().maxAnisotropy, 16u);
    EXPECT_EQ(aniso::Request::parity().requirement, aniso::Requirement::RequireQualified);
    const TextureDesc desc;
    const auto base = anisotropy(1);
    std::set<size_t> hashes;
    for (unsigned int ratio : {1u, 2u, 3u, 4u, 5u, 7u, 8u, 16u}) {
        auto request = anisotropy(ratio);
        EXPECT_EQ(request == base, ratio == 1);
        hashes.insert(internal::samplerHash(desc, cap::MipPolicy::Required, request));
    }
    EXPECT_EQ(hashes.size(), 8u);
    auto changed = base;
    changed.requirement = aniso::Requirement::RequireQualified;
    EXPECT_FALSE(changed == base);
    EXPECT_NE(internal::samplerHash(desc, cap::MipPolicy::Required, base),
              internal::samplerHash(desc, cap::MipPolicy::Required, changed));
    EXPECT_FALSE(aniso::Request::parity() == anisotropy(16));
    changed = base;
    ++changed.abi.version;
    EXPECT_FALSE(changed == base);
    changed = base;
    --changed.abi.byteSize;
    EXPECT_FALSE(changed == base);
}

class Anisotropy : public HipTestFixture {
protected:
    void request(DemandTextureLoader& loader, const std::vector<uint32_t>& ids, size_t expected, bool async = false) {
        loader.launchPrepare();
        auto ctx = loader.getDeviceContext();
        const uint32_t count = static_cast<uint32_t>(ids.size());
        ASSERT_LE(count, ctx.maxRequests);
        ASSERT_EQ(hipMemcpy(ctx.requests, ids.data(), ids.size() * sizeof(uint32_t), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(ctx.requestCount, &count, sizeof(count), hipMemcpyHostToDevice), hipSuccess);
        if (async) {
            const auto before = loader.getResidentTextureCount();
            auto ticket = loader.processRequestsAsync(nullptr, ctx);
            ticket.wait();
            EXPECT_EQ(ticket.numTasksRemaining(), 0);
            EXPECT_EQ(loader.getResidentTextureCount(), before + expected);
        } else {
            const auto loaded = loader.processRequests(nullptr, ctx);
            if (loaded != expected) {
                for (uint32_t id : ids) {
                    cap::Status s;
                    ASSERT_EQ(loader.getTextureStatusV1(id, s), Outcome::Success);
                    std::cout << "id=" << id << " operation=" << static_cast<unsigned int>(s.primary.operation)
                              << " rawHIP=" << s.primary.rawHipError << " outcome=" << static_cast<unsigned int>(s.primary.outcome)
                              << " submitted=" << s.submitted << " maxAnisotropy=" << s.submittedSampler.maxAnisotropy << '\n';
                }
            }
            EXPECT_EQ(loaded, expected);
        }
        loader.launchPrepare();
    }
    aniso::Status status(DemandTextureLoader& loader, uint32_t id) {
        aniso::Status result;
        EXPECT_EQ(loader.getTextureAnisotropyStatusV1(id, result), Outcome::Success);
        return result;
    }
    hipResourceDesc resource(DemandTextureLoader& loader, uint32_t id) {
        hipTextureObject_t object = 0;
        EXPECT_EQ(hipMemcpy(&object, loader.getDeviceContext().textures + id, sizeof(object), hipMemcpyDeviceToHost),
                  hipSuccess);
        hipResourceDesc result{};
        EXPECT_NE(object, hipTextureObject_t{});
        if (object)
            EXPECT_EQ(hipGetTextureObjectResourceDesc(&result, object), hipSuccess);
        return result;
    }
};

class AnisotropySettings : public Anisotropy,
    public testing::WithParamInterface<std::tuple<unsigned int, bool, bool>> {};

TEST_P(AnisotropySettings, SharedStorageOrdersReadbackAndReload) {
    const auto [ratio, mipmapped, reverse] = GetParam();
    AnisotropyFaults faults;
    DemandTextureLoader loader(anisotropyOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(true));
    TextureDesc desc;
    desc.filterMode = hipFilterModePoint;
    desc.mipmapFilterMode = hipFilterModePoint;
    cap::Policy policy;
    if (!mipmapped) policy.mipPolicy = cap::MipPolicy::Disabled;
    const auto firstRequest = reverse ? anisotropy(ratio) : aniso::Request::legacy();
    const auto secondRequest = reverse ? aniso::Request::legacy() : anisotropy(ratio);
    const auto first = loader.createTextureAnisotropyV1(source, desc, firstRequest, policy);
    const auto second = loader.createTextureAnisotropyV1(source, desc, secondRequest, policy);
    ASSERT_TRUE(first.valid && second.valid);
    ASSERT_NE(first.id, second.id);
    EXPECT_EQ(loader.createTextureAnisotropyV1(source, desc, firstRequest, policy).id, first.id);
    EXPECT_EQ(loader.createTextureAnisotropyV1(source, desc, secondRequest, policy).id, second.id);
    EXPECT_EQ(status(loader, first.id).texture.submitted, 0u);
    // Both demand orders must work independently of creation order.
    for (bool reverseDemand : {false, true}) {
        request(loader, reverseDemand ? std::vector<uint32_t>{second.id, first.id} :
                                       std::vector<uint32_t>{first.id, second.id}, 2, reverseDemand);
        for (const auto& pair : {std::make_pair(first.id, firstRequest), std::make_pair(second.id, secondRequest)}) {
            const auto s = status(loader, pair.first);
            EXPECT_TRUE(s.requested == pair.second);
            EXPECT_EQ(s.qualification, aniso::Qualification::Unqualified);
            EXPECT_EQ(s.texture.submitted, 1u);
            EXPECT_EQ(s.texture.returned, 1u);
            EXPECT_EQ(s.texture.published, 1u);
            EXPECT_EQ(s.texture.submittedSampler.maxAnisotropy, pair.second.maxAnisotropy);
            hipTextureObject_t object = 0;
            ASSERT_EQ(hipMemcpy(&object, loader.getDeviceContext().textures + pair.first, sizeof(object),
                                hipMemcpyDeviceToHost), hipSuccess);
            hipTextureDesc actual{};
            ASSERT_EQ(hipGetTextureObjectTextureDesc(&actual, object), hipSuccess);
            EXPECT_EQ(s.texture.returnedSampler.maxAnisotropy, actual.maxAnisotropy);
            EXPECT_EQ(s.texture.submittedSampler.filterMode, desc.filterMode);
            if (mipmapped) EXPECT_EQ(s.texture.submittedSampler.mipmapFilterMode, desc.mipmapFilterMode);
            EXPECT_EQ(s.texture.resourceLevels, mipmapped ? 4u : 1u);
            EXPECT_EQ(s.texture.resourceWidth, 8u);
            if (pair.second.maxAnisotropy > 1)
                EXPECT_EQ(s.texture.state, cap::State::Degraded);
            EXPECT_EQ(bool(s.limitations & aniso::SingleLevel), !mipmapped);
        }
        const auto a = resource(loader, first.id), b = resource(loader, second.id);
        EXPECT_EQ(a.resType, mipmapped ? hipResourceTypeMipmappedArray : hipResourceTypeArray);
        EXPECT_EQ(b.resType, a.resType);
        if (mipmapped) EXPECT_EQ(a.res.mipmap.mipmap, b.res.mipmap.mipmap);
        else EXPECT_EQ(a.res.array.array, b.res.array.array);
        const size_t payload = mipmapped ? 1360 : 1024;
        EXPECT_EQ(loader.getTotalTextureMemory(), payload);
        loader.unloadTexture(first.id);
        EXPECT_EQ(loader.getResidentTextureCount(), 1u);
        EXPECT_EQ(loader.getTotalTextureMemory(), payload);
        const auto reads = source->reads;
        request(loader, {first.id}, 1);
        EXPECT_EQ(source->reads, reads);
        EXPECT_EQ(status(loader, first.id).texture.submittedSampler.maxAnisotropy, firstRequest.maxAnisotropy);
        loader.unloadAll();
        EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    }
    EXPECT_EQ(faults.calls(mipmapped ? HipOperation::AllocateMipmapped : HipOperation::AllocateArray), 2u);
    EXPECT_EQ(source->reads, mipmapped ? 8u : 2u);
}

INSTANTIATE_TEST_SUITE_P(RatiosResourcesOrders, AnisotropySettings,
    testing::Combine(testing::Values(1u, 2u, 3u, 4u, 5u, 7u, 8u, 16u), testing::Bool(), testing::Bool()));

class AnisotropyInvalid : public Anisotropy, public testing::WithParamInterface<int> {};
TEST_P(AnisotropyInvalid, AllRegistrationOverloadsRejectWithoutConsumingIdentity) {
    DemandTextureLoader loader(anisotropyOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
    auto bad = anisotropy(1);
    switch (GetParam()) {
        case 0: bad.maxAnisotropy = 0; break;
        case 1: bad.maxAnisotropy = 17; break;
        case 2: bad.maxAnisotropy = UINT32_MAX; break;
        case 3: bad.abi.version = 0; break;
        case 4: bad.abi.version = 2; break;
        case 5: --bad.abi.byteSize; break;
        case 6: ++bad.abi.byteSize; break;
        case 7: bad.profile = static_cast<aniso::Profile>(UINT32_MAX); break;
        case 8: bad.requirement = static_cast<aniso::Requirement>(UINT32_MAX); break;
        case 9: bad.profile = aniso::Profile::LegacyCompatibility; break;
        case 10: bad.profile = aniso::Profile::Parity16; break;
        case 11: bad = aniso::Request::legacy(); bad.requirement = aniso::Requirement::RequireQualified; break;
    }
    for (const auto& result : {
        loader.createTextureAnisotropyV1(source, {}, bad),
        loader.createTextureAnisotropyV1("anisotropy-not-opened.png", {}, bad),
        loader.createTextureFromMemoryAnisotropyV1(source->pixels.data(), 8, 8, 4, {}, bad)}) {
        EXPECT_FALSE(result.valid);
        EXPECT_EQ(result.id, InvalidTextureId);
        EXPECT_EQ(result.error, LoaderError::InvalidParameter);
    }
    const auto good = loader.createTextureAnisotropyV1(source, {}, anisotropy(1));
    ASSERT_TRUE(good.valid);
    EXPECT_EQ(good.id, 0u);
}
INSTANTIATE_TEST_SUITE_P(MalformedRequests, AnisotropyInvalid, testing::Range(0, 12));

class AnisotropyRegistration : public Anisotropy, public testing::WithParamInterface<unsigned int> {};
TEST_P(AnisotropyRegistration, ValidRequestsAreImmutableAndDeduplicateAtCapacity) {
    auto options = anisotropyOptions();
    options.maxTextures = 2;
    DemandTextureLoader loader(options);
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(true));
    auto requested = anisotropy(GetParam());
    const auto first = loader.createTextureAnisotropyV1(source, {}, requested);
    ASSERT_TRUE(first.valid);
    const auto original = requested;
    requested.requirement = aniso::Requirement::RequireQualified;
    const auto strict = loader.createTextureAnisotropyV1(source, {}, requested);
    ASSERT_TRUE(strict.valid);
    EXPECT_NE(first.id, strict.id);
    EXPECT_EQ(loader.createTextureAnisotropyV1(source, {}, original).id, first.id);
    EXPECT_EQ(loader.createTextureAnisotropyV1(source, {}, requested).id, strict.id);
    EXPECT_TRUE(status(loader, first.id).requested == original);
    EXPECT_EQ(status(loader, first.id).texture.attempts, 0u);
    const auto full = loader.createTextureAnisotropyV1(source, {}, aniso::Request::legacy());
    EXPECT_FALSE(full.valid);
    EXPECT_EQ(full.error, LoaderError::MaxTexturesExceeded);
    EXPECT_EQ(loader.createTextureAnisotropyV1(source, {}, original).id, first.id);
}
INSTANTIATE_TEST_SUITE_P(ValidRatios, AnisotropyRegistration, testing::Values(1u, 2u, 3u, 4u, 5u, 7u, 8u, 16u));

TEST_F(Anisotropy, StatusAbiFailureDoesNotWriteAndLegacyCallsStillSubmitZero) {
    DemandTextureLoader loader(anisotropyOptions());
    const auto pixels = generateTestImage(4, 4, 4);
    TextureDesc desc;
    desc.generateMipmaps = false;
    const auto old = loader.createTextureFromMemory(pixels.data(), 4, 4, 4, desc);
    cap::Policy policy;
    policy.mipPolicy = cap::MipPolicy::Disabled;
    const auto versioned = loader.createTextureFromMemoryV1(pixels.data(), 4, 4, 4, desc, policy);
    const auto explicitOne = loader.createTextureFromMemoryAnisotropyV1(pixels.data(), 4, 4, 4, desc, anisotropy(1), policy);
    ASSERT_TRUE(old.valid && versioned.valid && explicitOne.valid);
    request(loader, {old.id, versioned.id}, 2);
    EXPECT_EQ(status(loader, old.id).texture.submittedSampler.maxAnisotropy, 0u);
    EXPECT_EQ(status(loader, versioned.id).texture.submittedSampler.maxAnisotropy, 0u);
    EXPECT_EQ(status(loader, explicitOne.id).requested.maxAnisotropy, 1u);
    EXPECT_EQ(status(loader, explicitOne.id).texture.submitted, 0u);
    for (int mode = 0; mode < 4; ++mode) {
        aniso::Status output;
        output.texture.textureId = 123;
        if (mode == 0) output.abi.version = 0;
        if (mode == 1) --output.abi.byteSize;
        if (mode == 2) ++output.abi.byteSize;
        std::array<unsigned char, sizeof(output)> before{};
        std::memcpy(before.data(), &output, sizeof(output));
        EXPECT_EQ(loader.getTextureAnisotropyStatusV1(mode == 3 ? InvalidTextureId : old.id, output),
                  mode == 3 ? Outcome::InvalidKey : Outcome::AbiMismatch);
        EXPECT_EQ(std::memcmp(before.data(), &output, sizeof(output)), 0);
    }
}

TEST_F(Anisotropy, StrictParityRejectsWithoutAllocationAndPreservesPermissiveSibling) {
    AnisotropyFaults faults;
    DemandTextureLoader loader(anisotropyOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(true));
    const auto strict = loader.createTextureAnisotropyV1(source, {}, aniso::Request::parity());
    const auto allowed = loader.createTextureAnisotropyV1(source, {}, aniso::Request::legacy());
    ASSERT_TRUE(strict.valid && allowed.valid);
    request(loader, {strict.id}, 0);
    auto rejected = status(loader, strict.id);
    EXPECT_EQ(rejected.requirementRejected, 1u);
    EXPECT_EQ(rejected.texture.primary.outcome, Outcome::Unsupported);
    EXPECT_EQ(rejected.texture.primary.operation, cap::Operation::QualifySampler);
    EXPECT_EQ(rejected.texture.primary.rawHipError, 0);
    EXPECT_EQ(rejected.texture.submitted, 0u);
    EXPECT_EQ(rejected.texture.published, 0u);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 0u);
    EXPECT_EQ(source->reads, 0u);
    request(loader, {strict.id, allowed.id}, 1);
    EXPECT_EQ(status(loader, strict.id).texture.published, 0u);
    EXPECT_EQ(status(loader, allowed.id).texture.published, 1u);
    EXPECT_EQ(status(loader, allowed.id).qualification, aniso::Qualification::Unqualified);
    const auto payload = loader.getTotalTextureMemory();
    request(loader, {strict.id}, 0);
    EXPECT_EQ(loader.getTotalTextureMemory(), payload);
    EXPECT_EQ(loader.getResidentTextureCount(), 1u);
}

TEST_F(Anisotropy, ReturnedMismatchIsNotRewrittenOrQualifiedAndSurvivesPublicationRetry) {
    AnisotropyFaults faults;
    faults.state->overrideReturnedAnisotropy(1);
    DemandTextureLoader loader(anisotropyOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(true));
    const auto handle = loader.createTextureAnisotropyV1(source, {}, anisotropy(16));
    ASSERT_TRUE(handle.valid);
    request(loader, {handle.id}, 1);
    const auto s = status(loader, handle.id);
    EXPECT_EQ(s.requested.maxAnisotropy, 16u);
    EXPECT_EQ(s.texture.submittedSampler.maxAnisotropy, 16u);
    EXPECT_EQ(s.texture.returnedSampler.maxAnisotropy, 1u);
    EXPECT_TRUE(s.limitations & aniso::DescriptorMismatch);
    EXPECT_EQ(s.qualification, aniso::Qualification::Unqualified);
    EXPECT_EQ(s.texture.state, cap::State::Degraded);
    loader.unloadTexture(handle.id);
    loader.launchPrepare();
    faults.state->fail(HipOperation::PublishMappings, hipErrorInvalidValue);
    request(loader, {handle.id}, 1);
    EXPECT_EQ(status(loader, handle.id).texture.state, cap::State::Failed);
    loader.launchPrepare();
    EXPECT_EQ(status(loader, handle.id).texture.state, cap::State::Degraded);
}

class AnisotropyFailure : public Anisotropy,
    public testing::WithParamInterface<std::tuple<HipOperation, hipError_t, bool>> {};
TEST_P(AnisotropyFailure, SamplerFailureDoesNotPoisonOtherRatiosAndRetryPreservesRequest) {
    const auto [operation, error, mipmapped] = GetParam();
    AnisotropyFaults faults;
    DemandTextureLoader loader(anisotropyOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
    cap::Policy policy;
    if (!mipmapped) policy.mipPolicy = cap::MipPolicy::Disabled;
    const auto first = loader.createTextureAnisotropyV1(source, {}, anisotropy(16), policy);
    const auto second = loader.createTextureAnisotropyV1(source, {}, anisotropy(1), policy);
    ASSERT_TRUE(first.valid && second.valid);
    faults.state->fail(operation, error);
    request(loader, {first.id, second.id}, 1);
    const auto failed = status(loader, first.id);
    EXPECT_EQ(failed.texture.state, cap::State::Failed);
    EXPECT_EQ(failed.texture.primary.rawHipError, static_cast<int32_t>(error));
    EXPECT_EQ(failed.texture.published, 0u);
    EXPECT_EQ(status(loader, second.id).texture.published, 1u);
    request(loader, {first.id}, 1);
    EXPECT_EQ(status(loader, first.id).texture.submittedSampler.maxAnisotropy, 16u);
    EXPECT_EQ(loader.getTotalTextureMemory(), mipmapped ? 340u : 256u);
}
INSTANTIATE_TEST_SUITE_P(SamplerErrors, AnisotropyFailure,
    testing::Combine(testing::Values(HipOperation::CreateSampler, HipOperation::ReadSampler),
                     testing::Values(hipErrorNotSupported, hipErrorOutOfMemory, hipErrorInvalidValue), testing::Bool()));

TEST_F(Anisotropy, ProbeConfigurationIsolationAndCleanupErrorRetention) {
    AnisotropyFaults faults;
    DemandTextureLoader loader(anisotropyOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(true));
    const auto rejected = loader.createTextureAnisotropyV1(source, {}, anisotropy(16));
    const auto accepted = loader.createTextureAnisotropyV1(source, {}, aniso::Request::legacy());
    ASSERT_TRUE(rejected.valid && accepted.valid);
    faults.state->fail(HipOperation::ProbeCreateSampler, hipErrorNotSupported);
    faults.state->fail(HipOperation::ProbeFree, hipErrorInvalidValue);
    request(loader, {rejected.id, accepted.id}, 1);
    auto s = status(loader, rejected.id);
    EXPECT_EQ(s.texture.primary.operation, cap::Operation::ProbeCreateSampler);
    EXPECT_EQ(s.texture.primary.rawHipError, hipErrorNotSupported);
    EXPECT_EQ(s.texture.cleanup.operation, cap::Operation::ProbeFree);
    EXPECT_EQ(s.texture.cleanup.rawHipError, hipErrorInvalidValue);
    EXPECT_EQ(status(loader, accepted.id).texture.published, 1u);
    EXPECT_EQ(faults.calls(HipOperation::ProbeAllocate), 2u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 1360u + 336u);
    loader.unloadAll();
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    // Retrying a cached unsupported configuration cannot qualify it.
    request(loader, {rejected.id}, 0);
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    request(loader, {accepted.id}, 1);
    EXPECT_EQ(loader.getTotalTextureMemory(), 1360u);
}

TEST_F(Anisotropy, InjectedUnsupportedSamplerRetainsLiveLegacySiblingAndFailureIdentity) {
    AnisotropyFaults faults;
    DemandTextureLoader loader(anisotropyOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(true));
    const auto live = loader.createTextureV1(source, {});
    const auto rejected = loader.createTextureAnisotropyV1(source, {}, anisotropy(16));
    ASSERT_TRUE(live.valid && rejected.valid);
    request(loader, {live.id}, 1);
    faults.state->fail(HipOperation::CreateSampler, hipErrorNotSupported);
    request(loader, {rejected.id}, 0);
    const auto failure = status(loader, rejected.id);
    EXPECT_EQ(failure.samplerSupport, cap::Support::Unsupported);
    EXPECT_EQ(failure.qualification, aniso::Qualification::Unqualified);
    EXPECT_EQ(failure.texture.primary.rawHipError, hipErrorNotSupported);
    EXPECT_EQ(failure.texture.primary.operation, cap::Operation::CreateSampler);
    EXPECT_EQ(failure.texture.submittedSampler.maxAnisotropy, 16u);
    EXPECT_EQ(failure.texture.returned, 0u);
    EXPECT_EQ(failure.texture.published, 0u);
    EXPECT_EQ(loader.getLastError(), LoaderError::Unsupported);
    EXPECT_EQ(status(loader, live.id).texture.published, 1u);
    EXPECT_EQ(loader.getResidentTextureCount(), 1u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 1360u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), 1u);
}

TEST_F(Anisotropy, InjectedReadbackDoesNotRewriteRequestOrClaimEffectiveBehavior) {
    AnisotropyFaults faults;
    faults.state->overrideReturnedAnisotropy(16);
    DemandTextureLoader loader(anisotropyOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(true));
    const auto handle = loader.createTextureAnisotropyV1(source, {}, aniso::Request::legacy());
    ASSERT_TRUE(handle.valid);
    request(loader, {handle.id}, 1);
    const auto snapshot = status(loader, handle.id);
    EXPECT_EQ(snapshot.requested.maxAnisotropy, 0u);
    EXPECT_EQ(snapshot.texture.submittedSampler.maxAnisotropy, 0u);
    EXPECT_EQ(snapshot.texture.returnedSampler.maxAnisotropy, 16u);
    EXPECT_TRUE(snapshot.limitations & aniso::DescriptorMismatch);
    EXPECT_EQ(snapshot.qualification, aniso::Qualification::Unqualified);
    EXPECT_EQ(snapshot.samplerSupport, cap::Support::OperationSupported);
    EXPECT_EQ(snapshot.texture.state, cap::State::Resident);
}

TEST_F(Anisotropy, AllowedBaseFallbackIsExplicitAndNotAnisotropyQualification) {
    AnisotropyFaults faults;
    DemandTextureLoader loader(anisotropyOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
    cap::Policy policy;
    policy.mipPolicy = cap::MipPolicy::AllowBaseLevelFallback;
    const auto handle = loader.createTextureAnisotropyV1(source, {}, anisotropy(8), policy);
    ASSERT_TRUE(handle.valid);
    faults.state->fail(HipOperation::ProbeAllocate, hipErrorNotSupported);
    request(loader, {handle.id}, 1);
    const auto s = status(loader, handle.id);
    EXPECT_EQ(s.texture.state, cap::State::Degraded);
    EXPECT_EQ(s.texture.resource, cap::Resource::Array);
    EXPECT_EQ(s.texture.resourceLevels, 1u);
    EXPECT_EQ(s.texture.originalLevels, 4u);
    EXPECT_EQ(s.texture.submittedSampler.maxAnisotropy, 8u);
    EXPECT_TRUE(s.limitations & aniso::BaseLevelFallback);
    EXPECT_TRUE(s.limitations & aniso::SingleLevel);
    EXPECT_TRUE(s.limitations & aniso::UnqualifiedBehavior);
    EXPECT_EQ(s.texture.fallback.rawHipError, hipErrorNotSupported);
}

TEST_F(Anisotropy, ConcurrentRegistrationAndPriorityMutationRetainCompleteKeys) {
    DemandTextureLoader loader(anisotropyOptions());
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
    std::vector<std::future<TextureHandle>> workers;
    for (unsigned int i = 0; i < 16; ++i)
        workers.push_back(std::async(std::launch::async, [&loader, source, i] {
            return loader.createTextureAnisotropyV1(source, {}, anisotropy(i % 2 ? 16 : 1));
        }));
    std::set<uint32_t> ids;
    for (auto& worker : workers) {
        const auto handle = worker.get();
        ASSERT_TRUE(handle.valid);
        ids.insert(handle.id);
    }
    ASSERT_EQ(ids.size(), 2u);
    request(loader, {ids.begin(), ids.end()}, 2, true);
    EXPECT_EQ(loader.getTotalTextureMemory(), 340u);
    for (uint32_t id : ids) {
        auto s = status(loader, id);
        loader.updateEvictionPriority(id, EvictionPriority::KeepResident);
        TextureDesc desc;
        desc.evictionPriority = EvictionPriority::KeepResident;
        EXPECT_EQ(loader.createTextureAnisotropyV1(source, desc, s.requested).id, id);
    }
}

#ifdef USE_OIIO
TEST_F(Anisotropy, FilenameVariantsUseTheSameBackingAndRetainLegacyEntryPoint) {
    struct File {
        std::filesystem::path path;
        ~File() { std::error_code error; std::filesystem::remove(path, error); }
    } file{std::filesystem::current_path() / ("anisotropy-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".exr")};
    auto output = OIIO::ImageOutput::create(file.path.string());
    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(output->open(file.path.string(), OIIO::ImageSpec(4, 4, 4, OIIO::TypeDesc::FLOAT)));
    std::vector<float> pixels(4 * 4 * 4, .625f);
    ASSERT_TRUE(output->write_image(OIIO::TypeDesc::FLOAT, pixels.data()));
    ASSERT_TRUE(output->close());
    output.reset();
    DemandTextureLoader loader(anisotropyOptions());
    TextureDesc desc;
    const auto old = loader.createTexture(file.path.string(), desc);
    cap::Policy policy;
    policy.mipPolicy = cap::MipPolicy::LegacyCompatibility;
    const auto legacy = loader.createTextureAnisotropyV1(file.path.string(), desc, aniso::Request::legacy(), policy);
    const auto explicitRequest = loader.createTextureAnisotropyV1(file.path.string(), desc, anisotropy(16), policy);
    ASSERT_TRUE(old.valid && legacy.valid && explicitRequest.valid);
    EXPECT_EQ(old.id, legacy.id);
    EXPECT_NE(old.id, explicitRequest.id);
    request(loader, {explicitRequest.id, old.id}, 2);
    EXPECT_EQ(resource(loader, old.id).res.mipmap.mipmap, resource(loader, explicitRequest.id).res.mipmap.mipmap);
    EXPECT_EQ(loader.getTotalTextureMemory(), 336u);
    EXPECT_EQ(status(loader, old.id).texture.submittedSampler.maxAnisotropy, 0u);
}
#endif
} // namespace
} }
