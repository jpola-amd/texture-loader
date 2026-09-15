// SPDX-License-Identifier: MIT
#include <DemandLoading/ContractState.h>
#include "ImageDataTestUtils.h"
#include <gtest/gtest.h>

#include <limits>
#include <future>
#include <thread>
#include <vector>

using namespace hip_demand::contract_v1;

namespace {
constexpr GpuKey key{0, 1, 1};
const ImageIdentity image{7, 1, ImageNamespace::Content, 0};

PublishedTexture publication() {
    PublishedTexture result;
    result.key = key;
    result.revision = 1;
    result.textureObject = 123;
    result.mips = {32, 16, 6, 2, 8, 4, 4};
    result.state = RegistrationState::Live;
    result.residency = Outcome::Success;
    return result;
}

void makeReady(ResourceLifecycle& resource) {
    ASSERT_EQ(resource.reserve(64), Outcome::Success);
    ASSERT_EQ(resource.allocated(), Outcome::Success);
    ASSERT_EQ(resource.uploaded(), Outcome::Success);
    ASSERT_EQ(resource.samplersCreated(), Outcome::Success);
}
} // namespace

TEST(SharedIdentity, DescriptorFieldsAreIndependent) {
    const SamplerDesc original;
    for (unsigned int field = 0; field != 13; ++field) {
        SamplerDesc changed = original;
        switch (field) {
            case 0: changed.addressMode[0] = AddressMode::Clamp; break;
            case 1: changed.addressMode[1] = AddressMode::Mirror; break;
            case 2: changed.spatialFilter = FilterMode::Point; break;
            case 3: changed.mipFilter = FilterMode::Point; break;
            case 4: changed.normalizedCoords = 0; break;
            case 5: changed.sRGB = 1; break;
            case 6: changed.generateMipmaps = 0; break;
            case 7: changed.maxMipLevels = 3; break;
            case 8: changed.priority = Priority::KeepResident; break;
            case 9: changed.maxAnisotropy = 16; break;
            case 10: changed.mipPolicy = MipPolicy::AllowBaseLevelFallback; break;
            case 11: changed.samplingPolicy = SamplingPolicy::AllowCoarsePreview; break;
            case 12: changed.abi.version = 2; break;
        }
        EXPECT_FALSE(original == changed) << field;
    }
    EXPECT_TRUE(original == SamplerDesc{});
}

TEST(SharedIdentity, StorageCompatibilityIsConservative) {
    SamplerDesc desc;
    const StorageKey original = storageKey(image, 16, desc);
    desc.addressMode[0] = AddressMode::Clamp;
    desc.mipFilter = FilterMode::Point;
    desc.maxAnisotropy = 16;
    desc.priority = Priority::KeepResident;
    desc.mipPolicy = MipPolicy::AllowBaseLevelFallback;
    EXPECT_TRUE(original == storageKey(image, 16, desc));
    EXPECT_FALSE(original == storageKey(image, 4, desc));
    desc.sRGB = 1;
    EXPECT_FALSE(original == storageKey(image, 16, desc));
    desc.sRGB = 0;
    desc.generateMipmaps = 0;
    EXPECT_FALSE(original == storageKey(image, 16, desc));
    desc.generateMipmaps = 1;
    desc.maxMipLevels = 2;
    EXPECT_FALSE(original == storageKey(image, 16, desc));
}

TEST(SharedIdentity, NamespacesRevisionsAndFullRequestKeys) {
    for (const auto space : {ImageNamespace::Filename, ImageNamespace::SourceObject, ImageNamespace::Memory}) {
        auto other = image;
        other.nameSpace = space;
        EXPECT_FALSE(image == other);
    }
    auto revised = image;
    ++revised.revision;
    EXPECT_FALSE(image == revised);
    RequestKey request{key, 1, 3, 0};
    EXPECT_TRUE(request == (RequestKey{key, 1, 3, 0}));
    EXPECT_FALSE(request == (RequestKey{key, 2, 3, 0}));
    EXPECT_FALSE(request == (RequestKey{key, 1, 4, 0}));
    EXPECT_FALSE(request == (RequestKey{{0, 2, 1}, 1, 3, 0}));
    EXPECT_FALSE(request == (RequestKey{{0, 1, 2}, 1, 3, 0}));
}

TEST(SharedIdentity, AcquisitionsOwnButCopiedKeysDoNot) {
    static_assert(!std::is_copy_constructible<RegistrationLease>::value);
    Registration registration(key, image, {});
    auto a = registration.acquire();
    auto b = registration.acquire();
    EXPECT_EQ(a.outcome, Outcome::Success);
    EXPECT_EQ(b.outcome, Outcome::Success);
    EXPECT_TRUE(registration.matches(image, {}));
    const auto copiedKey = a.lease.key();
    EXPECT_EQ(registration.ownerCount(), 2);
    EXPECT_EQ(a.lease.release(), Outcome::Success);
    EXPECT_EQ(a.lease.release(), Outcome::InvalidKey);
    EXPECT_EQ(registration.state(), RegistrationState::Live);
    EXPECT_TRUE(copiedKey == b.lease.key());
    {
        RegistrationLease moved = std::move(b.lease);
        EXPECT_FALSE(valid(b.lease.key()));
        EXPECT_EQ(registration.ownerCount(), 1);
    }
    EXPECT_EQ(registration.ownerCount(), 0);
    EXPECT_EQ(registration.state(), RegistrationState::Retiring);
    EXPECT_EQ(registration.acquire().outcome, Outcome::InvalidKey);
}

TEST(SharedIdentity, LeaseMovesReleasePreviousOwnership) {
    Registration a(key, image, {});
    Registration b({1, 1, 1}, image, {});
    auto first = a.acquire();
    auto second = b.acquire();
    first.lease = std::move(second.lease);
    EXPECT_EQ(a.state(), RegistrationState::Retiring);
    EXPECT_EQ(b.ownerCount(), 1);
    EXPECT_EQ(first.lease.key().slot, 1);
}

TEST(SharedIdentity, ConcurrentAcquisitionsKeepStableOwnership) {
    Registration registration(key, image, {});
    auto anchor = registration.acquire();
    std::vector<std::thread> workers;
    for (int i = 0; i != 8; ++i) {
        workers.emplace_back([&] {
            for (int iteration = 0; iteration != 1000; ++iteration) {
                auto acquired = registration.acquire();
                EXPECT_EQ(acquired.outcome, Outcome::Success);
                EXPECT_TRUE(acquired.lease.key() == key);
            }
        });
    }
    for (auto& worker : workers)
        worker.join();
    EXPECT_EQ(registration.ownerCount(), 1);
    EXPECT_EQ(registration.state(), RegistrationState::Live);
}

TEST(SharedIdentity, InvalidRegistrationsGainNoOwnership) {
    EXPECT_THROW(Registration({}, image, {}), std::invalid_argument);
    EXPECT_THROW(Registration(key, {}, {}), std::invalid_argument);
    auto desc = SamplerDesc{};
    desc.abi.version = 0;
    EXPECT_THROW(Registration(key, image, desc), std::invalid_argument);
    Registration registration(key, image, {});
    EXPECT_FALSE(registration.matches(image, desc));
}

TEST(SharedIdentity, CountersNeverWrap) {
    NonWrappingCounter counter(UINT64_MAX - 1);
    uint64_t output = 0;
    EXPECT_EQ(counter.next(output), Outcome::Success);
    EXPECT_EQ(output, UINT64_MAX);
    for (int i = 0; i != 2; ++i) {
        EXPECT_EQ(counter.next(output), Outcome::IdentityExhausted);
        EXPECT_EQ(output, 0);
    }
    uint32_t generation = UINT32_MAX - 1;
    EXPECT_EQ(advanceGeneration(generation), Outcome::Success);
    EXPECT_EQ(advanceGeneration(generation), Outcome::IdentityExhausted);
    EXPECT_EQ(generation, UINT32_MAX);
    uint64_t a = 0, b = 0;
    EXPECT_EQ(allocateLoaderIncarnation(a), Outcome::Success);
    EXPECT_EQ(allocateLoaderIncarnation(b), Outcome::Success);
    EXPECT_NE(a, 0);
    EXPECT_GT(b, a);
}

TEST(SharedIdentity, ConcurrentCountersIssueUniqueValues) {
    NonWrappingCounter counter;
    std::vector<uint64_t> values(128);
    std::vector<std::thread> workers;
    for (size_t i = 0; i != values.size(); ++i)
        workers.emplace_back([&, i] { EXPECT_EQ(counter.next(values[i]), Outcome::Success); });
    for (auto& worker : workers)
        worker.join();
    EXPECT_EQ(std::set<uint64_t>(values.begin(), values.end()).size(), values.size());
}

TEST(SharedOutcomes, InvalidDefaultsDoNotAliasZero) {
    RegistrationResult failed;
    EXPECT_EQ(failed.key.slot, InvalidSlot);
    EXPECT_FALSE(failed.succeeded());
    EXPECT_FALSE(valid(failed.key));
    RegistrationResult success{key, Outcome::Success};
    EXPECT_TRUE(success.succeeded());
    EXPECT_EQ(success.key.slot, 0);
    for (uint32_t value = 1; value <= static_cast<uint32_t>(Outcome::RuntimeFailure); ++value) {
        failed.outcome = static_cast<Outcome>(value);
        EXPECT_FALSE(failed.succeeded());
    }
    failed.outcome = Outcome::Success;
    EXPECT_FALSE(failed.succeeded());
}

TEST(SharedOutcomes, RetryabilityIsExplicitForEveryOutcome) {
    for (uint32_t value = 0; value <= static_cast<uint32_t>(Outcome::RuntimeFailure); ++value) {
        const auto outcome = static_cast<Outcome>(value);
        const bool expected = outcome == Outcome::Pending || outcome == Outcome::Deferred ||
                              outcome == Outcome::RequestOverflow;
        EXPECT_EQ(retryable(outcome), expected) << value;
    }
    EXPECT_FALSE(retryable(static_cast<Outcome>(UINT32_MAX)));
}

TEST(SharedOutcomes, CompletionIsNotSuccessAndErrorsAreIndependent) {
    BatchResult batch;
    EXPECT_FALSE(batch.succeeded());
    batch.completion = Completion::Complete;
    batch.outcome = Outcome::SourceFailure;
    EXPECT_FALSE(batch.succeeded());
    batch.outcome = Outcome::Success;
    EXPECT_TRUE(batch.succeeded());
    batch.completion = Completion::Running;
    EXPECT_FALSE(batch.succeeded());
    OperationStatus status;
    status.recordPrimary({Outcome::DeviceOutOfMemory, Operation::Allocate, 2});
    status.recordPrimary({Outcome::RuntimeFailure, Operation::Upload, 3});
    status.recordCleanup({Outcome::RuntimeFailure, Operation::Destroy, 7});
    status.recordCleanup({Outcome::RuntimeFailure, Operation::Destroy, 8});
    EXPECT_EQ(status.primary.outcome, Outcome::DeviceOutOfMemory);
    EXPECT_EQ(status.primary.operation, Operation::Allocate);
    EXPECT_EQ(status.primary.rawError, 2);
    EXPECT_EQ(status.cleanup.rawError, 7);
    EXPECT_EQ(BatchResult({Completion::Complete, Outcome::Success}).outcome, Outcome::Success);
}

TEST(SharedOutcomes, TerminalKeysAndResidencyNeverGenerateDemand) {
    auto texture = publication();
    DeviceContext context{{Version, sizeof(DeviceContext)}, 1, &texture, 1, 0};
    for (const GpuKey invalid : {GpuKey{}, GpuKey{1, 1, 1}, GpuKey{0, 2, 1}, GpuKey{0, 1, 2}}) {
        const auto decision = resolve(context, invalid, 1, 0, {});
        EXPECT_EQ(decision.outcome, Outcome::InvalidKey);
        EXPECT_FALSE(decision.needsRequest());
    }
    EXPECT_FALSE(resolve(context, key, 2, 0, {}).needsRequest());
    texture.state = RegistrationState::Retiring;
    EXPECT_FALSE(resolve(context, key, 1, 0, {}).needsRequest());
    texture.state = RegistrationState::Live;
    for (const auto outcome : {Outcome::SourceFailure, Outcome::Unsupported, Outcome::DeviceOutOfMemory,
                               Outcome::Cancelled, Outcome::NoProgress, Outcome::DemandTooLarge}) {
        texture.residency = outcome;
        const auto decision = resolve(context, key, 1, 0, {});
        EXPECT_EQ(decision.outcome, outcome);
        EXPECT_FALSE(decision.needsRequest());
    }
    texture.residency = Outcome::Success;
    EXPECT_TRUE(resolve(context, key, 1, 3, {}).contributesStrictSample());
}

TEST(SharedMips, SuffixDimensionsAndPolicyLimits) {
    for (const MipLayout layout : {
             MipLayout{32, 16, 6, 2, 8, 4, 4}, MipLayout{32, 16, 4, 2, 8, 4, 2},
             MipLayout{7, 3, 3, 1, 3, 1, 2}, MipLayout{1, 31, 5, 3, 1, 3, 2},
             MipLayout{31, 1, 5, 3, 3, 1, 2}, MipLayout{1, 1, 1, 0, 1, 1, 1},
             MipLayout{8, 8, 4, 0, 8, 8, 4}, MipLayout{8, 8, 4, InvalidSlot, 0, 0, 0},
             MipLayout{UINT32_MAX, 1, 32, 31, 1, 1, 1}}) {
        EXPECT_EQ(validate(layout), Outcome::Success);
    }
    EXPECT_EQ(fullMipCount(0, 4), 0);
    EXPECT_EQ(fullMipCount(1, 1), 1);
    EXPECT_EQ(fullMipCount(7, 3), 3);
}

TEST(SharedMips, InvalidRangesAndUnallocatedFineLevelsAreRejected) {
    for (const MipLayout layout : {
             MipLayout{}, MipLayout{0, 1, 1, 0, 1, 1, 1}, MipLayout{4, 4, 4, 0, 4, 4, 4},
             MipLayout{4, 4, 3, 3, 1, 1, 0}, MipLayout{4, 4, 3, 1, 4, 4, 2},
             MipLayout{4, 4, 3, 1, 2, 2, 1}, MipLayout{4, 4, 3, InvalidSlot, 4, 4, 3}}) {
        EXPECT_EQ(validate(layout), Outcome::InvalidInput);
    }
}

struct LodCase { float lod; FilterMode filter; uint32_t first; uint32_t last; float clamped; };
class SharedMipSelection : public testing::TestWithParam<LodCase> {};
TEST_P(SharedMipSelection, OriginalRequirementsBeforeResidencyClamp) {
    const auto test = GetParam();
    const auto requirement = requiredLevels(test.lod, 5, test.filter);
    EXPECT_EQ(requirement.outcome, Outcome::Success);
    EXPECT_EQ(requirement.levels.first, test.first);
    EXPECT_EQ(requirement.levels.last, test.last);
    EXPECT_FLOAT_EQ(requirement.clampedOriginalLod, test.clamped);
}
INSTANTIATE_TEST_SUITE_P(OriginalLod, SharedMipSelection, testing::Values(
    LodCase{-10, FilterMode::Point, 0, 0, 0}, LodCase{10, FilterMode::Linear, 4, 4, 4},
    LodCase{2, FilterMode::Point, 2, 2, 2}, LodCase{2, FilterMode::Linear, 2, 2, 2},
    LodCase{1.25f, FilterMode::Point, 1, 1, 1.25f}, LodCase{1.75f, FilterMode::Point, 2, 2, 1.75f},
    LodCase{1.5f, FilterMode::Point, 1, 2, 1.5f}, LodCase{1.25f, FilterMode::Linear, 1, 2, 1.25f},
    LodCase{1.75f, FilterMode::Linear, 1, 2, 1.75f}));

TEST(SharedMips, InvalidLodsAndUnqualifiedAnisotropyAreExplicit) {
    for (float lod : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()})
        EXPECT_EQ(requiredLevels(lod, 5, FilterMode::Point).outcome, Outcome::InvalidInput);
    EXPECT_EQ(requiredLevels(0, 0, FilterMode::Point).outcome, Outcome::InvalidInput);
    EXPECT_EQ(requiredLevels(0, 33, FilterMode::Point).outcome, Outcome::InvalidInput);
    for (uint32_t ratio : {2u, 3u, 4u, 5u, 7u, 8u, 16u})
        EXPECT_EQ(requiredLevels(2, 5, FilterMode::Linear, ratio).outcome, Outcome::Unsupported);
    for (uint32_t ratio : {0u, 17u, UINT32_MAX})
        EXPECT_EQ(requiredLevels(2, 5, FilterMode::Linear, ratio).outcome, Outcome::InvalidInput);
    const auto singleton = requiredLevels(10, 1, FilterMode::Linear);
    EXPECT_EQ(singleton.levels.first, 0);
    EXPECT_EQ(singleton.levels.last, 0);
}

TEST(SharedMips, StrictMissPreviewAndExplicitRebaseAreDistinct) {
    const auto layout = publication().mips;
    auto decision = evaluate(layout, requiredLevels(2.75f, 6, FilterMode::Linear), SamplingPolicy::Strict);
    EXPECT_EQ(decision.validity, SampleValidity::Complete);
    EXPECT_FLOAT_EQ(decision.resourceLod, 0.75f);
    EXPECT_FALSE(decision.needsRequest());
    decision = evaluate(layout, requiredLevels(1.75f, 6, FilterMode::Linear), SamplingPolicy::Strict);
    EXPECT_EQ(decision.validity, SampleValidity::Missing);
    EXPECT_EQ(decision.required.first, 1);
    EXPECT_TRUE(decision.needsRequest());
    EXPECT_FALSE(decision.contributesStrictSample());
    decision = evaluate(layout, requiredLevels(1.75f, 6, FilterMode::Point), SamplingPolicy::Strict);
    EXPECT_EQ(decision.validity, SampleValidity::Complete);
    decision = evaluate(layout, requiredLevels(0, 6, FilterMode::Linear), SamplingPolicy::AllowCoarsePreview);
    EXPECT_EQ(decision.validity, SampleValidity::CoarsePreview);
    EXPECT_TRUE(decision.needsRequest());
    EXPECT_FALSE(decision.contributesStrictSample());
    EXPECT_EQ(evaluate(layout, {Outcome::Success, {3, 2}, 2}, SamplingPolicy::Strict).outcome,
              Outcome::InvalidInput);
    EXPECT_EQ(evaluate(layout, {Outcome::Success, {2, 6}, 2}, SamplingPolicy::Strict).outcome,
              Outcome::InvalidInput);
}

TEST(SharedMips, EvictionKeepsIdentityAndRequestsOriginalDetail) {
    auto texture = publication();
    DeviceContext context{{Version, sizeof(DeviceContext)}, 1, &texture, 1, 0};
    texture.mips = {32, 16, 6, InvalidSlot, 0, 0, 0};
    texture.textureObject = 0;
    texture.residency = Outcome::Deferred;
    const auto decision = resolve(context, key, 1, 3, {});
    EXPECT_TRUE(texture.key == key);
    EXPECT_EQ(decision.validity, SampleValidity::Missing);
    EXPECT_EQ(decision.required.first, 3);
    EXPECT_TRUE(decision.needsRequest());
}

TEST(SharedMips, RequiredPolicyCannotBeHiddenByReducedOriginalLevelCount) {
    auto texture = publication();
    texture.mips = {32, 16, 1, 0, 32, 16, 1};
    DeviceContext context{{Version, sizeof(DeviceContext)}, 1, &texture, 1, 0};
    SamplerDesc desc;
    for (const auto policy : {MipPolicy::Required, MipPolicy::AllowBaseLevelFallback}) {
        desc.mipPolicy = policy;
        auto decision = resolve(context, key, 1, 4, desc);
        EXPECT_EQ(decision.outcome, Outcome::Unsupported);
        EXPECT_FALSE(decision.contributesStrictSample());
        EXPECT_FALSE(decision.needsRequest());
    }
    desc.mipPolicy = MipPolicy::Required;
    desc.maxMipLevels = 1;
    EXPECT_TRUE(resolve(context, key, 1, 4, desc).contributesStrictSample());
    desc = {};
    desc.mipPolicy = MipPolicy::Disabled;
    EXPECT_TRUE(resolve(context, key, 1, 4, desc).contributesStrictSample());
    texture.mips = {32, 16, 4, 2, 8, 4, 2};
    desc = {};
    desc.maxMipLevels = 4;
    EXPECT_TRUE(resolve(context, key, 1, 3, desc).contributesStrictSample());
    desc = {};
    EXPECT_EQ(resolve(context, key, 1, 3, desc).outcome, Outcome::Unsupported);
}

TEST(SharedLifetime, PublicationRequiresUploadObjectsAndFinishedWorkers) {
    BudgetLedger ledger(256);
    ResourceLifecycle resource(ledger, key, 1);
    EXPECT_EQ(resource.uploaded(), Outcome::InvalidTransition);
    EXPECT_EQ(resource.publish(key, 1), Outcome::InvalidTransition);
    EXPECT_EQ(resource.reserve(64), Outcome::Success);
    uint64_t worker = 0;
    EXPECT_EQ(resource.beginWorker(worker), Outcome::Success);
    EXPECT_EQ(resource.samplersCreated(), Outcome::InvalidTransition);
    EXPECT_EQ(resource.allocated(), Outcome::Success);
    EXPECT_EQ(resource.publish(key, 1), Outcome::InvalidTransition);
    EXPECT_EQ(resource.uploaded(), Outcome::Success);
    EXPECT_EQ(resource.samplersCreated(), Outcome::Success);
    EXPECT_EQ(resource.publish(key, 1), Outcome::InvalidTransition);
    EXPECT_EQ(resource.completeWorker(worker), Outcome::Success);
    EXPECT_EQ(resource.completeWorker(worker), Outcome::InvalidInput);
    EXPECT_EQ(resource.publish(key, 2), Outcome::InvalidKey);
    EXPECT_EQ(resource.publish({0, 1, 2}, 1), Outcome::InvalidKey);
    EXPECT_EQ(resource.publish(key, 1), Outcome::Success);
    EXPECT_EQ(ledger.charged(Charge::Pending), 0);
    EXPECT_EQ(ledger.charged(Charge::Resident), 64);
    EXPECT_EQ(ledger.total(), 64);
}

TEST(SharedLifetime, EveryOldConsumerAndMappingMustBeFenced) {
    BudgetLedger ledger(256);
    ResourceLifecycle old(ledger, key, 1), replacement(ledger, key, 2);
    makeReady(old);
    ASSERT_EQ(old.publish(key, 1), Outcome::Success);
    uint64_t first = 0, second = 0;
    EXPECT_EQ(old.beginConsumer(first), Outcome::Success);
    EXPECT_EQ(old.beginConsumer(second), Outcome::Success);
    makeReady(replacement);
    EXPECT_EQ(replacement.publish(key, 2), Outcome::Success);
    EXPECT_EQ(old.retire(), Outcome::Success);
    EXPECT_EQ(ledger.total(), 128);
    EXPECT_EQ(ledger.peak(), 128);
    EXPECT_EQ(ledger.charged(Charge::Retiring), 64);
    EXPECT_EQ(old.destroyed(), Outcome::Pending);
    EXPECT_EQ(old.completeConsumer(first), Outcome::Success);
    EXPECT_EQ(old.invalidateMapping(), Outcome::Success);
    EXPECT_EQ(old.destroyed(), Outcome::Pending);
    EXPECT_EQ(old.completeConsumer(second), Outcome::Success);
    EXPECT_EQ(old.destroyed({Outcome::RuntimeFailure, Operation::Destroy, 7}), Outcome::RuntimeFailure);
    EXPECT_EQ(ledger.total(), 128);
    EXPECT_EQ(old.destroyed(), Outcome::Success);
    EXPECT_EQ(ledger.total(), 64);
    EXPECT_EQ(old.destroyed(), Outcome::InvalidTransition);
    EXPECT_EQ(old.beginConsumer(first), Outcome::InvalidTransition);
}

TEST(SharedLifetime, CancelledWorkCannotRepublishAndRemainsCharged) {
    BudgetLedger ledger(64);
    ResourceLifecycle resource(ledger, key, 1);
    EXPECT_EQ(resource.reserve(64), Outcome::Success);
    uint64_t worker = 0;
    EXPECT_EQ(resource.beginWorker(worker), Outcome::Success);
    EXPECT_EQ(resource.cancel(), Outcome::Success);
    EXPECT_EQ(resource.allocated(), Outcome::InvalidTransition);
    EXPECT_EQ(resource.publish(key, 1), Outcome::InvalidTransition);
    EXPECT_EQ(resource.destroyed(), Outcome::Pending);
    EXPECT_EQ(ledger.charged(Charge::Retiring), 64);
    EXPECT_EQ(resource.completeWorker(worker), Outcome::Success);
    EXPECT_EQ(resource.destroyed(), Outcome::Success);
    EXPECT_EQ(ledger.total(), 0);
    EXPECT_EQ(resource.status().primary.outcome, Outcome::Cancelled);
}

TEST(SharedLifetime, BlockedClosureRetainsSourceUntilWorkerCompletion) {
    auto source = std::make_shared<hip_demand::test::TypedImageSource>(hip_demand::test::makeSource(0, 4));
    std::weak_ptr<hip_demand::ImageSource> lifetime = source;
    auto registration = std::make_unique<Registration>(key, image, SamplerDesc{}, source);
    auto acquisition = registration->acquire();
    BudgetLedger ledger(64);
    ResourceLifecycle resource(ledger, key, 1);
    ASSERT_EQ(resource.reserve(64), Outcome::Success);
    uint64_t token = 0;
    ASSERT_EQ(resource.beginWorker(token), Outcome::Success);
    std::promise<void> started, unblock;
    auto gate = unblock.get_future();
    std::thread worker([retainedSource = source, &started, &gate] {
        started.set_value();
        gate.wait();
        std::vector<char> pixels(retainedSource->pixels.size());
        EXPECT_TRUE(retainedSource->readMipLevel(pixels.data(), 0, 3, 2, nullptr));
    });
    started.get_future().wait();
    EXPECT_EQ(resource.cancel(), Outcome::Success);
    EXPECT_EQ(acquisition.lease.release(), Outcome::Success);
    source.reset();
    registration.reset();
    EXPECT_FALSE(lifetime.expired());
    EXPECT_EQ(resource.canDestroy(), Outcome::Pending);
    EXPECT_EQ(ledger.charged(Charge::Retiring), 64);
    unblock.set_value();
    worker.join();
    EXPECT_TRUE(lifetime.expired());
    EXPECT_EQ(resource.completeWorker(token), Outcome::Success);
    EXPECT_EQ(resource.uploaded(), Outcome::InvalidTransition);
    EXPECT_EQ(resource.destroyed(), Outcome::Success);
}

class SharedResourceFailure : public testing::TestWithParam<Operation> {};
TEST_P(SharedResourceFailure, PrimarySurvivesCleanupFailureAndRollback) {
    BudgetLedger ledger(64);
    ResourceLifecycle resource(ledger, key, 1);
    ASSERT_EQ(resource.reserve(64), Outcome::Success);
    if (GetParam() != Operation::Allocate)
        ASSERT_EQ(resource.allocated(), Outcome::Success);
    if (GetParam() == Operation::CreateSampler)
        ASSERT_EQ(resource.uploaded(), Outcome::Success);
    const auto error = GetParam() == Operation::Allocate ? Outcome::DeviceOutOfMemory : Outcome::RuntimeFailure;
    EXPECT_EQ(resource.fail({error, GetParam(), 2}), Outcome::Success);
    EXPECT_EQ(resource.destroyed({Outcome::RuntimeFailure, Operation::Destroy, 7}), Outcome::RuntimeFailure);
    EXPECT_EQ(resource.status().primary.outcome, error);
    EXPECT_EQ(resource.status().primary.operation, GetParam());
    EXPECT_EQ(resource.status().primary.rawError, 2);
    EXPECT_EQ(resource.status().cleanup.rawError, 7);
    EXPECT_EQ(ledger.total(), 64);
    EXPECT_EQ(resource.destroyed(), Outcome::Success);
    EXPECT_EQ(ledger.total(), 0);
}
INSTANTIATE_TEST_SUITE_P(Faults, SharedResourceFailure, testing::Values(
    Operation::Allocate, Operation::Upload, Operation::CreateSampler));

TEST(SharedAccounting, DisjointTransfersAndAdmissionAreTransactional) {
    BudgetLedger ledger(100);
    EXPECT_EQ(ledger.reserve(Charge::Pending, 101), Outcome::DemandTooLarge);
    EXPECT_EQ(ledger.total(), 0);
    EXPECT_EQ(ledger.reserve(Charge::Pending, 60), Outcome::Success);
    EXPECT_EQ(ledger.reserve(Charge::Temporary, 20), Outcome::Success);
    EXPECT_EQ(ledger.reserve(Charge::Overhead, 10), Outcome::Success);
    EXPECT_EQ(ledger.reserve(Charge::Pending, 11), Outcome::Deferred);
    EXPECT_EQ(ledger.transfer(Charge::Pending, Charge::Resident, 60), Outcome::Success);
    EXPECT_EQ(ledger.transfer(Charge::Pending, Charge::Resident, 1), Outcome::InvalidInput);
    EXPECT_EQ(ledger.total(), 90);
    EXPECT_EQ(ledger.transfer(Charge::Resident, Charge::Retiring, 60), Outcome::Success);
    EXPECT_EQ(ledger.charged(Charge::Retiring), 60);
    EXPECT_EQ(ledger.release(Charge::Retiring, 61), Outcome::InvalidInput);
    EXPECT_EQ(ledger.total(), 90);
    EXPECT_EQ(ledger.release(Charge::Retiring, 60), Outcome::Success);
    EXPECT_EQ(ledger.total(), 30);
    EXPECT_EQ(ledger.peak(), 90);
}

TEST(SharedAccounting, OverflowZeroAndInvalidChargesAreExplicit) {
    BudgetLedger ledger(UINT64_MAX);
    EXPECT_EQ(ledger.reserve(Charge::Pending, UINT64_MAX), Outcome::Success);
    EXPECT_EQ(ledger.reserve(Charge::Pending, 1), Outcome::Deferred);
    EXPECT_EQ(ledger.transfer(Charge::Pending, Charge::Resident, UINT64_MAX), Outcome::Success);
    EXPECT_EQ(ledger.total(), UINT64_MAX);
    EXPECT_EQ(ledger.release(Charge::Resident, UINT64_MAX), Outcome::Success);
    EXPECT_EQ(ledger.reserve(Charge::Pending, 0), Outcome::InvalidInput);
    EXPECT_EQ(ledger.reserve(Charge::Count, 1), Outcome::InvalidInput);
    EXPECT_EQ(ledger.release(Charge::Count, 1), Outcome::InvalidInput);
    EXPECT_EQ(ledger.transfer(Charge::Count, Charge::Resident, 1), Outcome::InvalidInput);
    EXPECT_THROW(ledger.charged(Charge::Count), std::invalid_argument);
    BudgetLedger zero(0);
    EXPECT_EQ(zero.reserve(Charge::Pending, 1), Outcome::DemandTooLarge);
}

TEST(SharedAbi, HandshakeRejectsMismatchesWithoutWritingOutput) {
    AbiInfo info;
    info.keyBytes = 999;
    EXPECT_EQ(hipDemandGetContractAbiV1(0, sizeof(info), &info), static_cast<uint32_t>(Outcome::AbiMismatch));
    EXPECT_EQ(hipDemandGetContractAbiV1(Version, sizeof(info) - 1, &info), static_cast<uint32_t>(Outcome::AbiMismatch));
    EXPECT_EQ(hipDemandGetContractAbiV1(Version, sizeof(info) + 1, &info), static_cast<uint32_t>(Outcome::AbiMismatch));
    EXPECT_EQ(info.keyBytes, 999);
    EXPECT_EQ(hipDemandGetContractAbiV1(Version, sizeof(info), nullptr), static_cast<uint32_t>(Outcome::InvalidInput));
    EXPECT_EQ(hipDemandGetContractAbiV1(Version, sizeof(info), &info), static_cast<uint32_t>(Outcome::Success));
    EXPECT_EQ(info.keyBytes, sizeof(GpuKey));
    EXPECT_EQ(info.contextBytes, sizeof(DeviceContext));
    EXPECT_EQ(info.descriptorBytes, sizeof(SamplerDesc));
    EXPECT_EQ(info.requestBytes, sizeof(RequestKey));
    EXPECT_EQ(info.publicationBytes, sizeof(PublishedTexture));
    EXPECT_EQ(info.productionIntegration, 0);
}

TEST(SharedAbi, DescriptorAndContextRejectBeforeTableAccess) {
    DeviceContext context{{Version + 1, sizeof(DeviceContext)}, 1, nullptr, 1, 0};
    EXPECT_EQ(resolve(context, key, 1, 0, {}).outcome, Outcome::AbiMismatch);
    context.abi = {Version, sizeof(DeviceContext) - 1};
    EXPECT_EQ(resolve(context, key, 1, 0, {}).outcome, Outcome::AbiMismatch);
    context.abi = {Version, sizeof(DeviceContext)};
    auto desc = SamplerDesc{};
    desc.abi.byteSize = sizeof(SamplerDesc) + 4;
    EXPECT_EQ(resolve(context, key, 1, 0, desc).outcome, Outcome::AbiMismatch);
    for (uint32_t ratio : {0u, 17u, UINT32_MAX}) {
        desc = {};
        desc.maxAnisotropy = ratio;
        EXPECT_EQ(validate(desc), Outcome::InvalidInput);
    }
    desc = {};
    desc.sRGB = 2;
    EXPECT_EQ(validate(desc), Outcome::InvalidInput);
    desc = {};
    desc.maxMipLevels = 33;
    EXPECT_EQ(validate(desc), Outcome::InvalidInput);
}
