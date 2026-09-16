// SPDX-License-Identifier: MIT
#include "TestUtils.h"
#include "ImageDataTestUtils.h"
#include "TextureSamplingHarness.h"
#include "WholeMipTestHarness.h"
#include <DemandLoading/WholeMipTexture.h>
#include <DemandLoading/Internal/HipCalls.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <limits>
#include <mutex>
#include <tuple>

namespace hip_demand { namespace test {
namespace {
namespace wm = whole_mip_v1;
namespace cv = contract_v1;
namespace cap = capability_v1;
namespace aniso = anisotropy_v1;
using cv::Outcome;
using internal::HipOperation;

struct RuntimeFaultScope {
    std::shared_ptr<internal::HipFaultState> state = std::make_shared<internal::HipFaultState>();
    RuntimeFaultScope() { internal::setHipFaultState(state); }
    ~RuntimeFaultScope() { internal::setHipFaultState(nullptr); }
    size_t calls(HipOperation operation, bool successful = true) const {
        const auto records = state->records();
        return static_cast<size_t>(std::count_if(records.begin(), records.end(), [&](const auto& record) {
            return record.operation == operation && (!successful || record.error == hipSuccess);
        }));
    }
};

wm::Options runtimeOptions() {
    wm::Options result;
    result.maxSamplers = 8;
    result.maxRequests = 16;
    return result;
}

TextureDesc runtimeDescriptor() {
    TextureDesc result;
    result.filterMode = hipFilterModePoint;
    result.mipmapFilterMode = hipFilterModePoint;
    return result;
}

uint64_t metadataBytes(const wm::Options& options) {
    return 2ULL * options.maxSamplers * sizeof(wm::Entry) +
           uint64_t{options.maxRequests} * sizeof(cv::RequestKey) + 2 * sizeof(uint32_t);
}

uint64_t suffixBytes(const TypedImageSource& source, uint32_t first, uint32_t levels = 0) {
    if (!levels) levels = source.info.numMipLevels;
    uint64_t bytes = 0;
    for (uint32_t level = first; level < levels; ++level)
        bytes += uint64_t{std::max(1u, source.info.width >> level)} *
                 std::max(1u, source.info.height >> level) *
                 (source.info.format == HIP_AD_FORMAT_UNSIGNED_INT8 ? 4 : 16);
    return bytes;
}

std::shared_ptr<TypedImageSource> runtimeSource(bool floating = false) {
    return std::make_shared<TypedImageSource>(makeAuthoredMipSource(floating));
}

class WholeMipRuntime : public HipTestFixture {
protected:
    wm::Status status(const wm::Texture& texture) {
        wm::Status result;
        EXPECT_EQ(texture.getStatus(result), Outcome::Success);
        return result;
    }

    std::vector<wm::Entry> readEntries(const wm::DeviceContext& context) {
        std::vector<wm::Entry> entries(context.numSamplers);
        EXPECT_EQ(hipMemcpy(entries.data(), context.entries, entries.size() * sizeof(wm::Entry),
                            hipMemcpyDeviceToHost), hipSuccess);
        return entries;
    }

    std::vector<wm::Entry> snapshot(wm::Texture& texture) {
        wm::DeviceContext context;
        EXPECT_EQ(texture.prepare(nullptr, context), Outcome::Success);
        if (!context.entries) return {};
        auto entries = readEntries(context);
        EXPECT_EQ(texture.processRequests(), Outcome::Success);
        return entries;
    }

    wm::Entry firstEntry(wm::Texture& texture) {
        const auto entries = snapshot(texture);
        EXPECT_FALSE(entries.empty());
        return entries.empty() ? wm::Entry{} : entries.front();
    }

    Outcome request(wm::Texture& texture, const std::vector<cv::RequestKey>& requests,
                    uint32_t overflow = 0, uint32_t countOverride = UINT32_MAX) {
        wm::DeviceContext context;
        const auto prepared = texture.prepare(nullptr, context);
        EXPECT_EQ(prepared, Outcome::Success);
        if (prepared != Outcome::Success) return prepared;
        if (!requests.empty())
            EXPECT_EQ(hipMemcpy(context.requests, requests.data(), requests.size() * sizeof(cv::RequestKey),
                                hipMemcpyHostToDevice), hipSuccess);
        const uint32_t count = countOverride == UINT32_MAX ? static_cast<uint32_t>(requests.size()) : countOverride;
        EXPECT_EQ(hipMemcpy(context.requestCount, &count, sizeof(count), hipMemcpyHostToDevice), hipSuccess);
        EXPECT_EQ(hipMemcpy(context.requestOverflow, &overflow, sizeof(overflow), hipMemcpyHostToDevice), hipSuccess);
        return texture.processRequests();
    }

    void expectPayload(const wm::Entry& entry, const TypedImageSource& source,
                       uint32_t first, uint32_t levels = 0) {
        if (!levels) levels = source.info.numMipLevels;
        ASSERT_NE(entry.texture.textureObject, 0u);
        EXPECT_EQ(entry.texture.residency, Outcome::Success);
        EXPECT_EQ(entry.texture.revision, 1u);
        const auto& layout = entry.texture.mips;
        EXPECT_EQ(layout.originalWidth, source.info.width);
        EXPECT_EQ(layout.originalHeight, source.info.height);
        EXPECT_EQ(layout.originalLevels, levels);
        EXPECT_EQ(layout.firstResidentMip, first);
        EXPECT_EQ(layout.resourceWidth, std::max(1u, source.info.width >> first));
        EXPECT_EQ(layout.resourceHeight, std::max(1u, source.info.height >> first));
        EXPECT_EQ(layout.resourceLevels, levels - first);
        hipResourceDesc resource{};
        const auto samplerHandle = reinterpret_cast<hipTextureObject_t>(entry.texture.textureObject);
        ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, samplerHandle), hipSuccess);
        const bool mipmapped = levels - first > 1;
        ASSERT_EQ(resource.resType, mipmapped ? hipResourceTypeMipmappedArray : hipResourceTypeArray);
        hipTextureDesc sampler{};
        ASSERT_EQ(hipGetTextureObjectTextureDesc(&sampler, samplerHandle), hipSuccess);
        EXPECT_EQ(sampler.normalizedCoords, static_cast<int>(entry.descriptor.normalizedCoords));
        EXPECT_EQ(sampler.maxAnisotropy, 0u);
        EXPECT_FLOAT_EQ(sampler.minMipmapLevelClamp, 0.f);
        EXPECT_FLOAT_EQ(sampler.maxMipmapLevelClamp, static_cast<float>(levels - first - 1));
        for (uint32_t original = first; original < levels; ++original) {
            hipArray_t array = nullptr;
            if (mipmapped) {
                ASSERT_EQ(hipGetMipmappedArrayLevel(&array, resource.res.mipmap.mipmap, original - first), hipSuccess);
            } else {
                array = resource.res.array.array;
            }
            hipChannelFormatDesc channel{};
            hipExtent extent{};
            unsigned int flags = 0;
            ASSERT_EQ(hipArrayGetInfo(&channel, &extent, &flags, array), hipSuccess);
            const uint32_t width = std::max(1u, source.info.width >> original);
            const uint32_t height = std::max(1u, source.info.height >> original);
            EXPECT_EQ(extent.width, width);
            EXPECT_EQ(extent.height, height);
            const bool floating = source.info.format == HIP_AD_FORMAT_FLOAT;
            EXPECT_EQ(channel.x, floating ? 32 : 8);
            EXPECT_EQ(channel.y, channel.x);
            EXPECT_EQ(channel.z, channel.x);
            EXPECT_EQ(channel.w, channel.x);
            EXPECT_EQ(channel.f, floating ? hipChannelFormatKindFloat : hipChannelFormatKindUnsigned);
            const size_t pitch = width * (floating ? 16u : 4u);
            std::vector<unsigned char> pixels(pitch * height);
            ASSERT_EQ(hipMemcpy2DFromArray(pixels.data(), pitch, array, 0, 0, pitch, height,
                                          hipMemcpyDeviceToHost), hipSuccess);
            ASSERT_LT(original, source.mipPixels.size());
            EXPECT_EQ(pixels, source.mipPixels[original]) << "Original mip " << original;
        }
    }

    void expectNoBacking(const wm::Status& state) {
        EXPECT_EQ(state.residentBytes, 0u);
        EXPECT_EQ(state.pendingBytes, 0u);
        EXPECT_EQ(state.retiringBytes, 0u);
        EXPECT_EQ(state.pinnedBytes, 0u);
        EXPECT_EQ(state.mips.firstResidentMip, UINT32_MAX);
        EXPECT_EQ(state.device.resource, cap::Resource::None);
        EXPECT_EQ(state.device.published, 0u);
    }
};

TEST_F(WholeMipRuntime, RegistrationIsLazyBoundedAndEqualKeysReuseEvenWhileResident) {
    RuntimeFaultScope faults;
    auto options = runtimeOptions();
    options.maxSamplers = 2;
    auto source = runtimeSource();
    const auto desc = runtimeDescriptor();
    wm::Texture texture(source, desc, options);
    const auto first = texture.addSampler(desc);
    ASSERT_TRUE(first.succeeded());
    EXPECT_EQ(first.key.slot, 0u);
    EXPECT_EQ(first.key.generation, 1u);
    EXPECT_NE(first.key.incarnation, 0u);
    EXPECT_TRUE(texture.addSampler(desc).key == first.key);
    auto variant = desc;
    variant.addressMode[0] = hipAddressModeClamp;
    const auto second = texture.addSampler(variant);
    ASSERT_TRUE(second.succeeded());
    EXPECT_EQ(second.key.slot, 1u);
    EXPECT_EQ(second.key.incarnation, first.key.incarnation);
    variant.addressMode[1] = hipAddressModeBorder;
    const auto exhausted = texture.addSampler(variant);
    EXPECT_EQ(exhausted.outcome, Outcome::CapacityExhausted);
    EXPECT_FALSE(cv::valid(exhausted.key));
    EXPECT_EQ(exhausted.key.slot, cv::InvalidSlot);
    EXPECT_EQ(source->reads, 0u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), 0u);
    EXPECT_EQ(status(texture).overheadBytes, metadataBytes(options));
    ASSERT_EQ(texture.resize(2), Outcome::Success);
    EXPECT_TRUE(texture.addSampler(desc).key == first.key);
    EXPECT_EQ(texture.addSampler(variant).outcome, Outcome::InvalidTransition);
    ASSERT_EQ(texture.unload(), Outcome::Success);
    EXPECT_TRUE(texture.addSampler(desc).key == first.key);
    EXPECT_EQ(texture.addSampler(variant).outcome, Outcome::CapacityExhausted);
    EXPECT_EQ(status(texture).numSamplers, 2u);
    ASSERT_EQ(texture.resize(2), Outcome::Success);
    EXPECT_TRUE(firstEntry(texture).texture.key == first.key);
}

TEST_F(WholeMipRuntime, EveryCompatibleDescriptorFieldAndPreviewPolicyHasIndependentIdentity) {
    const auto base = runtimeDescriptor();
    auto options = runtimeOptions();
    options.maxSamplers = 9;
    wm::Texture texture(runtimeSource(), base, options);
    const auto first = texture.addSampler(base);
    ASSERT_TRUE(first.succeeded());
    std::vector<TextureDesc> variants(7, base);
    variants[0].addressMode[0] = hipAddressModeClamp;
    variants[1].addressMode[1] = hipAddressModeBorder;
    variants[2].filterMode = hipFilterModeLinear;
    variants[3].mipmapFilterMode = hipFilterModeLinear;
    variants[4].evictionPriority = EvictionPriority::Low;
    variants[5].evictionPriority = EvictionPriority::High;
    variants[6].evictionPriority = EvictionPriority::KeepResident;
    for (uint32_t i = 0; i < variants.size(); ++i) {
        const auto result = texture.addSampler(variants[i]);
        ASSERT_TRUE(result.succeeded());
        EXPECT_EQ(result.key.slot, i + 1);
        EXPECT_TRUE(texture.addSampler(variants[i]).key == result.key);
    }
    const auto preview = texture.addSampler(base, cv::SamplingPolicy::AllowCoarsePreview);
    ASSERT_TRUE(preview.succeeded());
    EXPECT_EQ(preview.key.slot, 8u);
    EXPECT_TRUE(texture.addSampler(base, cv::SamplingPolicy::AllowCoarsePreview).key == preview.key);
    ASSERT_EQ(texture.resize(2), Outcome::Success);
    const auto entries = snapshot(texture);
    ASSERT_EQ(entries.size(), 9u);
    EXPECT_EQ(entries[0].descriptor.samplingPolicy, cv::SamplingPolicy::Strict);
    EXPECT_EQ(entries[8].descriptor.samplingPolicy, cv::SamplingPolicy::AllowCoarsePreview);
}

TEST_F(WholeMipRuntime, UnnormalizedCoordinatesAreDistinctForIntentionallySingleLevelStorage) {
    auto options = runtimeOptions();
    options.mipPolicy = cap::MipPolicy::Disabled;
    const auto base = runtimeDescriptor();
    auto unnormalized = base;
    unnormalized.normalizedCoords = false;
    wm::Texture texture(runtimeSource(), base, options);
    const auto a = texture.addSampler(base), b = texture.addSampler(unnormalized);
    ASSERT_TRUE(a.succeeded());
    ASSERT_TRUE(b.succeeded());
    EXPECT_FALSE(a.key == b.key);
    EXPECT_TRUE(texture.addSampler(unnormalized).key == b.key);
}

class WholeMipStorageMismatch : public WholeMipRuntime, public testing::WithParamInterface<int> {};
TEST_P(WholeMipStorageMismatch, IncompatibleStorageNeverAliasesExistingSampler) {
    const auto base = runtimeDescriptor();
    wm::Texture texture(runtimeSource(), base, runtimeOptions());
    const auto first = texture.addSampler(base);
    ASSERT_TRUE(first.succeeded());
    auto changed = base;
    switch (GetParam()) {
    case 0: changed.sRGB = true; break;
    case 1: changed.generateMipmaps = false; break;
    case 2: changed.maxMipLevel = 2; break;
    case 3: changed.normalizedCoords = false; break;
    }
    const auto rejected = texture.addSampler(changed);
    EXPECT_EQ(rejected.outcome, Outcome::Unsupported);
    EXPECT_FALSE(cv::valid(rejected.key));
    EXPECT_EQ(status(texture).numSamplers, 1u);
    EXPECT_TRUE(texture.addSampler(base).key == first.key);
}
INSTANTIATE_TEST_SUITE_P(StorageFields, WholeMipStorageMismatch, testing::Range(0, 4));

class WholeMipInvalidOptions : public WholeMipRuntime, public testing::WithParamInterface<int> {};
TEST_P(WholeMipInvalidOptions, RejectsBeforeAnyHipAllocationAndNeverRegistersTextureZero) {
    RuntimeFaultScope faults;
    auto options = runtimeOptions();
    auto desc = runtimeDescriptor();
    auto source = runtimeSource();
    Outcome expected = Outcome::InvalidInput;
    switch (GetParam()) {
    case 0: options.abi.version = 0; expected = Outcome::AbiMismatch; break;
    case 1: --options.abi.byteSize; expected = Outcome::AbiMismatch; break;
    case 2: options.maxSamplers = 0; break;
    case 3: options.maxSamplers = 4097; break;
    case 4: options.maxRequests = 0; break;
    case 5: options.maxRequests = 1048577; break;
    case 6: options.maxManagedBytes = 0; break;
    case 7: options.maxDecodedBytes = 0; break;
    case 8: options.maxPinnedBytes = 0; break;
    case 9: options.reserved = 1; break;
    case 10: source.reset(); break;
    case 11: desc.addressMode[0] = static_cast<hipTextureAddressMode>(99); break;
    case 12: desc.filterMode = static_cast<hipTextureFilterMode>(99); break;
    case 13: options.mipPolicy = cap::MipPolicy::AllowBaseLevelFallback; expected = Outcome::Unsupported; break;
    case 14: options.mipPolicy = cap::MipPolicy::LegacyCompatibility; expected = Outcome::Unsupported; break;
    case 15: options.mipPolicy = static_cast<cap::MipPolicy>(99); expected = Outcome::Unsupported; break;
    }
    wm::Texture texture(source, desc, options);
    EXPECT_EQ(status(texture).initialization, expected);
    const auto registration = texture.addSampler(desc);
    EXPECT_EQ(registration.outcome, expected);
    EXPECT_FALSE(registration.succeeded());
    EXPECT_EQ(registration.key.slot, cv::InvalidSlot);
    EXPECT_EQ(texture.resize(0), expected);
    wm::DeviceContext context;
    EXPECT_EQ(texture.prepare(nullptr, context), expected);
    EXPECT_EQ(context.entries, nullptr);
    EXPECT_EQ(faults.calls(HipOperation::DeviceAllocation, false), 0u);
}
INSTANTIATE_TEST_SUITE_P(InvalidConfiguration, WholeMipInvalidOptions, testing::Range(0, 16));

class WholeMipCapacityBoundary : public WholeMipRuntime, public testing::WithParamInterface<bool> {};
TEST_P(WholeMipCapacityBoundary, AcceptsInclusiveMetadataCapacityEndpoints) {
    auto options = runtimeOptions();
    options.maxSamplers = GetParam() ? 4096 : 1;
    options.maxRequests = GetParam() ? 1048576 : 1;
    options.maxManagedBytes = metadataBytes(options);
    wm::Texture texture(runtimeSource(), runtimeDescriptor(), options);
    EXPECT_EQ(status(texture).initialization, Outcome::Success);
    EXPECT_EQ(status(texture).overheadBytes, options.maxManagedBytes);
    EXPECT_EQ(status(texture).managedPeakBytes, options.maxManagedBytes);
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
}
INSTANTIATE_TEST_SUITE_P(InclusiveLimits, WholeMipCapacityBoundary, testing::Bool());

class WholeMipInvalidSampler : public WholeMipRuntime, public testing::WithParamInterface<int> {};
TEST_P(WholeMipInvalidSampler, FailedVariantDoesNotConsumeSlotOrAliasZero) {
    const auto base = runtimeDescriptor();
    wm::Texture texture(runtimeSource(), base, runtimeOptions());
    auto changed = base;
    auto sampling = cv::SamplingPolicy::Strict;
    switch (GetParam()) {
    case 0: changed.addressMode[0] = static_cast<hipTextureAddressMode>(99); break;
    case 1: changed.addressMode[1] = static_cast<hipTextureAddressMode>(99); break;
    case 2: changed.filterMode = static_cast<hipTextureFilterMode>(99); break;
    case 3: changed.mipmapFilterMode = static_cast<hipTextureFilterMode>(99); break;
    case 4: changed.evictionPriority = static_cast<EvictionPriority>(99); break;
    case 5: sampling = static_cast<cv::SamplingPolicy>(99); break;
    }
    const auto rejected = texture.addSampler(changed, sampling);
    EXPECT_EQ(rejected.outcome, Outcome::InvalidInput);
    EXPECT_FALSE(cv::valid(rejected.key));
    const auto accepted = texture.addSampler(base);
    ASSERT_TRUE(accepted.succeeded());
    EXPECT_EQ(accepted.key.slot, 0u);
}
INSTANTIATE_TEST_SUITE_P(InvalidSamplerFields, WholeMipInvalidSampler, testing::Range(0, 6));

class WholeMipAnisotropyRejection : public WholeMipRuntime,
    public testing::WithParamInterface<std::tuple<uint32_t, bool>> {};
TEST_P(WholeMipAnisotropyRejection, NonlegacyRatiosAreUnsupportedNotSilentLegacyAliases) {
    const auto base = runtimeDescriptor();
    wm::Texture texture(runtimeSource(), base, runtimeOptions());
    const auto legacy = texture.addSampler(base);
    ASSERT_TRUE(legacy.succeeded());
    aniso::Request request;
    request.maxAnisotropy = std::get<0>(GetParam());
    request.requirement = std::get<1>(GetParam()) ? aniso::Requirement::RequireQualified :
                                                   aniso::Requirement::AllowUnqualified;
    const auto result = texture.addSampler(base, cv::SamplingPolicy::Strict, request);
    EXPECT_EQ(result.outcome, Outcome::Unsupported);
    EXPECT_FALSE(cv::valid(result.key));
    EXPECT_EQ(status(texture).numSamplers, 1u);
    EXPECT_TRUE(texture.addSampler(base).key == legacy.key);
}
INSTANTIATE_TEST_SUITE_P(AllNonlegacyRatios, WholeMipAnisotropyRejection,
    testing::Combine(testing::Range(1u, 17u), testing::Bool()));

TEST_F(WholeMipRuntime, ParityAndMalformedAnisotropyRequestsNeverRegister) {
    const auto desc = runtimeDescriptor();
    wm::Texture texture(runtimeSource(), desc, runtimeOptions());
    EXPECT_EQ(texture.addSampler(desc, cv::SamplingPolicy::Strict, aniso::Request::parity()).outcome,
              Outcome::Unsupported);
    for (int field = 0; field < 7; ++field) {
        SCOPED_TRACE(field);
        aniso::Request request;
        Outcome expected = Outcome::InvalidInput;
        switch (field) {
        case 0: request.abi.version = 0; expected = Outcome::AbiMismatch; break;
        case 1: --request.abi.byteSize; expected = Outcome::AbiMismatch; break;
        case 2: request.maxAnisotropy = 0; break;
        case 3: request.maxAnisotropy = 17; break;
        case 4: request.maxAnisotropy = UINT32_MAX; break;
        case 5: request.profile = static_cast<aniso::Profile>(99); break;
        case 6: request.requirement = static_cast<aniso::Requirement>(99); break;
        }
        const auto result = texture.addSampler(desc, cv::SamplingPolicy::Strict, request);
        EXPECT_EQ(result.outcome, expected);
        EXPECT_FALSE(cv::valid(result.key));
    }
    EXPECT_EQ(status(texture).numSamplers, 0u);
    EXPECT_EQ(texture.addSampler(desc).key.slot, 0u);
}

TEST_F(WholeMipRuntime, StatusAndContextRejectBothAbiFieldsWithoutMutatingCallerRecords) {
    wm::Texture texture(runtimeSource(), runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    for (bool version : {false, true}) {
        wm::Status result;
        result.residentBytes = 123;
        if (version) ++result.abi.version;
        else --result.abi.byteSize;
        EXPECT_EQ(texture.getStatus(result), Outcome::AbiMismatch);
        EXPECT_EQ(result.residentBytes, 123u);
        wm::DeviceContext context;
        context.incarnation = 456;
        if (version) ++context.abi.version;
        else --context.abi.byteSize;
        EXPECT_EQ(texture.prepare(nullptr, context), Outcome::AbiMismatch);
        EXPECT_EQ(context.incarnation, 456u);
        EXPECT_EQ(context.entries, nullptr);
    }
    EXPECT_EQ(request(texture, {}), Outcome::Success);
}

TEST_F(WholeMipRuntime, EmptyAndRepeatedEpochsPerformNoHiddenLoading) {
    RuntimeFaultScope faults;
    auto source = runtimeSource();
    wm::Texture texture(source, runtimeDescriptor(), runtimeOptions());
    wm::DeviceContext context;
    EXPECT_EQ(texture.resize(0), Outcome::InvalidTransition);
    EXPECT_EQ(texture.prepare(nullptr, context), Outcome::InvalidTransition);
    EXPECT_EQ(texture.processRequests(), Outcome::InvalidTransition);
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    ASSERT_EQ(texture.prepare(nullptr, context), Outcome::Success);
    const auto initial = readEntries(context);
    ASSERT_EQ(initial.size(), 1u);
    EXPECT_EQ(initial[0].texture.residency, Outcome::Pending);
    EXPECT_EQ(initial[0].texture.textureObject, 0u);
    wm::DeviceContext duplicate;
    EXPECT_EQ(texture.prepare(nullptr, duplicate), Outcome::InvalidTransition);
    EXPECT_EQ(texture.processRequests(), Outcome::Success);
    EXPECT_EQ(texture.processRequests(), Outcome::InvalidTransition);
    EXPECT_EQ(request(texture, {}), Outcome::Success);
    EXPECT_EQ(source->reads, 0u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), 0u);
    expectNoBacking(status(texture));
}

TEST_F(WholeMipRuntime, RequestsGrowToMinimumOriginalMipIndependentOfOrderAndNeverTrim) {
    auto source = runtimeSource();
    const auto desc = runtimeDescriptor();
    auto variant = desc;
    variant.addressMode[0] = hipAddressModeClamp;
    wm::Texture texture(source, desc, runtimeOptions());
    const auto a = texture.addSampler(desc), b = texture.addSampler(variant);
    ASSERT_TRUE(a.succeeded());
    ASSERT_TRUE(b.succeeded());
    EXPECT_EQ(request(texture, {{a.key, 1, 3, 0}, {b.key, 1, 2, 0},
                               {a.key, 1, 2, 0}, {b.key, 1, 3, 0}}), Outcome::Success);
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{2, 3}));
    EXPECT_EQ(status(texture).requestCount, 4u);
    EXPECT_EQ(status(texture).desiredFirstMip, 2u);
    EXPECT_EQ(request(texture, {{b.key, 1, 3, 0}}), Outcome::Success);
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{2, 3}));
    EXPECT_EQ(status(texture).mips.firstResidentMip, 2u);
    EXPECT_EQ(request(texture, {{b.key, 1, 1, 0}, {a.key, 1, 0, 0}}), Outcome::Success);
    EXPECT_EQ(status(texture).mips.firstResidentMip, 0u);
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{2, 3, 0, 1, 2, 3}));
    EXPECT_EQ(request(texture, {}), Outcome::Success);
    EXPECT_EQ(status(texture).requestCount, 0u);
    EXPECT_EQ(status(texture).requestOverflow, 0u);
}

class WholeMipInvalidRequest : public WholeMipRuntime, public testing::WithParamInterface<int> {};
TEST_P(WholeMipInvalidRequest, ValidatesEntireKeyBeforeAnyReadAndRecoversOnExplicitNextEpoch) {
    auto source = runtimeSource();
    const auto desc = runtimeDescriptor();
    wm::Texture texture(source, desc, runtimeOptions());
    wm::Texture other(source, desc, runtimeOptions());
    const auto registered = texture.addSampler(desc), foreign = other.addSampler(desc);
    ASSERT_TRUE(registered.succeeded());
    ASSERT_TRUE(foreign.succeeded());
    EXPECT_EQ(registered.key.slot, foreign.key.slot);
    EXPECT_NE(registered.key.incarnation, foreign.key.incarnation);
    cv::RequestKey invalid{registered.key, 1, 2, 0};
    switch (GetParam()) {
    case 0: invalid.texture = {}; break;
    case 1: invalid.texture.slot = 1; break;
    case 2: invalid.texture.generation = 0; break;
    case 3: ++invalid.texture.generation; break;
    case 4: invalid.texture = foreign.key; break;
    case 5: invalid.revision = 0; break;
    case 6: ++invalid.revision; break;
    case 7: invalid.reserved = 1; break;
    case 8: invalid.originalMip = 4; break;
    case 9: invalid.originalMip = UINT32_MAX; break;
    }
    EXPECT_EQ(request(texture, {invalid}), Outcome::InvalidKey);
    EXPECT_EQ(status(texture).rejectedRequests, 1u);
    EXPECT_EQ(source->reads, 0u);
    expectNoBacking(status(texture));
    EXPECT_EQ(request(texture, {{registered.key, 1, 2, 0}}), Outcome::Success);
    EXPECT_EQ(status(texture).rejectedRequests, 0u);
    EXPECT_EQ(status(texture).mips.firstResidentMip, 2u);
}
INSTANTIATE_TEST_SUITE_P(CompleteIdentity, WholeMipInvalidRequest, testing::Range(0, 10));

TEST_F(WholeMipRuntime, ARejectedMixedRequestBatchDoesNotPartiallyLoadValidDemand) {
    auto source = runtimeSource();
    wm::Texture texture(source, runtimeDescriptor(), runtimeOptions());
    const auto registered = texture.addSampler(runtimeDescriptor());
    ASSERT_TRUE(registered.succeeded());
    EXPECT_EQ(request(texture, {{registered.key, 1, 2, 0}, {registered.key, 2, 0, 0}}), Outcome::InvalidKey);
    EXPECT_EQ(status(texture).rejectedRequests, 1u);
    EXPECT_EQ(status(texture).requestCount, 2u);
    EXPECT_EQ(source->reads, 0u);
    EXPECT_EQ(request(texture, {{registered.key, 1, 2, 0}}), Outcome::Success);
}

class WholeMipRequestOverflow : public WholeMipRuntime, public testing::WithParamInterface<bool> {};
TEST_P(WholeMipRequestOverflow, OverflowFlagAndOversizedCounterAreVisibleAndDoNotReadPastCapacity) {
    auto options = runtimeOptions();
    options.maxRequests = 1;
    auto source = runtimeSource();
    wm::Texture texture(source, runtimeDescriptor(), options);
    const auto registered = texture.addSampler(runtimeDescriptor());
    ASSERT_TRUE(registered.succeeded());
    EXPECT_EQ(request(texture, {{registered.key, 1, 2, 0}}, GetParam() ? 1 : 0, GetParam() ? 1 : 2),
              Outcome::RequestOverflow);
    const auto failed = status(texture);
    EXPECT_EQ(failed.requestCount, GetParam() ? 1u : 2u);
    EXPECT_EQ(failed.requestOverflow, GetParam() ? 1u : 0u);
    EXPECT_EQ(source->reads, 0u);
    EXPECT_EQ(request(texture, {{registered.key, 1, 2, 0}}), Outcome::Success);
    EXPECT_EQ(status(texture).requestOverflow, 0u);
}
INSTANTIATE_TEST_SUITE_P(OverflowSignals, WholeMipRequestOverflow, testing::Bool());

class WholeMipRuntimePayload : public WholeMipRuntime, public testing::WithParamInterface<bool> {};
TEST_P(WholeMipRuntimePayload, CoarseGrowthAndTrimUseExactAuthoredPixelsAndDisjointOverlapAccounting) {
    RuntimeFaultScope faults;
    const auto source = runtimeSource(GetParam());
    const auto options = runtimeOptions();
    wm::Texture texture(source, runtimeDescriptor(), options);
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    const uint64_t overhead = metadataBytes(options);
    uint64_t uploaded = 0, peak = overhead, previous = 0;
    std::vector<wm::Status> milestones;
    for (uint32_t first : {2u, 0u, 2u, 1u, 3u, 0u}) {
        SCOPED_TRACE(first);
        const uint64_t bytes = suffixBytes(*source, first);
        peak = std::max(peak, overhead + previous + bytes);
        ASSERT_EQ(texture.resize(first), Outcome::Success);
        uploaded += bytes;
        const auto state = status(texture);
        if (milestones.size() < 3) milestones.push_back(state);
        EXPECT_EQ(state.residentBytes, bytes);
        EXPECT_EQ(state.pendingBytes, 0u);
        EXPECT_EQ(state.retiringBytes, 0u);
        EXPECT_EQ(state.overheadBytes, overhead);
        EXPECT_EQ(state.managedPeakBytes, peak);
        EXPECT_EQ(state.uploadedBytes, uploaded);
        EXPECT_EQ(state.sourceBytes, uploaded);
        EXPECT_EQ(state.generatedLevels, 0u);
        EXPECT_EQ(state.pinnedBytes, 0u);
        EXPECT_EQ(state.device.payloadBytes, bytes);
        EXPECT_EQ(state.device.resourceLevels, source->info.numMipLevels - first);
        const auto entries = snapshot(texture);
        ASSERT_EQ(entries.size(), 1u);
        expectPayload(entries[0], *source, first);
        previous = bytes;
    }
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{2, 3, 0, 1, 2, 3, 2, 3,
                                                           1, 2, 3, 3, 0, 1, 2, 3}));
    EXPECT_EQ(status(texture).authoredReads, source->readLevels.size());
    EXPECT_EQ(status(texture).decodedPeakBytes, source->mipPixels[0].size());
    EXPECT_EQ(status(texture).pinnedPeakBytes, source->mipPixels[0].size());
    EXPECT_EQ(status(texture).replacements, 5u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), 5u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateArray), 1u);
    const auto reads = source->reads;
    const auto allocations = faults.calls(HipOperation::AllocateMipmapped);
    EXPECT_EQ(texture.resize(0), Outcome::Success);
    EXPECT_EQ(source->reads, reads);
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), allocations);
    ASSERT_EQ(texture.unload(), Outcome::Success);
    expectNoBacking(status(texture));
    EXPECT_EQ(status(texture).managedPeakBytes, peak);
    EXPECT_EQ(faults.calls(HipOperation::FreeMipmapped), 5u);
    EXPECT_EQ(faults.calls(HipOperation::FreeArray), 1u);
    if (!GetParam() && !HasFailure()) {
        std::cout << "WholeMip RGBA8 coarse->grow->trim";
        const auto printBytes = [&](const char* name, uint64_t wm::Status::*field) {
            std::cout << ' ' << name << "=[" << milestones[0].*field << ','
                      << milestones[1].*field << ',' << milestones[2].*field << ']';
        };
        printBytes("resident", &wm::Status::residentBytes);
        printBytes("pending", &wm::Status::pendingBytes);
        printBytes("retiring", &wm::Status::retiringBytes);
        printBytes("overhead", &wm::Status::overheadBytes);
        printBytes("managedPeak", &wm::Status::managedPeakBytes);
        printBytes("decodedPeak", &wm::Status::decodedPeakBytes);
        printBytes("pinnedPeak", &wm::Status::pinnedPeakBytes);
        printBytes("sourceBytes", &wm::Status::sourceBytes);
        printBytes("uploadBytes", &wm::Status::uploadedBytes);
        std::cout << " dimensions=[";
        for (size_t i = 0; i < milestones.size(); ++i) {
            const auto& mips = milestones[i].mips;
            if (i) std::cout << ',';
            std::cout << mips.resourceWidth << 'x' << mips.resourceHeight << '/'
                      << mips.resourceLevels << "levels";
        }
        std::cout << "]\n";
    }
}
INSTANTIATE_TEST_SUITE_P(AuthoredRepresentations, WholeMipRuntimePayload, testing::Bool());

TEST_F(WholeMipRuntime, SharedVariantsRebuildAllObjectsButChargeAndUploadOneBacking) {
    RuntimeFaultScope faults;
    const auto source = runtimeSource();
    const auto base = runtimeDescriptor();
    wm::Texture texture(source, base, runtimeOptions());
    auto clamp = base, linear = base;
    clamp.addressMode[0] = hipAddressModeClamp;
    clamp.addressMode[1] = hipAddressModeBorder;
    linear.filterMode = hipFilterModeLinear;
    linear.mipmapFilterMode = hipFilterModeLinear;
    const std::vector<TextureDesc> descriptors{base, clamp, linear};
    for (const auto& desc : descriptors) ASSERT_TRUE(texture.addSampler(desc).succeeded());
    for (uint32_t first : {2u, 0u, 1u}) {
        ASSERT_EQ(texture.resize(first), Outcome::Success);
        const auto entries = snapshot(texture);
        ASSERT_EQ(entries.size(), descriptors.size());
        hipMipmappedArray_t shared = nullptr;
        for (size_t id = 0; id < entries.size(); ++id) {
            hipResourceDesc resource{};
            hipTextureDesc sampler{};
            const auto samplerHandle = reinterpret_cast<hipTextureObject_t>(entries[id].texture.textureObject);
            ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, samplerHandle), hipSuccess);
            ASSERT_EQ(resource.resType, hipResourceTypeMipmappedArray);
            if (!id) shared = resource.res.mipmap.mipmap;
            EXPECT_EQ(resource.res.mipmap.mipmap, shared);
            ASSERT_EQ(hipGetTextureObjectTextureDesc(&sampler, samplerHandle), hipSuccess);
            EXPECT_EQ(sampler.addressMode[0], descriptors[id].addressMode[0]);
            EXPECT_EQ(sampler.addressMode[1], descriptors[id].addressMode[1]);
            EXPECT_EQ(sampler.filterMode, descriptors[id].filterMode);
            EXPECT_EQ(sampler.mipmapFilterMode, descriptors[id].mipmapFilterMode);
            EXPECT_EQ(sampler.normalizedCoords, 1);
            EXPECT_EQ(sampler.maxAnisotropy, 0u);
            EXPECT_FLOAT_EQ(sampler.maxMipmapLevelClamp, static_cast<float>(3 - first));
            for (size_t earlier = 0; earlier < id; ++earlier)
                EXPECT_NE(entries[id].texture.textureObject, entries[earlier].texture.textureObject);
            EXPECT_TRUE(texture.addSampler(descriptors[id]).key == entries[id].texture.key);
        }
        EXPECT_EQ(status(texture).residentBytes, suffixBytes(*source, first));
    }
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), 3u);
    EXPECT_EQ(faults.calls(HipOperation::CreateSampler), 9u);
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{2, 3, 0, 1, 2, 3, 1, 2, 3}));
}

class WholeMipSmallTexture : public WholeMipRuntime, public testing::WithParamInterface<int> {};
TEST_P(WholeMipSmallTexture, FullChainSingletonDisabledAndLimitedPathsHaveExplicitResourceFacts) {
    auto options = runtimeOptions();
    auto desc = runtimeDescriptor();
    auto source = runtimeSource();
    uint32_t levels = 4;
    cap::Reason reason = cap::Reason::None;
    switch (GetParam()) {
    case 0: break;
    case 1:
        source = std::make_shared<TypedImageSource>(makeFilteringMipSource(1, 1));
        levels = 1; reason = cap::Reason::Singleton; break;
    case 2:
        options.mipPolicy = cap::MipPolicy::Disabled;
        levels = 1; reason = cap::Reason::Disabled; break;
    case 3:
        desc.maxMipLevel = 1;
        levels = 1; reason = cap::Reason::LevelLimit; break;
    case 4:
        desc.maxMipLevel = 3; levels = 3; break;
    }
    wm::Texture texture(source, desc, options);
    ASSERT_TRUE(texture.addSampler(desc).succeeded());
    ASSERT_EQ(texture.resize(0), Outcome::Success);
    const auto state = status(texture);
    EXPECT_EQ(state.device.policy, options.mipPolicy);
    EXPECT_EQ(state.device.reason, reason);
    EXPECT_EQ(state.device.originalLevels, levels);
    EXPECT_EQ(state.device.resourceLevels, levels);
    EXPECT_EQ(state.device.resource, levels == 1 ? cap::Resource::Array : cap::Resource::MipmappedArray);
    EXPECT_EQ(state.residentBytes, suffixBytes(*source, 0, levels));
    EXPECT_EQ(state.authoredReads, levels);
    expectPayload(firstEntry(texture), *source, 0, levels);
    EXPECT_EQ(texture.resize(levels), Outcome::InvalidInput);
    EXPECT_EQ(texture.resize(UINT32_MAX), Outcome::InvalidInput);
    EXPECT_EQ(status(texture).mips.firstResidentMip, 0u);
}
INSTANTIATE_TEST_SUITE_P(DenseSmallPaths, WholeMipSmallTexture, testing::Range(0, 5));

TEST_F(WholeMipRuntime, ExactManagedAdmissionIncludesMetadataAndRejectsOneByteOversizeTerminally) {
    for (bool exact : {false, true}) {
        SCOPED_TRACE(exact);
        RuntimeFaultScope faults;
        auto options = runtimeOptions();
        auto source = runtimeSource();
        const uint64_t payload = suffixBytes(*source, 2);
        options.maxManagedBytes = metadataBytes(options) + payload - (exact ? 0 : 1);
        wm::Texture texture(source, runtimeDescriptor(), options);
        ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
        EXPECT_EQ(texture.resize(2), exact ? Outcome::Success : Outcome::DemandTooLarge);
        EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), exact ? 1u : 0u);
        EXPECT_EQ(source->reads, exact ? 2u : 0u);
        EXPECT_LE(status(texture).managedPeakBytes, options.maxManagedBytes);
        if (!exact) {
            const auto failed = status(texture);
            EXPECT_EQ(failed.primary.outcome, Outcome::DemandTooLarge);
            expectNoBacking(failed);
            const auto entries = snapshot(texture);
            ASSERT_EQ(entries.size(), 1u);
            EXPECT_EQ(entries[0].texture.residency, Outcome::DemandTooLarge);
            EXPECT_EQ(entries[0].texture.textureObject, 0u);
        }
    }
}

TEST_F(WholeMipRuntime, MetadataReservationItselfIsBoundedBeforeAllocatingAnyDeviceBuffers) {
    RuntimeFaultScope faults;
    auto options = runtimeOptions();
    options.maxManagedBytes = metadataBytes(options) - 1;
    wm::Texture texture(runtimeSource(), runtimeDescriptor(), options);
    EXPECT_EQ(status(texture).initialization, Outcome::DemandTooLarge);
    EXPECT_EQ(faults.calls(HipOperation::DeviceAllocation, false), 0u);
    EXPECT_EQ(status(texture).overheadBytes, 0u);
}

TEST_F(WholeMipRuntime, OverlapDeferralDoesNoIoOrAllocationAndOnlyExplicitUnloadAllowsRetry) {
    RuntimeFaultScope faults;
    auto options = runtimeOptions();
    const auto source = runtimeSource();
    options.maxManagedBytes = metadataBytes(options) + suffixBytes(*source, 0);
    wm::Texture texture(source, runtimeDescriptor(), options);
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    ASSERT_EQ(texture.resize(2), Outcome::Success);
    const auto previous = firstEntry(texture).texture.textureObject;
    const auto reads = source->reads;
    for (int attempt = 0; attempt < 3; ++attempt) {
        EXPECT_EQ(texture.resize(0), Outcome::Deferred);
        const auto state = status(texture);
        EXPECT_EQ(state.primary.outcome, Outcome::Deferred);
        EXPECT_EQ(state.desiredFirstMip, 0u);
        EXPECT_EQ(state.mips.firstResidentMip, 2u);
        EXPECT_EQ(state.residentBytes, suffixBytes(*source, 2));
        EXPECT_EQ(state.pendingBytes, 0u);
        EXPECT_EQ(state.retiringBytes, 0u);
        EXPECT_EQ(firstEntry(texture).texture.textureObject, previous);
    }
    EXPECT_EQ(source->reads, reads);
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), 1u);
    ASSERT_EQ(texture.unload(), Outcome::Success);
    ASSERT_EQ(texture.resize(0), Outcome::Success);
    EXPECT_EQ(status(texture).managedPeakBytes, options.maxManagedBytes);
    expectPayload(firstEntry(texture), *source, 0);
}

class WholeMipHostLimit : public WholeMipRuntime,
    public testing::WithParamInterface<std::tuple<bool, bool>> {};
TEST_P(WholeMipHostLimit, DecodedAndInUsePinnedLimitsAreInclusiveAndRejectBeforeSourceReads) {
    const auto [pinned, exact] = GetParam();
    RuntimeFaultScope faults;
    auto options = runtimeOptions();
    const auto source = runtimeSource();
    const uint64_t largestLevel = source->mipPixels[2].size();
    if (pinned) options.maxPinnedBytes = largestLevel - (exact ? 0 : 1);
    else options.maxDecodedBytes = largestLevel - (exact ? 0 : 1);
    wm::Texture texture(source, runtimeDescriptor(), options);
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    EXPECT_EQ(texture.resize(2), exact ? Outcome::Success : Outcome::DemandTooLarge);
    const auto state = status(texture);
    EXPECT_LE(state.pinnedPeakBytes, options.maxPinnedBytes);
    EXPECT_LE(state.decodedPeakBytes, options.maxDecodedBytes);
    EXPECT_EQ(state.pinnedBytes, 0u);
    if (exact) {
        EXPECT_EQ(state.decodedPeakBytes, largestLevel);
        EXPECT_EQ(state.pinnedPeakBytes, largestLevel);
        EXPECT_EQ(state.authoredReads, 2u);
        expectPayload(firstEntry(texture), *source, 2);
    } else {
        EXPECT_EQ(source->reads, 0u);
        EXPECT_EQ(state.decodedPeakBytes, 0u);
        expectNoBacking(state);
        if (pinned) EXPECT_EQ(faults.calls(HipOperation::HostAllocation, false), 0u);
    }
}
INSTANTIATE_TEST_SUITE_P(HostLimitEdges, WholeMipHostLimit, testing::Combine(testing::Bool(), testing::Bool()));

struct RuntimeFailureCase {
    HipOperation hipOperation;
    cap::Operation reportedOperation;
    size_t invocation;
    uint32_t first;
};

const std::vector<RuntimeFailureCase> runtimeFailureCases{
    {HipOperation::AllocateMipmapped, cap::Operation::AllocateMipmapped, 1, 0},
    {HipOperation::AllocateArray, cap::Operation::AllocateArray, 1, 3},
    {HipOperation::HostAllocation, cap::Operation::AllocateHost, 1, 0},
    {HipOperation::GetLevel, cap::Operation::GetLevel, 1, 0},
    {HipOperation::GetLevel, cap::Operation::GetLevel, 2, 0},
    {HipOperation::GetLevel, cap::Operation::GetLevel, 3, 0},
    {HipOperation::GetLevel, cap::Operation::GetLevel, 4, 0},
    {HipOperation::Upload, cap::Operation::Upload, 1, 0},
    {HipOperation::Upload, cap::Operation::Upload, 2, 0},
    {HipOperation::Upload, cap::Operation::Upload, 3, 0},
    {HipOperation::Upload, cap::Operation::Upload, 4, 0},
    {HipOperation::Upload, cap::Operation::Upload, 1, 3},
    {HipOperation::CreateSampler, cap::Operation::CreateSampler, 1, 0},
    {HipOperation::CreateSampler, cap::Operation::CreateSampler, 2, 0},
    {HipOperation::ReadSampler, cap::Operation::ReadSampler, 1, 0},
    {HipOperation::ReadSampler, cap::Operation::ReadSampler, 2, 0},
    {HipOperation::PublishMappings, cap::Operation::Publish, 1, 0},
    {HipOperation::SynchronizeConsumers, cap::Operation::Publish, 1, 0}
};

class WholeMipRollback : public WholeMipRuntime,
    public testing::WithParamInterface<std::tuple<RuntimeFailureCase, bool>> {};
TEST_P(WholeMipRollback, EveryConstructionFailurePreservesOldResourceAndExplicitRetryReplacesIt) {
    const auto [failure, initiallyResident] = GetParam();
    RuntimeFaultScope faults;
    const auto source = runtimeSource();
    const auto desc = runtimeDescriptor();
    auto variant = desc;
    variant.addressMode[0] = hipAddressModeClamp;
    wm::Texture texture(source, desc, runtimeOptions());
    ASSERT_TRUE(texture.addSampler(desc).succeeded());
    ASSERT_TRUE(texture.addSampler(variant).succeeded());
    std::vector<wm::Entry> old;
    if (initiallyResident) {
        ASSERT_EQ(texture.resize(2), Outcome::Success);
        old = snapshot(texture);
        ASSERT_EQ(old.size(), 2u);
    }
    faults.state->fail(failure.hipOperation, hipErrorUnknown, failure.invocation);
    ASSERT_EQ(texture.resize(failure.first), Outcome::RuntimeFailure);
    const auto state = status(texture);
    EXPECT_EQ(state.operation, Outcome::RuntimeFailure);
    EXPECT_EQ(state.primary.operation, failure.reportedOperation);
    EXPECT_EQ(state.primary.rawHipError, static_cast<int32_t>(hipErrorUnknown));
    EXPECT_EQ(state.primary.outcome, Outcome::RuntimeFailure);
    EXPECT_EQ(state.cleanup.outcome, Outcome::Success);
    EXPECT_EQ(state.pendingBytes, 0u);
    EXPECT_EQ(state.retiringBytes, 0u);
    EXPECT_EQ(state.pinnedBytes, 0u);
    if (initiallyResident) {
        EXPECT_EQ(state.residentBytes, suffixBytes(*source, 2));
        EXPECT_FLOAT_EQ(state.device.submittedSampler.maxMipmapLevelClamp, 1.f);
        EXPECT_FLOAT_EQ(state.device.returnedSampler.maxMipmapLevelClamp, 1.f);
        EXPECT_EQ(state.device.submittedSampler.mipmapFilterMode, desc.mipmapFilterMode);
        EXPECT_EQ(state.device.returnedSampler.mipmapFilterMode, desc.mipmapFilterMode);
        const auto unchanged = snapshot(texture);
        ASSERT_EQ(unchanged.size(), 2u);
        for (size_t id = 0; id < old.size(); ++id) {
            EXPECT_EQ(unchanged[id].texture.textureObject, old[id].texture.textureObject);
            expectPayload(unchanged[id], *source, 2);
        }
    } else {
        expectNoBacking(state);
        const auto failed = snapshot(texture);
        ASSERT_EQ(failed.size(), 2u);
        for (const auto& entry : failed) {
            EXPECT_EQ(entry.texture.textureObject, 0u);
            EXPECT_EQ(entry.texture.residency, Outcome::RuntimeFailure);
        }
    }
    ASSERT_EQ(texture.resize(failure.first), Outcome::Success);
    const auto retried = snapshot(texture);
    ASSERT_EQ(retried.size(), 2u);
    expectPayload(retried[0], *source, failure.first);
    EXPECT_EQ(status(texture).primary.outcome, Outcome::Success);
}
INSTANTIATE_TEST_SUITE_P(EveryRuntimeStage, WholeMipRollback,
    testing::Combine(testing::ValuesIn(runtimeFailureCases), testing::Bool()));

class WholeMipRawError : public WholeMipRuntime, public testing::WithParamInterface<hipError_t> {};
TEST_P(WholeMipRawError, AllocationFailureClassificationIsExactAndDoesNotPoisonFutureCapability) {
    RuntimeFaultScope faults;
    wm::Texture texture(runtimeSource(), runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    faults.state->fail(HipOperation::AllocateMipmapped, GetParam());
    const auto expected = GetParam() == hipErrorOutOfMemory ? Outcome::DeviceOutOfMemory :
                          GetParam() == hipErrorNotSupported ? Outcome::Unsupported : Outcome::InvalidInput;
    EXPECT_EQ(texture.resize(2), expected);
    const auto state = status(texture);
    EXPECT_EQ(state.primary.outcome, expected);
    EXPECT_EQ(state.primary.operation, cap::Operation::AllocateMipmapped);
    EXPECT_EQ(state.primary.rawHipError, static_cast<int32_t>(GetParam()));
    EXPECT_EQ(state.device.resource, cap::Resource::None);
    EXPECT_EQ(faults.calls(HipOperation::AllocateArray, false), 0u) << "Required never falls back";
    ASSERT_EQ(texture.resize(2), Outcome::Success);
    EXPECT_EQ(status(texture).device.capability, cap::Support::OperationSupported);
}
INSTANTIATE_TEST_SUITE_P(RawHipErrors, WholeMipRawError,
    testing::Values(hipErrorOutOfMemory, hipErrorNotSupported, hipErrorInvalidValue));

TEST_F(WholeMipRuntime, PinnedAllocationOomIsHostNotDevicePressure) {
    RuntimeFaultScope faults;
    wm::Texture texture(runtimeSource(), runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    faults.state->fail(HipOperation::HostAllocation, hipErrorOutOfMemory);
    EXPECT_EQ(texture.resize(2), Outcome::HostOutOfMemory);
    const auto state = status(texture);
    EXPECT_EQ(state.primary.outcome, Outcome::HostOutOfMemory);
    EXPECT_EQ(state.primary.operation, cap::Operation::AllocateHost);
    EXPECT_EQ(state.primary.rawHipError, static_cast<int32_t>(hipErrorOutOfMemory));
    expectNoBacking(state);
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped, false), 0u);
    EXPECT_EQ(texture.resize(2), Outcome::Success);
}

class WholeMipRetirementFailure : public WholeMipRuntime,
    public testing::WithParamInterface<std::tuple<bool, bool>> {};
TEST_P(WholeMipRetirementFailure, RetiringPayloadRemainsChargedUntilEachCleanupActuallySucceeds) {
    const auto [ordinaryArray, destroySampler] = GetParam();
    RuntimeFaultScope faults;
    auto source = runtimeSource();
    wm::Texture texture(source, runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    const uint32_t first = ordinaryArray ? 3 : 2;
    ASSERT_EQ(texture.resize(first), Outcome::Success);
    const auto cleanup = destroySampler ? HipOperation::DestroySampler :
                         ordinaryArray ? HipOperation::FreeArray : HipOperation::FreeMipmapped;
    const auto reported = destroySampler ? cap::Operation::DestroySampler :
                          ordinaryArray ? cap::Operation::FreeArray : cap::Operation::FreeMipmapped;
    faults.state->fail(cleanup, hipErrorUnknown);
    EXPECT_EQ(texture.unload(), Outcome::RuntimeFailure);
    auto state = status(texture);
    EXPECT_EQ(state.residentBytes, 0u);
    EXPECT_EQ(state.pendingBytes, 0u);
    EXPECT_EQ(state.retiringBytes, suffixBytes(*source, first));
    EXPECT_EQ(state.primary.outcome, Outcome::Success);
    EXPECT_EQ(state.cleanup.operation, reported);
    EXPECT_EQ(state.cleanup.rawHipError, static_cast<int32_t>(hipErrorUnknown));
    EXPECT_EQ(state.device.resource, cap::Resource::None);
    EXPECT_EQ(faults.calls(cleanup), 0u);
    faults.state->fail(cleanup, hipErrorUnknown);
    EXPECT_EQ(texture.collectRetired(), Outcome::RuntimeFailure);
    EXPECT_EQ(status(texture).retiringBytes, suffixBytes(*source, first));
    EXPECT_EQ(faults.calls(cleanup), 0u);
    EXPECT_EQ(texture.collectRetired(), Outcome::Success);
    expectNoBacking(status(texture));
    EXPECT_EQ(faults.calls(cleanup), 1u);
    EXPECT_EQ(texture.resize(first), Outcome::Success);
}
INSTANTIATE_TEST_SUITE_P(RetainedCleanup, WholeMipRetirementFailure,
    testing::Combine(testing::Bool(), testing::Bool()));

TEST_F(WholeMipRuntime, NewBackingCanPublishWhileOldFailedRetirementRemainsFullyCharged) {
    RuntimeFaultScope faults;
    auto source = runtimeSource();
    const auto options = runtimeOptions();
    wm::Texture texture(source, runtimeDescriptor(), options);
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    ASSERT_EQ(texture.resize(0), Outcome::Success);
    faults.state->fail(HipOperation::FreeMipmapped, hipErrorUnknown);
    EXPECT_EQ(texture.resize(2), Outcome::RuntimeFailure);
    const auto state = status(texture);
    EXPECT_EQ(state.residentBytes, suffixBytes(*source, 2));
    EXPECT_EQ(state.retiringBytes, suffixBytes(*source, 0));
    EXPECT_EQ(state.pendingBytes, 0u);
    EXPECT_EQ(state.managedPeakBytes, metadataBytes(options) + suffixBytes(*source, 0) + suffixBytes(*source, 2));
    EXPECT_EQ(state.primary.outcome, Outcome::Success);
    EXPECT_EQ(state.cleanup.operation, cap::Operation::FreeMipmapped);
    expectPayload(firstEntry(texture), *source, 2);
    EXPECT_EQ(texture.collectRetired(), Outcome::Success);
    EXPECT_EQ(status(texture).retiringBytes, 0u);
}

TEST_F(WholeMipRuntime, PrimaryUploadFailureAndCleanupFailureRemainIndependentAcrossCollection) {
    RuntimeFaultScope faults;
    auto source = runtimeSource();
    wm::Texture texture(source, runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    ASSERT_EQ(texture.resize(2), Outcome::Success);
    faults.state->fail(HipOperation::Upload, hipErrorInvalidValue, 2);
    faults.state->fail(HipOperation::FreeMipmapped, hipErrorUnknown);
    EXPECT_EQ(texture.resize(0), Outcome::InvalidInput);
    const auto failed = status(texture);
    EXPECT_EQ(failed.primary.outcome, Outcome::InvalidInput);
    EXPECT_EQ(failed.primary.operation, cap::Operation::Upload);
    EXPECT_EQ(failed.primary.rawHipError, static_cast<int32_t>(hipErrorInvalidValue));
    EXPECT_EQ(failed.cleanup.outcome, Outcome::RuntimeFailure);
    EXPECT_EQ(failed.cleanup.operation, cap::Operation::FreeMipmapped);
    EXPECT_EQ(failed.cleanup.rawHipError, static_cast<int32_t>(hipErrorUnknown));
    EXPECT_EQ(failed.residentBytes, suffixBytes(*source, 2));
    EXPECT_EQ(failed.retiringBytes, suffixBytes(*source, 0));
    EXPECT_EQ(texture.collectRetired(), Outcome::Success);
    const auto collected = status(texture);
    EXPECT_EQ(collected.retiringBytes, 0u);
    EXPECT_EQ(collected.primary.rawHipError, failed.primary.rawHipError);
    EXPECT_EQ(collected.cleanup.rawHipError, failed.cleanup.rawHipError);
    expectPayload(firstEntry(texture), *source, 2);
    EXPECT_EQ(texture.resize(0), Outcome::Success);
}

TEST_F(WholeMipRuntime, InitialPrimaryFailureSurvivesFailureToPublishTerminalMetadata) {
    RuntimeFaultScope faults;
    wm::Texture texture(runtimeSource(), runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    faults.state->fail(HipOperation::AllocateMipmapped, hipErrorOutOfMemory);
    faults.state->fail(HipOperation::PublishMappings, hipErrorUnknown);
    EXPECT_EQ(texture.resize(2), Outcome::DeviceOutOfMemory);
    const auto failed = status(texture);
    EXPECT_EQ(failed.primary.operation, cap::Operation::AllocateMipmapped);
    EXPECT_EQ(failed.primary.rawHipError, static_cast<int32_t>(hipErrorOutOfMemory));
    EXPECT_EQ(failed.cleanup.operation, cap::Operation::Publish);
    EXPECT_EQ(failed.cleanup.rawHipError, static_cast<int32_t>(hipErrorUnknown));
    EXPECT_EQ(firstEntry(texture).texture.residency, Outcome::DeviceOutOfMemory);
    EXPECT_EQ(texture.resize(2), Outcome::Success);
}

TEST_F(WholeMipRuntime, FailedPinnedFreeKeepsInUseBytesChargedUntilExplicitCollectionSucceeds) {
    RuntimeFaultScope faults;
    auto options = runtimeOptions();
    options.maxPinnedBytes = 16;
    wm::Texture texture(runtimeSource(), runtimeDescriptor(), options);
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    faults.state->fail(HipOperation::FreeHost, hipErrorUnknown);
    faults.state->fail(HipOperation::FreeHost, hipErrorUnknown, 2);
    EXPECT_EQ(texture.resize(2), Outcome::RuntimeFailure);
    const auto failed = status(texture);
    EXPECT_EQ(failed.residentBytes, 0u);
    EXPECT_EQ(failed.pinnedBytes, 16u);
    EXPECT_EQ(failed.pinnedPeakBytes, 16u);
    EXPECT_EQ(failed.cleanup.operation, cap::Operation::FreeHost);
    EXPECT_EQ(failed.cleanup.rawHipError, static_cast<int32_t>(hipErrorUnknown));
    const auto allocated = faults.calls(HipOperation::HostAllocation);
    faults.state->fail(HipOperation::FreeHost, hipErrorUnknown);
    EXPECT_EQ(texture.resize(2), Outcome::RuntimeFailure);
    EXPECT_EQ(faults.calls(HipOperation::HostAllocation), allocated);
    EXPECT_EQ(status(texture).pinnedBytes, 16u);
    EXPECT_EQ(texture.collectRetired(), Outcome::Success);
    EXPECT_EQ(status(texture).pinnedBytes, 0u);
    EXPECT_EQ(texture.resize(2), Outcome::Success);
    EXPECT_EQ(status(texture).pinnedPeakBytes, 16u);
}

TEST_F(WholeMipRuntime, CleanupSynchronizationFailureCannotDestroyUnfencedBackingOrPinnedStorage) {
    RuntimeFaultScope faults;
    auto source = runtimeSource();
    wm::Texture texture(source, runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    faults.state->fail(HipOperation::Upload, hipErrorInvalidValue);
    faults.state->fail(HipOperation::SynchronizeConsumers, hipErrorUnknown, 2);
    EXPECT_EQ(texture.resize(2), Outcome::InvalidInput);
    const auto state = status(texture);
    EXPECT_EQ(state.primary.operation, cap::Operation::Upload);
    EXPECT_EQ(state.cleanup.operation, cap::Operation::Publish);
    EXPECT_EQ(state.retiringBytes, suffixBytes(*source, 2));
    EXPECT_EQ(state.pinnedBytes, source->mipPixels[2].size());
    EXPECT_EQ(faults.calls(HipOperation::FreeHost), 0u);
    EXPECT_EQ(faults.calls(HipOperation::FreeMipmapped), 0u);
    EXPECT_EQ(texture.collectRetired(), Outcome::Success);
    expectNoBacking(status(texture));
}

class WholeMipInitializationFailure : public WholeMipRuntime, public testing::WithParamInterface<size_t> {};
TEST_P(WholeMipInitializationFailure, EachMetadataAllocationFailureOwnsAndCleansEarlierBuffers) {
    RuntimeFaultScope faults;
    faults.state->fail(HipOperation::DeviceAllocation, hipErrorOutOfMemory, GetParam());
    {
        wm::Texture texture(runtimeSource(), runtimeDescriptor(), runtimeOptions());
        EXPECT_EQ(status(texture).initialization, Outcome::DeviceOutOfMemory);
        EXPECT_FALSE(texture.addSampler(runtimeDescriptor()).succeeded());
        wm::DeviceContext context;
        EXPECT_EQ(texture.prepare(nullptr, context), Outcome::DeviceOutOfMemory);
        EXPECT_EQ(context.entries, nullptr);
    }
    EXPECT_EQ(faults.calls(HipOperation::DeviceAllocation), GetParam() - 1);
    EXPECT_EQ(faults.calls(HipOperation::FreeDevice), GetParam() - 1);
}
INSTANTIATE_TEST_SUITE_P(MetadataAllocations, WholeMipInitializationFailure,
    testing::Values(size_t{1}, size_t{2}, size_t{3}, size_t{4}));

class WholeMipDestructorCleanup : public WholeMipRuntime, public testing::WithParamInterface<HipOperation> {};
TEST_P(WholeMipDestructorCleanup, OneShotCleanupFailureRetriesWithoutLeakingOwnedResources) {
    RuntimeFaultScope faults;
    const bool array = GetParam() == HipOperation::FreeArray;
    {
        wm::Texture texture(runtimeSource(), runtimeDescriptor(), runtimeOptions());
        ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
        ASSERT_EQ(texture.resize(array ? 3 : 2), Outcome::Success);
        faults.state->fail(GetParam(), hipErrorUnknown);
    }
    EXPECT_EQ(faults.calls(HipOperation::DestroySampler), 1u);
    EXPECT_EQ(faults.calls(array ? HipOperation::FreeArray : HipOperation::FreeMipmapped), 1u);
    EXPECT_EQ(faults.calls(HipOperation::FreeDevice), 4u);
}
INSTANTIATE_TEST_SUITE_P(DestructorOwnership, WholeMipDestructorCleanup,
    testing::Values(HipOperation::DestroySampler, HipOperation::FreeArray,
                    HipOperation::FreeMipmapped, HipOperation::FreeDevice));

class ThrowingRuntimeSource : public TypedImageSource {
public:
    explicit ThrowingRuntimeSource(int failure)
        : TypedImageSource(makeAuthoredMipSource(false)), kind(failure) {}
    bool readMipLevel(char* destination, unsigned int level, unsigned int width,
                      unsigned int height, hipStream_t stream) override {
        if (enabled && level == 1) {
            if (kind == 0) return false;
            if (kind == 1) throw std::runtime_error("Injected whole-mip source read failure");
            if (kind == 2) throw std::bad_alloc();
            throw 7;
        }
        return TypedImageSource::readMipLevel(destination, level, width, height, stream);
    }
    int kind;
    bool enabled = true;
};

class WholeMipSourceFailure : public WholeMipRuntime, public testing::WithParamInterface<int> {};
TEST_P(WholeMipSourceFailure, ReadFailuresAndExceptionsPreserveResidentSuffixAndCanExplicitlyRetry) {
    auto source = std::make_shared<ThrowingRuntimeSource>(GetParam());
    wm::Texture texture(source, runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    ASSERT_EQ(texture.resize(2), Outcome::Success);
    const auto previous = firstEntry(texture);
    const auto expected = GetParam() == 2 ? Outcome::HostOutOfMemory : Outcome::SourceFailure;
    EXPECT_EQ(texture.resize(0), expected);
    const auto state = status(texture);
    EXPECT_EQ(state.primary.outcome, expected);
    EXPECT_EQ(state.primary.operation, cap::Operation::SourceRead);
    EXPECT_EQ(state.primary.rawHipError, 0);
    EXPECT_EQ(state.cleanup.outcome, Outcome::Success);
    EXPECT_EQ(state.residentBytes, 20u);
    EXPECT_EQ(state.pendingBytes, 0u);
    EXPECT_EQ(state.retiringBytes, 0u);
    EXPECT_EQ(state.pinnedBytes, 0u);
    EXPECT_EQ(state.sourceBytes, 20u + 256u);
    EXPECT_EQ(state.uploadedBytes, 20u + 256u);
    const auto preserved = firstEntry(texture);
    EXPECT_EQ(preserved.texture.textureObject, previous.texture.textureObject);
    expectPayload(preserved, *source, 2);
    source->enabled = false;
    EXPECT_EQ(texture.resize(0), Outcome::Success);
    expectPayload(firstEntry(texture), *source, 0);
}
INSTANTIATE_TEST_SUITE_P(SourceErrorKinds, WholeMipSourceFailure, testing::Range(0, 3));

TEST_F(WholeMipRuntime, NonstandardSourceExceptionPropagatesAndRetainsCandidateUntilExplicitCollection) {
    RuntimeFaultScope faults;
    auto source = std::make_shared<ThrowingRuntimeSource>(3);
    auto options = runtimeOptions();
    options.maxPinnedBytes = 256;
    options.maxManagedBytes = metadataBytes(options) + 20 + 340;
    wm::Texture texture(source, runtimeDescriptor(), options);
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    ASSERT_EQ(texture.resize(2), Outcome::Success);
    const auto previous = firstEntry(texture);
    const auto freedStaging = faults.calls(HipOperation::FreeHost);

    ASSERT_THROW(texture.resize(0), int);
    const auto interrupted = status(texture);
    EXPECT_EQ(interrupted.operation, Outcome::RuntimeFailure);
    EXPECT_EQ(interrupted.primary.outcome, Outcome::RuntimeFailure);
    EXPECT_EQ(interrupted.cleanup.outcome, Outcome::Success);
    EXPECT_EQ(interrupted.residentBytes, 20u);
    EXPECT_EQ(interrupted.pendingBytes, 0u);
    EXPECT_EQ(interrupted.retiringBytes, 340u);
    EXPECT_EQ(interrupted.pinnedBytes, 256u);
    EXPECT_EQ(interrupted.pinnedPeakBytes, options.maxPinnedBytes);
    EXPECT_EQ(interrupted.decodedPeakBytes, 256u);
    EXPECT_EQ(interrupted.sourceBytes, 20u + 256u);
    EXPECT_EQ(interrupted.uploadedBytes, 20u + 256u);
    EXPECT_EQ(interrupted.authoredReads, 3u);
    EXPECT_EQ(interrupted.managedPeakBytes, options.maxManagedBytes);
    EXPECT_EQ(interrupted.mips.firstResidentMip, 2u);
    EXPECT_FLOAT_EQ(interrupted.device.submittedSampler.maxMipmapLevelClamp, 1.f);
    EXPECT_FLOAT_EQ(interrupted.device.returnedSampler.maxMipmapLevelClamp, 1.f);
    EXPECT_EQ(faults.calls(HipOperation::FreeHost), freedStaging);
    EXPECT_EQ(faults.calls(HipOperation::FreeMipmapped), 0u);
    EXPECT_EQ(faults.calls(HipOperation::DestroySampler), 0u);

    const auto preserved = firstEntry(texture);
    EXPECT_EQ(preserved.texture.textureObject, previous.texture.textureObject);
    expectPayload(preserved, *source, 2);
    EXPECT_EQ(status(texture).retiringBytes, 340u);
    EXPECT_EQ(status(texture).pinnedBytes, 256u);
    ASSERT_EQ(texture.collectRetired(), Outcome::Success);
    const auto collected = status(texture);
    EXPECT_EQ(collected.residentBytes, 20u);
    EXPECT_EQ(collected.pendingBytes, 0u);
    EXPECT_EQ(collected.retiringBytes, 0u);
    EXPECT_EQ(collected.pinnedBytes, 0u);
    EXPECT_EQ(faults.calls(HipOperation::FreeHost), freedStaging + 1);
    EXPECT_EQ(faults.calls(HipOperation::FreeMipmapped), 1u);
    EXPECT_EQ(faults.calls(HipOperation::DestroySampler), 0u);
    EXPECT_EQ(firstEntry(texture).texture.textureObject, previous.texture.textureObject);
    source->enabled = false;
    EXPECT_EQ(texture.resize(0), Outcome::Success);
    expectPayload(firstEntry(texture), *source, 0);
}

struct RuntimeReadGate {
    std::mutex mutex;
    std::condition_variable condition;
    bool entered = false;
    bool released = false;
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        condition.notify_all();
    }
    bool waitForRead() {
        std::unique_lock<std::mutex> lock(mutex);
        return condition.wait_for(lock, std::chrono::seconds(10), [&] { return entered; });
    }
};

class BlockedRuntimeSource : public TypedImageSource {
public:
    explicit BlockedRuntimeSource(std::shared_ptr<RuntimeReadGate> gate)
        : TypedImageSource(makeAuthoredMipSource(false)), gate_(std::move(gate)) {}
    bool readMipLevel(char* destination, unsigned int level, unsigned int width,
                      unsigned int height, hipStream_t stream) override {
        if (block) {
            std::unique_lock<std::mutex> lock(gate_->mutex);
            gate_->entered = true;
            gate_->condition.notify_all();
            gate_->condition.wait(lock, [&] { return gate_->released; });
        }
        return TypedImageSource::readMipLevel(destination, level, width, height, stream);
    }
    bool block = true;
private:
    std::shared_ptr<RuntimeReadGate> gate_;
};

struct ReleaseRuntimeGate {
    std::shared_ptr<RuntimeReadGate> gate;
    ~ReleaseRuntimeGate() { gate->release(); }
};

class WholeMipBlockedRead : public WholeMipRuntime, public testing::WithParamInterface<bool> {};
TEST_P(WholeMipBlockedRead, StatusPollsDoNotWaitForSourceAndCancelCannotRepublishCapturedWork) {
    RuntimeFaultScope faults;
    const auto gate = std::make_shared<RuntimeReadGate>();
    auto source = std::make_shared<BlockedRuntimeSource>(gate);
    std::weak_ptr<BlockedRuntimeSource> weak = source;
    auto options = runtimeOptions();
    options.maxPinnedBytes = 256;
    options.maxDecodedBytes = 256;
    options.maxManagedBytes = metadataBytes(options) + 340 + (GetParam() ? 20 : 0);
    auto texture = std::make_unique<wm::Texture>(source, runtimeDescriptor(), options);
    ASSERT_TRUE(texture->addSampler(runtimeDescriptor()).succeeded());
    if (GetParam()) {
        source->block = false;
        ASSERT_EQ(texture->resize(2), Outcome::Success);
        source->block = true;
    }
    auto loading = std::async(std::launch::async, [&] { return texture->resize(0); });
    ReleaseRuntimeGate release{gate};
    ASSERT_TRUE(gate->waitForRead());
    source.reset();
    EXPECT_FALSE(weak.expired());
    auto polling = std::async(std::launch::async, [&] {
        wm::Status observed;
        const auto outcome = texture->getStatus(observed);
        return std::make_pair(outcome, observed);
    });
    const auto responsive = polling.wait_for(std::chrono::seconds(2));
    if (responsive != std::future_status::ready) gate->release();
    ASSERT_EQ(responsive, std::future_status::ready) << "Status must not hold the source operation lock";
    const auto observed = polling.get();
    EXPECT_EQ(observed.first, Outcome::Success);
    EXPECT_EQ(observed.second.operation, Outcome::Pending);
    EXPECT_EQ(observed.second.pendingBytes, 340u);
    EXPECT_EQ(observed.second.residentBytes, GetParam() ? 20u : 0u);
    EXPECT_EQ(observed.second.pinnedBytes, 256u);
    EXPECT_EQ(observed.second.pinnedPeakBytes, 256u);
    EXPECT_LE(observed.second.pinnedBytes, options.maxPinnedBytes);
    EXPECT_EQ(observed.second.overheadBytes, metadataBytes(options));
    EXPECT_EQ(observed.second.managedPeakBytes, options.maxManagedBytes);
    EXPECT_EQ(observed.second.residentBytes + observed.second.pendingBytes +
              observed.second.retiringBytes + observed.second.overheadBytes, options.maxManagedBytes);
    std::promise<void> started;
    auto startedFuture = started.get_future();
    auto cancelling = std::async(std::launch::async, [&] {
        started.set_value();
        return texture->cancel();
    });
    startedFuture.wait();
    EXPECT_EQ(cancelling.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
    EXPECT_FALSE(weak.expired());
    gate->release();
    EXPECT_EQ(loading.get(), Outcome::Cancelled);
    EXPECT_EQ(cancelling.get(), Outcome::Success);
    const auto cancelled = status(*texture);
    expectNoBacking(cancelled);
    EXPECT_EQ(cancelled.device.state, cap::State::Cancelled);
    EXPECT_EQ(cancelled.pinnedPeakBytes, options.maxPinnedBytes);
    EXPECT_EQ(cancelled.decodedPeakBytes, options.maxDecodedBytes);
    EXPECT_LE(cancelled.managedPeakBytes, options.maxManagedBytes);
    const auto uploadCalls = faults.calls(HipOperation::Upload);
    EXPECT_EQ(uploadCalls, GetParam() ? 2u : 0u);
    EXPECT_EQ(texture->resize(0), Outcome::Cancelled);
    EXPECT_EQ(texture->addSampler(runtimeDescriptor()).outcome, Outcome::Cancelled);
    wm::DeviceContext context;
    EXPECT_EQ(texture->prepare(nullptr, context), Outcome::Cancelled);
    EXPECT_EQ(context.entries, nullptr);
    EXPECT_FALSE(weak.expired()) << "Cancellation is not release of source ownership";
    texture.reset();
    EXPECT_TRUE(weak.expired());
}
INSTANTIATE_TEST_SUITE_P(InitialAndResidentCancellation, WholeMipBlockedRead, testing::Bool());

TEST_F(WholeMipRuntime, SourceRemainsOwnedAfterCallerReleaseThroughUploadUnloadAndReload) {
    auto source = runtimeSource();
    std::weak_ptr<TypedImageSource> weak = source;
    auto texture = std::make_unique<wm::Texture>(source, runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture->addSampler(runtimeDescriptor()).succeeded());
    source.reset();
    EXPECT_FALSE(weak.expired());
    ASSERT_EQ(texture->resize(2), Outcome::Success);
    expectPayload(firstEntry(*texture), *weak.lock(), 2);
    ASSERT_EQ(texture->unload(), Outcome::Success);
    EXPECT_FALSE(weak.expired());
    ASSERT_EQ(texture->resize(0), Outcome::Success);
    expectPayload(firstEntry(*texture), *weak.lock(), 0);
    texture.reset();
    EXPECT_TRUE(weak.expired());
}

struct RetainedLegacySnapshot {
    DeviceContext context{};
    ~RetainedLegacySnapshot() {
        if (context.textures) EXPECT_EQ(hipFree(context.textures), hipSuccess);
        if (context.residentFlags) EXPECT_EQ(hipFree(context.residentFlags), hipSuccess);
    }
    void initialize(hipTextureObject_t sampler) {
        context.maxTextures = 1;
        ASSERT_EQ(hipMalloc(&context.textures, sizeof(TextureObject)), hipSuccess);
        ASSERT_EQ(hipMalloc(&context.residentFlags, sizeof(uint32_t)), hipSuccess);
        const uint32_t resident = 1;
        ASSERT_EQ(hipMemcpy(context.textures, &sampler, sizeof(sampler), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(context.residentFlags, &resident, sizeof(resident), hipMemcpyHostToDevice), hipSuccess);
    }
};

class WholeMipQueuedConsumer : public WholeMipRuntime, public testing::WithParamInterface<int> {};
TEST_P(WholeMipQueuedConsumer, RetirementFencesARealQueuedConsumerThatAlreadyHoldsOldSampler) {
    RuntimeFaultScope faults;
    const auto source = runtimeSource(true);
    auto texture = std::make_unique<wm::Texture>(source, runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture->addSampler(runtimeDescriptor()).succeeded());
    ASSERT_EQ(texture->resize(1), Outcome::Success);
    const auto old = firstEntry(*texture);
    RetainedLegacySnapshot retained;
    retained.initialize(reinterpret_cast<hipTextureObject_t>(old.texture.textureObject));
    TextureSamplingHarness harness;
    ASSERT_EQ(harness.open(samplingModulePath()), hipSuccess);
    SamplingInput input;
    input.textureId = 0;
    input.path = SamplingPath::DelayedSnapshot;
    input.lod = 0;
    std::vector<SamplingResult> results;
    Outcome operation = Outcome::Pending;
    ASSERT_EQ(harness.sample(retained.context, {input}, results, [&] {
        switch (GetParam()) {
        case 0: operation = texture->resize(0); break;
        case 1: operation = texture->resize(2); break;
        case 2: operation = texture->unload(); break;
        case 3: operation = texture->cancel(); break;
        case 4: texture.reset(); operation = Outcome::Success; break;
        }
    }), hipSuccess);
    EXPECT_EQ(operation, Outcome::Success);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].resident, 1u);
    const auto expected = authoredFloatMipColors[1];
    EXPECT_NEAR(results[0].value.x, expected[0], 1e-6f);
    EXPECT_NEAR(results[0].value.y, expected[1], 1e-6f);
    EXPECT_NEAR(results[0].value.z, expected[2], 1e-6f);
    EXPECT_NEAR(results[0].value.w, expected[3], 1e-6f);
    EXPECT_GE(faults.calls(HipOperation::DestroySampler), 1u);
    EXPECT_GE(faults.calls(HipOperation::FreeMipmapped), 1u);
    if (texture && GetParam() <= 1)
        expectPayload(firstEntry(*texture), *source, GetParam() == 0 ? 0 : 2);
}
INSTANTIATE_TEST_SUITE_P(AllRetirementBoundaries, WholeMipQueuedConsumer, testing::Range(0, 5));

TEST_F(WholeMipRuntime, ResizeEndsPreparedEpochAndNextPrepareUsesNewCompleteSnapshot) {
    wm::Texture texture(runtimeSource(), runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    ASSERT_EQ(texture.resize(2), Outcome::Success);
    wm::DeviceContext before;
    ASSERT_EQ(texture.prepare(nullptr, before), Outcome::Success);
    ASSERT_EQ(texture.resize(0), Outcome::Success);
    wm::DeviceContext after;
    ASSERT_EQ(texture.prepare(nullptr, after), Outcome::Success);
    const auto entries = readEntries(after);
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].texture.mips.firstResidentMip, 0u);
    EXPECT_EQ(texture.processRequests(), Outcome::Success);
}

class WholeMipPreparedDemand : public WholeMipRuntime, public testing::WithParamInterface<int> {};
TEST_P(WholeMipPreparedDemand, ManualResizeNeverDiscardsAnUndrainedRequestOrItsOverflowFlag) {
    RuntimeFaultScope faults;
    auto options = runtimeOptions();
    options.maxRequests = 1;
    auto source = runtimeSource();
    wm::Texture texture(source, runtimeDescriptor(), options);
    const auto registered = texture.addSampler(runtimeDescriptor());
    ASSERT_TRUE(registered.succeeded());
    ASSERT_EQ(texture.resize(2), Outcome::Success);
    wm::DeviceContext context;
    ASSERT_EQ(texture.prepare(nullptr, context), Outcome::Success);
    const auto before = readEntries(context);
    ASSERT_EQ(before.size(), 1u);
    const cv::RequestKey queued{registered.key, 1, 0, 0};
    const uint32_t count = GetParam() == 2 ? 2 : 1, overflow = GetParam() == 1 ? 1 : 0;
    ASSERT_EQ(hipMemcpy(context.requests, &queued, sizeof(queued), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(context.requestCount, &count, sizeof(count), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(context.requestOverflow, &overflow, sizeof(overflow), hipMemcpyHostToDevice), hipSuccess);

    EXPECT_EQ(texture.resize(3), GetParam() ? Outcome::RequestOverflow : Outcome::InvalidTransition);
    EXPECT_EQ(status(texture).mips.firstResidentMip, 2u);
    EXPECT_EQ(status(texture).residentBytes, 20u);
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{2, 3}));
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped), 1u);
    EXPECT_EQ(faults.calls(HipOperation::AllocateArray), 0u);
    EXPECT_EQ(faults.calls(HipOperation::DestroySampler), 0u);
    EXPECT_EQ(faults.calls(HipOperation::FreeMipmapped), 0u);
    expectPayload(before[0], *source, 2);
    wm::DeviceContext next;
    EXPECT_EQ(texture.prepare(nullptr, next), Outcome::InvalidTransition);
    uint32_t retainedCount = 0, retainedOverflow = 0;
    cv::RequestKey retained;
    ASSERT_EQ(hipMemcpy(&retainedCount, context.requestCount, sizeof(retainedCount),
                        hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&retainedOverflow, context.requestOverflow, sizeof(retainedOverflow),
                        hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&retained, context.requests, sizeof(retained), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(retainedCount, count);
    EXPECT_EQ(retainedOverflow, overflow);
    EXPECT_TRUE(retained.texture == queued.texture);
    EXPECT_EQ(retained.revision, queued.revision);
    EXPECT_EQ(retained.originalMip, queued.originalMip);
    EXPECT_EQ(retained.reserved, queued.reserved);

    EXPECT_EQ(texture.processRequests(), GetParam() ? Outcome::RequestOverflow : Outcome::Success);
    EXPECT_EQ(status(texture).requestCount, count);
    EXPECT_EQ(status(texture).requestOverflow, overflow);
    if (GetParam()) {
        EXPECT_EQ(status(texture).mips.firstResidentMip, 2u);
        EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{2, 3}));
        ASSERT_EQ(request(texture, {queued}), Outcome::Success);
    }
    EXPECT_EQ(status(texture).mips.firstResidentMip, 0u);
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{2, 3, 0, 1, 2, 3}));
    expectPayload(firstEntry(texture), *source, 0);
}
INSTANTIATE_TEST_SUITE_P(PreparedQueuePreservation, WholeMipPreparedDemand, testing::Range(0, 3));

TEST_F(WholeMipRuntime, UnloadPermitsNewVariantsButNeverRecyclesEarlierRegistrationSlots) {
    const auto desc = runtimeDescriptor();
    wm::Texture texture(runtimeSource(), desc, runtimeOptions());
    const auto first = texture.addSampler(desc);
    ASSERT_TRUE(first.succeeded());
    ASSERT_EQ(texture.resize(2), Outcome::Success);
    ASSERT_EQ(texture.unload(), Outcome::Success);
    auto variant = desc;
    variant.addressMode[1] = hipAddressModeClamp;
    const auto second = texture.addSampler(variant);
    ASSERT_TRUE(second.succeeded());
    EXPECT_EQ(second.key.slot, first.key.slot + 1);
    EXPECT_EQ(second.key.generation, first.key.generation);
    EXPECT_EQ(second.key.incarnation, first.key.incarnation);
    EXPECT_TRUE(texture.addSampler(desc).key == first.key);
    EXPECT_EQ(request(texture, {{second.key, 1, 2, 0}}), Outcome::Success);
    const auto entries = snapshot(texture);
    ASSERT_EQ(entries.size(), 2u);
    EXPECT_TRUE(entries[0].texture.key == first.key);
    EXPECT_TRUE(entries[1].texture.key == second.key);
    EXPECT_NE(entries[0].texture.textureObject, 0u);
    EXPECT_NE(entries[1].texture.textureObject, 0u);
}

class WholeMipPhysicalShape : public WholeMipRuntime,
    public testing::WithParamInterface<std::tuple<uint32_t, uint32_t>> {};
TEST_P(WholeMipPhysicalShape, AuthoredNpotRectangularAndThinSuffixesHaveOnlyTheirTrueExtentAndPayload) {
    const auto [width, height] = GetParam();
    auto source = std::make_shared<TypedImageSource>(makeFilteringMipSource(width, height, true));
    wm::Texture texture(source, runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    const uint32_t first = std::min(2u, source->info.numMipLevels - 1);
    ASSERT_EQ(texture.resize(first), Outcome::Success);
    expectPayload(firstEntry(texture), *source, first);
    const auto state = status(texture);
    EXPECT_EQ(state.residentBytes, suffixBytes(*source, first));
    EXPECT_EQ(state.uploadedBytes, suffixBytes(*source, first));
    EXPECT_EQ(state.sourceBytes, suffixBytes(*source, first));
    EXPECT_EQ(state.authoredReads, source->info.numMipLevels - first);
    ASSERT_FALSE(source->readLevels.empty());
    EXPECT_EQ(source->readLevels.front(), first);
    EXPECT_EQ(state.pinnedPeakBytes, source->mipPixels[first].size());
    EXPECT_EQ(state.decodedPeakBytes, source->mipPixels[first].size());
    ASSERT_EQ(texture.resize(0), Outcome::Success);
    expectPayload(firstEntry(texture), *source, 0);
    ASSERT_EQ(texture.resize(first), Outcome::Success);
    expectPayload(firstEntry(texture), *source, first);
}
INSTANTIATE_TEST_SUITE_P(PhysicalShapes, WholeMipPhysicalShape,
    testing::Values(std::make_tuple(16u, 8u), std::make_tuple(13u, 7u),
                    std::make_tuple(1u, 17u), std::make_tuple(17u, 1u), std::make_tuple(1u, 1u)));

TEST_F(WholeMipRuntime, AuthoredMipPolicyDoesNotRegenerateOrReadUnrequestedFineLevels) {
    RuntimeFaultScope faults;
    const auto source = runtimeSource();
    auto desc = runtimeDescriptor();
    desc.generateMipmaps = false;
    desc.maxMipLevel = 3;
    wm::Texture texture(source, desc, runtimeOptions());
    ASSERT_TRUE(texture.addSampler(desc).succeeded());
    ASSERT_EQ(texture.resize(1), Outcome::Success);
    expectPayload(firstEntry(texture), *source, 1, 3);
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{1, 2}));
    EXPECT_EQ(status(texture).generatedLevels, 0u);
    EXPECT_EQ(status(texture).sourceBytes, 80u);
    EXPECT_EQ(status(texture).uploadedBytes, 80u);
    EXPECT_EQ(status(texture).residentBytes, 80u);
    EXPECT_EQ(faults.calls(HipOperation::GetLevel), 2u);
    EXPECT_EQ(faults.calls(HipOperation::Upload), 2u);
}

TEST_F(WholeMipRuntime, InitialSourceFailurePublishesTerminalEntriesAndNeverRetriesOnAnEmptyEpoch) {
    auto source = runtimeSource();
    source->failRead = true;
    wm::Texture texture(source, runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    EXPECT_EQ(texture.resize(2), Outcome::SourceFailure);
    const auto failed = status(texture);
    EXPECT_EQ(failed.primary.operation, cap::Operation::SourceRead);
    EXPECT_EQ(failed.primary.outcome, Outcome::SourceFailure);
    EXPECT_EQ(failed.sourceBytes, 0u);
    EXPECT_EQ(failed.uploadedBytes, 0u);
    expectNoBacking(failed);
    const auto entry = firstEntry(texture);
    EXPECT_EQ(entry.texture.residency, Outcome::SourceFailure);
    EXPECT_EQ(entry.texture.textureObject, 0u);
    const auto reads = source->reads;
    source->failRead = false;
    EXPECT_EQ(request(texture, {}), Outcome::Success);
    EXPECT_EQ(source->reads, reads);
    EXPECT_EQ(firstEntry(texture).texture.residency, Outcome::SourceFailure);
    EXPECT_EQ(texture.resize(2), Outcome::Success);
    expectPayload(firstEntry(texture), *source, 2);
}

TEST_F(WholeMipRuntime, SamplerFailureRetainsOriginalErrorWhenDestroyingEarlierSiblingAlsoFails) {
    RuntimeFaultScope faults;
    auto source = runtimeSource();
    const auto desc = runtimeDescriptor();
    auto variant = desc;
    variant.addressMode[0] = hipAddressModeClamp;
    wm::Texture texture(source, desc, runtimeOptions());
    ASSERT_TRUE(texture.addSampler(desc).succeeded());
    ASSERT_TRUE(texture.addSampler(variant).succeeded());
    faults.state->fail(HipOperation::CreateSampler, hipErrorNotSupported, 2);
    faults.state->fail(HipOperation::DestroySampler, hipErrorUnknown);
    EXPECT_EQ(texture.resize(2), Outcome::Unsupported);
    const auto state = status(texture);
    EXPECT_EQ(state.primary.operation, cap::Operation::CreateSampler);
    EXPECT_EQ(state.primary.rawHipError, static_cast<int32_t>(hipErrorNotSupported));
    EXPECT_EQ(state.cleanup.operation, cap::Operation::DestroySampler);
    EXPECT_EQ(state.cleanup.rawHipError, static_cast<int32_t>(hipErrorUnknown));
    EXPECT_EQ(state.residentBytes, 0u);
    EXPECT_EQ(state.retiringBytes, 20u);
    EXPECT_EQ(faults.calls(HipOperation::CreateSampler), 1u);
    EXPECT_EQ(faults.calls(HipOperation::FreeMipmapped), 0u);
    EXPECT_EQ(texture.collectRetired(), Outcome::Success);
    EXPECT_EQ(faults.calls(HipOperation::DestroySampler), 1u);
    EXPECT_EQ(faults.calls(HipOperation::FreeMipmapped), 1u);
    EXPECT_EQ(status(texture).retiringBytes, 0u);
    EXPECT_EQ(texture.resize(2), Outcome::Success);
}

TEST_F(WholeMipRuntime, ReturnedNonlegacyAnisotropyIsUnsupportedAndCannotReplaceValidBacking) {
    RuntimeFaultScope faults;
    auto source = runtimeSource();
    wm::Texture texture(source, runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    ASSERT_EQ(texture.resize(2), Outcome::Success);
    const auto old = firstEntry(texture);
    faults.state->overrideReturnedAnisotropy(4);
    EXPECT_EQ(texture.resize(0), Outcome::Unsupported);
    const auto failed = status(texture);
    EXPECT_EQ(failed.primary.operation, cap::Operation::ReadSampler);
    EXPECT_EQ(failed.primary.rawHipError, 0);
    EXPECT_EQ(failed.residentBytes, 20u);
    EXPECT_EQ(failed.retiringBytes, 0u);
    EXPECT_EQ(firstEntry(texture).texture.textureObject, old.texture.textureObject);
    faults.state->overrideReturnedAnisotropy(0);
    EXPECT_EQ(texture.resize(0), Outcome::Success);
    EXPECT_EQ(status(texture).submittedMaxAnisotropy, 0u);
    EXPECT_EQ(status(texture).device.submittedSampler.maxAnisotropy, 0u);
    EXPECT_EQ(status(texture).device.returnedSampler.maxAnisotropy, 0u);
}

TEST_F(WholeMipRuntime, FailureSeamIsCapturedByItsOwnerAndUnrelatedTextureUsesTheRealRuntime) {
    RuntimeFaultScope faults;
    wm::Texture captured(runtimeSource(), runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(captured.addSampler(runtimeDescriptor()).succeeded());
    internal::setHipFaultState(nullptr);
    wm::Texture unrelated(runtimeSource(), runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(unrelated.addSampler(runtimeDescriptor()).succeeded());
    faults.state->fail(HipOperation::AllocateMipmapped, hipErrorNotSupported);
    EXPECT_EQ(unrelated.resize(2), Outcome::Success);
    EXPECT_EQ(captured.resize(2), Outcome::Unsupported);
    EXPECT_EQ(captured.resize(2), Outcome::Success);
    EXPECT_EQ(status(unrelated).device.device, device_);
    EXPECT_EQ(status(captured).device.device, device_);
    EXPECT_EQ(faults.calls(HipOperation::AllocateMipmapped, false), 2u);
}

class WholeMipPrepareFailure : public WholeMipRuntime, public testing::WithParamInterface<HipOperation> {};
TEST_P(WholeMipPrepareFailure, FailedEpochPreparationDoesNotExposeContextOrDestroyResidentObjects) {
    RuntimeFaultScope faults;
    wm::Texture texture(runtimeSource(), runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    ASSERT_EQ(texture.resize(2), Outcome::Success);
    const auto old = firstEntry(texture);
    faults.state->fail(GetParam(), hipErrorUnknown);
    wm::DeviceContext context;
    EXPECT_EQ(texture.prepare(nullptr, context), Outcome::RuntimeFailure);
    EXPECT_EQ(context.entries, nullptr);
    EXPECT_EQ(status(texture).residentBytes, 20u);
    EXPECT_EQ(faults.calls(HipOperation::DestroySampler), 0u);
    EXPECT_EQ(faults.calls(HipOperation::FreeMipmapped), 0u);
    EXPECT_EQ(firstEntry(texture).texture.textureObject, old.texture.textureObject);
}
INSTANTIATE_TEST_SUITE_P(EpochFailures, WholeMipPrepareFailure,
    testing::Values(HipOperation::PublishMappings, HipOperation::SynchronizeConsumers, HipOperation::Initialize));

class WholeMipUnloadFailure : public WholeMipRuntime, public testing::WithParamInterface<HipOperation> {};
TEST_P(WholeMipUnloadFailure, FailedFenceOrInvalidationLeavesAllResidentStorageUsableForExplicitRetry) {
    RuntimeFaultScope faults;
    auto source = runtimeSource();
    wm::Texture texture(source, runtimeDescriptor(), runtimeOptions());
    ASSERT_TRUE(texture.addSampler(runtimeDescriptor()).succeeded());
    ASSERT_EQ(texture.resize(2), Outcome::Success);
    const auto old = firstEntry(texture);
    faults.state->fail(GetParam(), hipErrorUnknown);
    EXPECT_EQ(texture.unload(), Outcome::RuntimeFailure);
    EXPECT_EQ(status(texture).residentBytes, 20u);
    EXPECT_EQ(status(texture).retiringBytes, 0u);
    EXPECT_EQ(status(texture).primary.operation, cap::Operation::Publish);
    EXPECT_EQ(faults.calls(HipOperation::DestroySampler), 0u);
    EXPECT_EQ(faults.calls(HipOperation::FreeMipmapped), 0u);
    const auto preserved = firstEntry(texture);
    EXPECT_EQ(preserved.texture.textureObject, old.texture.textureObject);
    expectPayload(preserved, *source, 2);
    EXPECT_EQ(texture.unload(), Outcome::Success);
    expectNoBacking(status(texture));
}
INSTANTIATE_TEST_SUITE_P(UnloadFailures, WholeMipUnloadFailure,
    testing::Values(HipOperation::PublishMappings, HipOperation::SynchronizeConsumers));

TEST_F(WholeMipRuntime, KeyFromDestroyedLoaderCannotRequestTheNewLoadersTextureZero) {
    const auto desc = runtimeDescriptor();
    cv::GpuKey stale;
    {
        wm::Texture previous(runtimeSource(), desc, runtimeOptions());
        const auto registration = previous.addSampler(desc);
        ASSERT_TRUE(registration.succeeded());
        stale = registration.key;
        ASSERT_EQ(previous.resize(2), Outcome::Success);
    }
    auto source = runtimeSource();
    wm::Texture replacement(source, desc, runtimeOptions());
    const auto current = replacement.addSampler(desc);
    ASSERT_TRUE(current.succeeded());
    EXPECT_EQ(current.key.slot, stale.slot);
    EXPECT_EQ(current.key.generation, stale.generation);
    EXPECT_NE(current.key.incarnation, stale.incarnation);
    EXPECT_EQ(request(replacement, {{stale, 1, 0, 0}}), Outcome::InvalidKey);
    EXPECT_EQ(source->reads, 0u);
    EXPECT_EQ(status(replacement).rejectedRequests, 1u);
    EXPECT_EQ(request(replacement, {{current.key, 1, 2, 0}}), Outcome::Success);
    expectPayload(firstEntry(replacement), *source, 2);
}

class WholeMipFailedRefinement : public WholeMipRuntime, public testing::WithParamInterface<HipOperation> {};
TEST_P(WholeMipFailedRefinement, HostTerminalFailurePreservesStrictMissPreviewAndCompleteCoarseDecisions) {
    RuntimeFaultScope faults;
    auto source = runtimeSource(true);
    const auto desc = runtimeDescriptor();
    wm::Texture texture(source, desc, runtimeOptions());
    const auto strict = texture.addSampler(desc);
    const auto preview = texture.addSampler(desc, cv::SamplingPolicy::AllowCoarsePreview);
    ASSERT_TRUE(strict.succeeded());
    ASSERT_TRUE(preview.succeeded());
    ASSERT_EQ(texture.resize(2), Outcome::Success);
    const auto old = snapshot(texture);
    ASSERT_EQ(old.size(), 2u);
    const auto raw = GetParam() == HipOperation::AllocateMipmapped ? hipErrorOutOfMemory :
                     GetParam() == HipOperation::Upload ? hipErrorInvalidValue : hipErrorNotSupported;
    const auto expected = GetParam() == HipOperation::AllocateMipmapped ? Outcome::DeviceOutOfMemory :
                          GetParam() == HipOperation::Upload ? Outcome::InvalidInput : Outcome::Unsupported;
    faults.state->fail(GetParam(), raw);
    ASSERT_EQ(texture.resize(0), expected);
    EXPECT_EQ(status(texture).primary.outcome, expected);
    EXPECT_EQ(status(texture).mips.firstResidentMip, 2u);

    WholeMipHarness harness;
    ASSERT_EQ(harness.open(), hipSuccess);
    wm::DeviceContext context;
    ASSERT_EQ(texture.prepare(harness.stream(), context), Outcome::Success);
    const auto preserved = readEntries(context);
    ASSERT_EQ(preserved.size(), 2u);
    for (size_t id = 0; id < preserved.size(); ++id) {
        EXPECT_EQ(preserved[id].texture.residency, Outcome::Success);
        EXPECT_EQ(preserved[id].texture.textureObject, old[id].texture.textureObject);
        expectPayload(preserved[id], *source, 2);
    }
    WholeMipInput strictFine;
    strictFine.key = strict.key;
    strictFine.lod = 0;
    strictFine.defaultColor = make_float4(-9, 7, 5, -1);
    auto strictCoarse = strictFine;
    strictCoarse.lod = 2;
    auto previewFine = strictFine;
    previewFine.key = preview.key;
    auto previewCoarse = previewFine;
    previewCoarse.lod = 2;
    std::vector<WholeMipResult> results;
    ASSERT_EQ(harness.sample(context, {strictFine, strictCoarse, previewFine, previewCoarse}, results),
              hipSuccess);
    ASSERT_EQ(results.size(), 4u);
    EXPECT_EQ(results[0].decision.validity, cv::SampleValidity::Missing);
    EXPECT_EQ(results[0].decision.outcome, Outcome::Pending);
    EXPECT_FALSE(results[0].decision.contributesStrictSample());
    EXPECT_TRUE(results[0].decision.needsRequest());
    EXPECT_FLOAT_EQ(results[0].value.x, strictFine.defaultColor.x);
    EXPECT_FLOAT_EQ(results[0].value.y, strictFine.defaultColor.y);
    EXPECT_FLOAT_EQ(results[0].value.z, strictFine.defaultColor.z);
    EXPECT_FLOAT_EQ(results[0].value.w, strictFine.defaultColor.w);
    EXPECT_EQ(results[2].decision.validity, cv::SampleValidity::CoarsePreview);
    EXPECT_EQ(results[2].decision.outcome, Outcome::Pending);
    EXPECT_FALSE(results[2].decision.contributesStrictSample());
    EXPECT_TRUE(results[2].decision.needsRequest());
    for (size_t id : {size_t{0}, size_t{2}}) {
        EXPECT_EQ(results[id].decision.required.first, 0u);
        EXPECT_EQ(results[id].decision.required.last, 0u);
    }
    for (size_t id : {size_t{1}, size_t{3}}) {
        EXPECT_EQ(results[id].decision.validity, cv::SampleValidity::Complete);
        EXPECT_EQ(results[id].decision.outcome, Outcome::Success);
        EXPECT_TRUE(results[id].decision.contributesStrictSample());
        EXPECT_FALSE(results[id].decision.needsRequest());
        EXPECT_EQ(results[id].decision.required.first, 2u);
        EXPECT_EQ(results[id].decision.required.last, 2u);
    }
    for (size_t id : {size_t{1}, size_t{2}, size_t{3}}) {
        const auto& color = authoredFloatMipColors[2];
        EXPECT_NEAR(results[id].value.x, color[0], 1e-6f);
        EXPECT_NEAR(results[id].value.y, color[1], 1e-6f);
        EXPECT_NEAR(results[id].value.z, color[2], 1e-6f);
        EXPECT_NEAR(results[id].value.w, color[3], 1e-6f);
    }
    uint32_t count = 0;
    ASSERT_EQ(hipMemcpy(&count, context.requestCount, sizeof(count), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(count, 2u);
    std::vector<cv::RequestKey> requests(count);
    ASSERT_EQ(hipMemcpy(requests.data(), context.requests, requests.size() * sizeof(cv::RequestKey),
                        hipMemcpyDeviceToHost), hipSuccess);
    for (const auto key : {strict.key, preview.key}) {
        EXPECT_EQ(std::count_if(requests.begin(), requests.end(), [&](const auto& request) {
            return request.texture == key && request.revision == 1 && request.originalMip == 0 &&
                   request.reserved == 0;
        }), 1);
    }
    faults.state->fail(GetParam(), raw);
    EXPECT_EQ(texture.processRequests(), expected);
    EXPECT_EQ(status(texture).mips.firstResidentMip, 2u);
    EXPECT_EQ(status(texture).residentBytes, suffixBytes(*source, 2));
    ASSERT_EQ(texture.resize(0), Outcome::Success);
    ASSERT_EQ(texture.prepare(harness.stream(), context), Outcome::Success);
    ASSERT_EQ(harness.sample(context, {strictFine, previewFine}, results), hipSuccess);
    ASSERT_EQ(results.size(), 2u);
    for (const auto& result : results) {
        EXPECT_EQ(result.decision.validity, cv::SampleValidity::Complete);
        EXPECT_TRUE(result.decision.contributesStrictSample());
        const auto& color = authoredFloatMipColors[0];
        EXPECT_NEAR(result.value.x, color[0], 1e-6f);
        EXPECT_NEAR(result.value.y, color[1], 1e-6f);
        EXPECT_NEAR(result.value.z, color[2], 1e-6f);
        EXPECT_NEAR(result.value.w, color[3], 1e-6f);
    }
    EXPECT_EQ(texture.processRequests(), Outcome::Success);
}
INSTANTIATE_TEST_SUITE_P(FailedRefinementDecisions, WholeMipFailedRefinement,
    testing::Values(HipOperation::AllocateMipmapped, HipOperation::Upload, HipOperation::CreateSampler));

} // namespace
} } // namespace hip_demand::test
