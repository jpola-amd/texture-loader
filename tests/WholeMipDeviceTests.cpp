// SPDX-License-Identifier: MIT
#include "TestUtils.h"
#include "ImageDataTestUtils.h"
#include "WholeMipTestHarness.h"
#include <DemandLoading/WholeMipTexture.h>
#include <cmath>
#include <set>
#include <tuple>

namespace hip_demand { namespace test {
namespace {
namespace cv = contract_v1;
namespace wm = whole_mip_v1;

// Fixed before running any candidate. Only gradient LOD quantization gets the
// existing contrast/128 allowance, including suffix versus full-chain comparison.
constexpr float PixelTolerance = 1e-6f;
constexpr float GradientWeightTolerance = 1.f / 128;
using Color = std::array<float, 4>;

Color color(float4 value) { return {value.x, value.y, value.z, value.w}; }

WholeMipInput input(cv::GpuKey key, float lod = 0, WholeMipPath path = WholeMipPath::Lod) {
    WholeMipInput result;
    result.key = key;
    result.lod = lod;
    result.path = path;
    return result;
}

Color spatialOracle(const TypedImageSource& source, uint32_t level, float u, float v,
                    const TextureDesc& desc) {
    const int width = cv::mipDimension(source.info.width, level);
    const int height = cv::mipDimension(source.info.height, level);
    const auto address = [](int value, int size, hipTextureAddressMode mode) {
        return mode == hipAddressModeWrap ? (value % size + size) % size : std::clamp(value, 0, size - 1);
    };
    const auto fetch = [&](int x, int y) {
        Color result{};
        const size_t index = size_t(address(y, height, desc.addressMode[1])) * width +
                             address(x, width, desc.addressMode[0]);
        std::memcpy(result.data(), source.mipPixels.at(level).data() + index * sizeof(Color), sizeof(Color));
        return result;
    };
    if (desc.normalizedCoords) {
        u *= width;
        v *= height;
    }
    if (desc.filterMode == hipFilterModePoint)
        return fetch(int(std::floor(u)), int(std::floor(v)));
    u -= .5f;
    v -= .5f;
    const int x = int(std::floor(u)), y = int(std::floor(v));
    const float a = u - x, b = v - y;
    const auto p = fetch(x, y), q = fetch(x + 1, y), r = fetch(x, y + 1), s = fetch(x + 1, y + 1);
    Color result{};
    for (size_t c = 0; c < 4; ++c)
        result[c] = (1-b) * ((1-a)*p[c] + a*q[c]) + b * ((1-a)*r[c] + a*s[c]);
    return result;
}

struct Expected {
    Color value{}, tolerance{PixelTolerance, PixelTolerance, PixelTolerance, PixelTolerance};
};

Expected oracle(const TypedImageSource& source, const TextureDesc& desc, const WholeMipInput& in,
                uint32_t first = 0) {
    const uint32_t count = desc.maxMipLevel ? std::min(desc.maxMipLevel, source.info.numMipLevels)
                                         : source.info.numMipLevels;
    const float lod = std::clamp(in.lod, float(first), float(count - 1));
    const bool point = desc.mipmapFilterMode == hipFilterModePoint;
    const auto low = uint32_t(std::floor(lod + (point ? .5f : 0)));
    const auto high = point ? low : std::min(low + 1, count - 1);
    const float fraction = point ? 0 : lod - low;
    const auto a = spatialOracle(source, low, in.u, in.v, desc);
    const auto b = spatialOracle(source, high, in.u, in.v, desc);
    Expected result;
    for (size_t c = 0; c < 4; ++c) {
        result.value[c] = a[c] + fraction * (b[c] - a[c]);
        if (in.path == WholeMipPath::Gradient && fraction != 0)
            result.tolerance[c] += std::abs(b[c] - a[c]) * GradientWeightTolerance;
    }
    return result;
}

void expectPixel(const WholeMipResult& actual, const Expected& expected) {
    const auto channels = color(actual.value);
    for (size_t c = 0; c < 4; ++c)
        EXPECT_NEAR(channels[c], expected.value[c], expected.tolerance[c]) << "channel=" << c;
}

class WholeMipDevice : public HipTestFixture {
protected:
    void SetUp() override {
        HipTestFixture::SetUp();
        if (HasFatalFailure()) return;
        ASSERT_NO_THROW(ASSERT_EQ(harness.open(), hipSuccess));
    }
    void TearDown() override {
        EXPECT_EQ(harness.close(), hipSuccess);
        HipTestFixture::TearDown();
    }
    void sample(wm::Texture& texture, const std::vector<WholeMipInput>& inputs,
                std::vector<WholeMipResult>& results) {
        wm::DeviceContext context;
        ASSERT_EQ(texture.prepare(harness.stream(), context), cv::Outcome::Success);
        ASSERT_EQ(harness.sample(context, inputs, results), hipSuccess);
        ASSERT_EQ(texture.processRequests(), cv::Outcome::Success);
    }
    void snapshot(wm::Texture& texture, std::vector<wm::Entry>& entries) {
        wm::DeviceContext context;
        ASSERT_EQ(texture.prepare(harness.stream(), context), cv::Outcome::Success);
        ASSERT_EQ(hipStreamSynchronize(harness.stream()), hipSuccess);
        entries.resize(context.numSamplers);
        ASSERT_EQ(hipMemcpy(entries.data(), context.entries, entries.size() * sizeof(wm::Entry),
                            hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(texture.processRequests(), cv::Outcome::Success);
    }
    void storage(wm::Texture& texture, const TypedImageSource& source, uint32_t first,
                 const std::vector<TextureDesc>& descriptors) {
        std::vector<wm::Entry> entries;
        ASSERT_NO_FATAL_FAILURE(snapshot(texture, entries));
        ASSERT_EQ(entries.size(), descriptors.size());
        wm::Status status;
        ASSERT_EQ(texture.getStatus(status), cv::Outcome::Success);
        EXPECT_EQ(status.mips.firstResidentMip, first);
        EXPECT_EQ(status.mips.resourceWidth, cv::mipDimension(source.info.width, first));
        EXPECT_EQ(status.mips.resourceHeight, cv::mipDimension(source.info.height, first));
        const uint32_t levels = status.mips.originalLevels - first;
        EXPECT_EQ(status.mips.resourceLevels, levels);
        uint64_t bytes = 0;
        for (uint32_t level = first; level < status.mips.originalLevels; ++level)
            bytes += source.mipPixels[level].size();
        EXPECT_EQ(status.residentBytes, bytes) << "Shared payload must be charged once";
        EXPECT_EQ(status.pendingBytes, 0u);
        EXPECT_EQ(status.retiringBytes, 0u);
        EXPECT_EQ(status.submittedMaxAnisotropy, 0u);
        hipResourceDesc common{};
        for (size_t slot = 0; slot < entries.size(); ++slot) {
            const auto object = hipTextureObject_t(entries[slot].texture.textureObject);
            ASSERT_NE(object, nullptr);
            hipResourceDesc resource{};
            hipTextureDesc sampler{};
            ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, object), hipSuccess);
            ASSERT_EQ(hipGetTextureObjectTextureDesc(&sampler, object), hipSuccess);
            EXPECT_EQ(resource.resType, levels == 1 ? hipResourceTypeArray : hipResourceTypeMipmappedArray);
            EXPECT_EQ(sampler.filterMode, descriptors[slot].filterMode);
            EXPECT_EQ(entries[slot].descriptor.mipFilter, descriptors[slot].mipmapFilterMode == hipFilterModePoint
                ? cv::FilterMode::Point : cv::FilterMode::Linear);
            if (levels != 1)
                EXPECT_EQ(sampler.mipmapFilterMode, descriptors[slot].mipmapFilterMode);
            EXPECT_EQ(sampler.addressMode[0], descriptors[slot].addressMode[0]);
            EXPECT_EQ(sampler.addressMode[1], descriptors[slot].addressMode[1]);
            EXPECT_EQ(sampler.maxAnisotropy, 0u);
            EXPECT_EQ(sampler.normalizedCoords, descriptors[slot].normalizedCoords);
            if (slot) {
                if (levels == 1)
                    EXPECT_EQ(resource.res.array.array, common.res.array.array);
                else
                    EXPECT_EQ(resource.res.mipmap.mipmap, common.res.mipmap.mipmap);
                continue;
            }
            common = resource;
            for (uint32_t level = 0; level < levels; ++level) {
                hipArray_t array = resource.res.array.array;
                if (levels != 1)
                    ASSERT_EQ(hipGetMipmappedArrayLevel(&array, resource.res.mipmap.mipmap, level), hipSuccess);
                hipChannelFormatDesc format{};
                hipExtent extent{};
                unsigned flags = 0;
                ASSERT_EQ(hipArrayGetInfo(&format, &extent, &flags, array), hipSuccess);
                const uint32_t original = first + level;
                EXPECT_EQ(extent.width, cv::mipDimension(source.info.width, original));
                EXPECT_EQ(extent.height, cv::mipDimension(source.info.height, original));
                EXPECT_EQ(format.x, 32);
                const auto& expected = source.mipPixels[original];
                std::vector<unsigned char> actual(expected.size());
                const size_t pitch = extent.width * sizeof(float4);
                ASSERT_EQ(hipMemcpy2DFromArray(actual.data(), pitch, array, 0, 0, pitch, extent.height,
                                               hipMemcpyDeviceToHost), hipSuccess);
                EXPECT_EQ(actual, expected) << "original mip=" << original;
            }
            if (levels != 1) {
                hipArray_t extra{};
                EXPECT_NE(hipGetMipmappedArrayLevel(&extra, resource.res.mipmap.mipmap, levels), hipSuccess);
            }
        }
    }
    WholeMipHarness harness;
};

class WholeMipRequests : public WholeMipDevice {
protected:
    void SetUp() override {
        WholeMipDevice::SetUp();
        if (HasFatalFailure()) return;
        ASSERT_EQ(hipMalloc(&deviceEntries, 2 * sizeof(wm::Entry)), hipSuccess);
        ASSERT_EQ(hipMalloc(&deviceRequests, 130 * sizeof(cv::RequestKey)), hipSuccess);
        ASSERT_EQ(hipMalloc(&deviceCounters, 2 * sizeof(uint32_t)), hipSuccess);
        context.incarnation = 7;
        context.entries = deviceEntries;
        context.requests = deviceRequests + 1;
        context.requestCount = deviceCounters;
        context.requestOverflow = deviceCounters + 1;
        context.numSamplers = 2;
        context.maxRequests = 128;
        for (uint32_t slot = 0; slot < 2; ++slot) {
            entries[slot].texture.key = {slot, 1, 7};
            entries[slot].texture.revision = 1;
            entries[slot].texture.state = cv::RegistrationState::Live;
            entries[slot].texture.mips = {16, 16, 5};
        }
        ASSERT_NO_FATAL_FAILURE(reset());
    }
    void TearDown() override {
        if (deviceEntries) EXPECT_EQ(hipFree(deviceEntries), hipSuccess);
        if (deviceRequests) EXPECT_EQ(hipFree(deviceRequests), hipSuccess);
        if (deviceCounters) EXPECT_EQ(hipFree(deviceCounters), hipSuccess);
        WholeMipDevice::TearDown();
    }
    void reset() {
        ASSERT_EQ(hipMemcpy(deviceEntries, entries.data(), sizeof(entries), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(deviceRequests, 0xab, 130 * sizeof(cv::RequestKey)), hipSuccess);
        ASSERT_EQ(hipMemset(deviceCounters, 0, 2 * sizeof(uint32_t)), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);
    }
    void read(std::vector<cv::RequestKey>& requests, uint32_t expectedCount, uint32_t overflow = 0) {
        uint32_t counters[2]{};
        ASSERT_EQ(hipMemcpy(counters, deviceCounters, sizeof(counters), hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(counters[0], expectedCount);
        EXPECT_EQ(counters[1], overflow);
        ASSERT_LE(counters[0], context.maxRequests);
        requests.resize(counters[0]);
        if (!requests.empty())
            ASSERT_EQ(hipMemcpy(requests.data(), context.requests, requests.size() * sizeof(cv::RequestKey),
                                hipMemcpyDeviceToHost), hipSuccess);
        std::array<unsigned char, sizeof(cv::RequestKey)> guard{};
        ASSERT_EQ(hipMemcpy(guard.data(), deviceRequests, guard.size(), hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_TRUE(std::all_of(guard.begin(), guard.end(), [](auto v) { return v == 0xab; }));
        ASSERT_EQ(hipMemcpy(guard.data(), context.requests + context.maxRequests, guard.size(),
                            hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_TRUE(std::all_of(guard.begin(), guard.end(), [](auto v) { return v == 0xab; }));
    }
    std::array<wm::Entry, 2> entries{};
    wm::DeviceContext context;
    wm::Entry* deviceEntries = nullptr;
    cv::RequestKey* deviceRequests = nullptr;
    uint32_t* deviceCounters = nullptr;
};

TEST_F(WholeMipRequests, PointAndLinearRecordOriginalRequirementsBeforeClamping) {
    for (auto filter : {cv::FilterMode::Point, cv::FilterMode::Linear}) {
        entries[0].descriptor.mipFilter = filter;
        for (float lod : {-4.f, 0.f, .25f, .5f, .75f, 1.25f, 3.75f, 4.f, 99.f}) {
            SCOPED_TRACE(testing::Message() << "filter=" << int(filter) << " lod=" << lod);
            ASSERT_NO_FATAL_FAILURE(reset());
            const auto requirement = cv::requiredLevels(lod, 5, filter);
            std::vector<WholeMipResult> results;
            ASSERT_EQ(harness.sample(context, {input(entries[0].texture.key, lod)}, results), hipSuccess);
            ASSERT_EQ(results.size(), 1u);
            EXPECT_EQ(results[0].decision.validity, cv::SampleValidity::Missing);
            EXPECT_EQ(results[0].decision.outcome, cv::Outcome::Pending);
            EXPECT_EQ(results[0].decision.required.first, requirement.levels.first);
            EXPECT_EQ(results[0].decision.required.last, requirement.levels.last);
            EXPECT_EQ(color(results[0].value), (Color{1, 0, 1, 1}));
            std::vector<cv::RequestKey> requests;
            ASSERT_NO_FATAL_FAILURE(read(requests, requirement.levels.last - requirement.levels.first + 1));
            for (size_t i = 0; i < requests.size(); ++i) {
                EXPECT_TRUE(requests[i].texture == entries[0].texture.key);
                EXPECT_EQ(requests[i].revision, 1u);
                EXPECT_EQ(requests[i].originalMip, requirement.levels.first + i);
            }
        }
    }
}

TEST_F(WholeMipRequests, FullKeyLeaderCoversSingletonPartialAndCompleteWaves) {
    for (uint32_t count : {1u, 3u, 32u, 64u}) {
        ASSERT_NO_FATAL_FAILURE(reset());
        std::vector<WholeMipInput> inputs(count, input(entries[0].texture.key, 0, WholeMipPath::WaveLeader));
        std::vector<WholeMipResult> results;
        ASSERT_EQ(harness.sample(context, inputs, results), hipSuccess);
        for (uint32_t index = 0; index < count; ++index) {
            const auto& result = results[index];
            SCOPED_TRACE(testing::Message() << "count=" << count << " lane=" << index
                << " active=" << result.value.y << " slotMatch=" << result.value.w);
            EXPECT_EQ(result.value.x, float(index % result.waveSize));
            EXPECT_EQ(result.value.z, index % result.waveSize == 0 ? 1.f : 0.f);
            const auto pointer = uint64_t(result.decision.required.first) |
                                 (uint64_t(result.decision.required.last) << 32);
            EXPECT_EQ(pointer, reinterpret_cast<uint64_t>(deviceCounters));
        }
        std::vector<cv::RequestKey> requests;
        ASSERT_NO_FATAL_FAILURE(read(requests, (count + results[0].waveSize - 1) / results[0].waveSize));
    }
}

TEST_F(WholeMipRequests, MalformedContextsAreRejectedBeforeAnyPointerLookup) {
    std::vector<std::pair<wm::DeviceContext, cv::Outcome>> cases;
    const auto add = [&](auto edit, cv::Outcome expected = cv::Outcome::InvalidInput) {
        auto changed = context;
        changed.entries = reinterpret_cast<const wm::Entry*>(uintptr_t(1));
        edit(changed);
        cases.emplace_back(changed, expected);
    };
    add([](auto& c) { ++c.abi.version; }, cv::Outcome::AbiMismatch);
    add([](auto& c) { --c.abi.byteSize; }, cv::Outcome::AbiMismatch);
    add([](auto& c) { c.incarnation = 0; });
    add([](auto& c) { c.entries = nullptr; });
    add([](auto& c) { c.requests = nullptr; });
    add([](auto& c) { c.requestCount = nullptr; });
    add([](auto& c) { c.requestOverflow = nullptr; });
    add([](auto& c) { c.numSamplers = 4097; });
    add([](auto& c) { c.maxRequests = 0; });
    add([](auto& c) { c.maxRequests = 1048577; });
    add([](auto& c) { c.numSamplers = 0; }, cv::Outcome::InvalidKey);
    for (const auto& test : cases) {
        std::vector<WholeMipResult> results;
        ASSERT_EQ(harness.sample(test.first, {input(entries[0].texture.key)}, results), hipSuccess);
        EXPECT_EQ(results[0].decision.outcome, test.second);
        EXPECT_EQ(results[0].decision.validity, cv::SampleValidity::Invalid);
    }
    std::vector<cv::RequestKey> requests;
    ASSERT_NO_FATAL_FAILURE(read(requests, 0));
}

TEST_F(WholeMipRequests, InvalidStaleRevisionAndOutOfBoundsKeysNeverDemandZero) {
    const auto good = entries[0].texture.key;
    std::vector<WholeMipInput> inputs;
    for (cv::GpuKey key : {cv::GpuKey{}, cv::GpuKey{2, 1, 7}, cv::GpuKey{UINT32_MAX, 1, 7},
                          cv::GpuKey{0, 0, 7}, cv::GpuKey{0, 2, 7}, cv::GpuKey{0, 1, 8},
                          cv::GpuKey{0, 1, 0}}) {
        inputs.push_back(input(key));
        inputs.push_back(input(key, 0, WholeMipPath::RecordRequest));
    }
    for (uint64_t revision : {uint64_t(0), uint64_t(2), UINT64_MAX}) {
        auto in = input(good);
        in.revision = revision;
        inputs.push_back(in);
    }
    std::vector<WholeMipResult> results;
    ASSERT_EQ(harness.sample(context, inputs, results), hipSuccess);
    for (const auto& result : results) {
        EXPECT_EQ(result.decision.outcome, cv::Outcome::InvalidKey);
        EXPECT_FALSE(result.decision.needsRequest());
        EXPECT_EQ(color(result.value), (Color{1, 0, 1, 1}));
    }
    std::vector<cv::RequestKey> requests;
    ASSERT_NO_FATAL_FAILURE(read(requests, 0));
}

TEST_F(WholeMipRequests, TerminalAndMalformedEntriesDoNotGenerateDemand) {
    const auto original = entries[0];
    for (auto outcome : {cv::Outcome::SourceFailure, cv::Outcome::Unsupported, cv::Outcome::DeviceOutOfMemory,
                        cv::Outcome::CapacityExhausted, cv::Outcome::DemandTooLarge,
                        cv::Outcome::ProtectedBudgetExhausted, cv::Outcome::Cancelled, cv::Outcome::NoProgress,
                        cv::Outcome::RuntimeFailure, cv::Outcome::HostOutOfMemory}) {
        entries[0] = original;
        entries[0].texture.residency = outcome;
        ASSERT_NO_FATAL_FAILURE(reset());
        std::vector<WholeMipResult> results;
        ASSERT_EQ(harness.sample(context, {input(original.texture.key),
            input(original.texture.key, 0, WholeMipPath::RecordRequest)}, results), hipSuccess);
        for (const auto& result : results) {
            EXPECT_EQ(result.decision.outcome, outcome);
            EXPECT_FALSE(result.decision.needsRequest());
        }
        std::vector<cv::RequestKey> requests;
        ASSERT_NO_FATAL_FAILURE(read(requests, 0));
    }
    const auto check = [&](auto mutate, cv::Outcome expected) {
        entries[0] = original;
        mutate(entries[0]);
        ASSERT_NO_FATAL_FAILURE(reset());
        std::vector<WholeMipResult> results;
        ASSERT_EQ(harness.sample(context, {input(original.texture.key)}, results), hipSuccess);
        EXPECT_EQ(results[0].decision.outcome, expected);
        std::vector<cv::RequestKey> requests;
        ASSERT_NO_FATAL_FAILURE(read(requests, 0));
    };
    check([](auto& e) { e.texture.state = cv::RegistrationState::Retiring; }, cv::Outcome::InvalidKey);
    check([](auto& e) { e.texture.reserved = 1; }, cv::Outcome::InvalidInput);
    check([](auto& e) { e.texture.mips.originalWidth = 0; }, cv::Outcome::InvalidInput);
    check([](auto& e) { e.texture.residency = cv::Outcome::Success; }, cv::Outcome::InvalidTransition);
    check([](auto& e) { e.texture.textureObject = 1; }, cv::Outcome::InvalidTransition);
    check([](auto& e) { e.texture.mips.originalLevels = 1; }, cv::Outcome::Unsupported);
    check([](auto& e) { e.descriptor.abi.version = 2; }, cv::Outcome::AbiMismatch);
    check([](auto& e) { e.descriptor.mipFilter = cv::FilterMode(99); }, cv::Outcome::InvalidInput);
    check([](auto& e) { e.descriptor.maxAnisotropy = 2; }, cv::Outcome::Unsupported);
    check([](auto& e) { e.descriptor.maxAnisotropy = 16; }, cv::Outcome::Unsupported);
}

TEST_F(WholeMipRequests, DirectRequestsAndNonfiniteSamplingInputsAreValidated) {
    std::vector<WholeMipInput> inputs;
    for (uint32_t level : {5u, UINT32_MAX}) {
        auto in = input(entries[0].texture.key, 0, WholeMipPath::RecordRequest);
        in.originalMip = level;
        inputs.push_back(in);
    }
    auto reserved = input(entries[0].texture.key, 0, WholeMipPath::RecordRequest);
    reserved.requestReserved = 1;
    inputs.push_back(reserved);
    for (float value : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::quiet_NaN()}) {
        inputs.push_back(input(entries[0].texture.key, value));
        for (uint32_t component = 0; component < 4; ++component) {
            auto in = input(entries[0].texture.key, 0, WholeMipPath::Gradient);
            if (component == 0) in.ddx.x = value;
            if (component == 1) in.ddx.y = value;
            if (component == 2) in.ddy.x = value;
            if (component == 3) in.ddy.y = value;
            inputs.push_back(in);
        }
        auto in = input(entries[0].texture.key);
        in.u = value;
        inputs.push_back(in);
    }
    std::vector<WholeMipResult> results;
    ASSERT_EQ(harness.sample(context, inputs, results), hipSuccess);
    for (const auto& result : results) {
        EXPECT_EQ(result.decision.outcome, cv::Outcome::InvalidInput);
        EXPECT_FALSE(result.decision.needsRequest());
    }
    std::vector<cv::RequestKey> requests;
    ASSERT_NO_FATAL_FAILURE(read(requests, 0));
}

TEST_F(WholeMipRequests, WaveDedupUsesEveryIdentityFieldAndActualWaveSize) {
    std::vector<WholeMipInput> inputs(64);
    for (uint32_t lane = 0; lane < inputs.size(); ++lane) {
        auto& in = inputs[lane];
        in = input({0, 1, 7}, 0, WholeMipPath::RecordLocalRequest);
        switch (lane % 8) {
        case 1: in.key.slot = 1; break;
        case 2: in.key.generation = 2; break;
        case 3: in.key.incarnation = 8; break;
        case 4: in.key.incarnation += uint64_t(1) << 32; break;
        case 5: in.revision = 2; break;
        case 6: in.revision += uint64_t(1) << 32; break;
        case 7: in.originalMip = 1; break;
        }
    }
    std::vector<WholeMipResult> results;
    ASSERT_EQ(harness.sample(context, inputs, results), hipSuccess);
    const uint32_t wave = results[0].waveSize;
    ASSERT_TRUE(wave == 32 || wave == 64);
    std::vector<cv::RequestKey> requests;
    ASSERT_NO_FATAL_FAILURE(read(requests, 8 * (64 / wave)));
    for (uint32_t variant = 0; variant < 8; ++variant) {
        const auto& in = inputs[variant];
        const cv::RequestKey expected{in.key, in.revision, in.originalMip, 0};
        EXPECT_EQ(std::count(requests.begin(), requests.end(), expected), 64 / wave);
    }
}

TEST_F(WholeMipRequests, DivergentInactiveInvalidAndMixedSamplerLanesRemainIndependent) {
    std::vector<WholeMipInput> inputs(61);
    for (uint32_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = input(entries[(i / 2) % 2].texture.key, 1);
        if (i % 4 == 0) inputs[i].path = WholeMipPath::Inactive;
        if (i % 4 == 2) inputs[i].key.generation = 99;
    }
    std::vector<WholeMipResult> results;
    ASSERT_EQ(harness.sample(context, inputs, results), hipSuccess);
    std::vector<cv::RequestKey> requests;
    ASSERT_NO_FATAL_FAILURE(read(requests, 2 * ((61 + results[0].waveSize - 1) / results[0].waveSize)));
    for (size_t i = 0; i < inputs.size(); ++i) {
        EXPECT_EQ(results[i].decision.needsRequest(), i % 2 == 1);
        EXPECT_EQ(color(results[i].value), (Color{1, 0, 1, 1}));
    }
}

TEST_F(WholeMipRequests, BoundedQueueSaturatesAtExactCapacityAndPreservesGuards) {
    for (uint32_t capacity : {1u, 2u, 8u, 32u, 64u}) {
        for (uint32_t count : {capacity, capacity + 1, 128u}) {
            SCOPED_TRACE(testing::Message() << "capacity=" << capacity << " count=" << count);
            context.maxRequests = capacity;
            ASSERT_NO_FATAL_FAILURE(reset());
            std::vector<WholeMipInput> inputs(count);
            for (uint32_t lane = 0; lane < count; ++lane) {
                inputs[lane] = input({0, 1, 7}, 0, WholeMipPath::RecordLocalRequest);
                inputs[lane].revision = lane + 1;
            }
            std::vector<WholeMipResult> results;
            ASSERT_EQ(harness.sample(context, inputs, results), hipSuccess);
            std::vector<cv::RequestKey> requests;
            ASSERT_NO_FATAL_FAILURE(read(requests, std::min(count, capacity), count > capacity ? 1 : 0));
            std::set<uint64_t> revisions;
            for (const auto& request : requests) revisions.insert(request.revision);
            EXPECT_EQ(revisions.size(), requests.size());
            if (count > capacity) {
                ASSERT_EQ(harness.sample(context, inputs, results), hipSuccess);
                ASSERT_NO_FATAL_FAILURE(read(requests, capacity, 1));
            }
        }
    }
}

TEST_F(WholeMipRequests, GradientDemandUsesOriginalDimensionsIncludingThinAxes) {
    for (const auto shape : {std::pair<uint32_t, uint32_t>{19, 11}, {19, 1}, {1, 19}}) {
        entries[0].texture.mips = {shape.first, shape.second, 5};
        for (auto filter : {cv::FilterMode::Point, cv::FilterMode::Linear}) {
            entries[0].descriptor.mipFilter = filter;
            for (float lod : {0.f, .25f, .75f, 1.f, 1.25f, 2.75f, 4.f, 40.f}) {
                for (uint32_t axis = 0; axis < 2; ++axis) {
                    ASSERT_NO_FATAL_FAILURE(reset());
                    auto in = input(entries[0].texture.key, lod, WholeMipPath::Gradient);
                    if (axis == 0) in.ddx.x = std::exp2(lod) / shape.first;
                    else in.ddy.y = -std::exp2(lod) / shape.second;
                    std::vector<WholeMipResult> results;
                    ASSERT_EQ(harness.sample(context, {in}, results), hipSuccess);
                    const auto levels = cv::requiredLevels(lod, 5, filter).levels;
                    EXPECT_EQ(results[0].decision.validity, cv::SampleValidity::Missing);
                    EXPECT_EQ(results[0].decision.required.first, levels.first);
                    EXPECT_EQ(results[0].decision.required.last, levels.last);
                    std::vector<cv::RequestKey> requests;
                    ASSERT_NO_FATAL_FAILURE(read(requests, levels.last - levels.first + 1));
                }
            }
        }
    }
}

TEST_F(WholeMipDevice, StrictMissAndExplicitCoarsePreviewDemandTheSameOriginalDetail) {
    auto source = std::make_shared<TypedImageSource>(makeFilteringMipSource(19, 11));
    TextureDesc desc;
    desc.mipmapFilterMode = hipFilterModeLinear;
    wm::Texture texture(source, desc);
    const auto strict = texture.addSampler(desc);
    const auto preview = texture.addSampler(desc, cv::SamplingPolicy::AllowCoarsePreview);
    ASSERT_TRUE(strict.succeeded());
    ASSERT_TRUE(preview.succeeded());
    ASSERT_EQ(texture.resize(2), cv::Outcome::Success);
    wm::DeviceContext context;
    ASSERT_EQ(texture.prepare(harness.stream(), context), cv::Outcome::Success);
    std::vector<WholeMipResult> results;
    ASSERT_EQ(harness.sample(context, {input(strict.key, .25f), input(preview.key, .25f)}, results), hipSuccess);
    EXPECT_EQ(results[0].decision.validity, cv::SampleValidity::Missing);
    EXPECT_EQ(color(results[0].value), (Color{1, 0, 1, 1}));
    EXPECT_EQ(results[1].decision.validity, cv::SampleValidity::CoarsePreview);
    EXPECT_FALSE(results[1].decision.contributesStrictSample());
    expectPixel(results[1], oracle(*source, desc, input(preview.key, .25f), 2));
    for (const auto& result : results) {
        EXPECT_EQ(result.decision.required.first, 0u);
        EXPECT_EQ(result.decision.required.last, 1u);
    }
    uint32_t count = 0;
    ASSERT_EQ(hipMemcpy(&count, context.requestCount, sizeof(count), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(count, 4u);
    ASSERT_EQ(texture.processRequests(), cv::Outcome::Success);
    wm::Status status;
    ASSERT_EQ(texture.getStatus(status), cv::Outcome::Success);
    EXPECT_EQ(status.mips.firstResidentMip, 0u);
    // Integer retry isolates residency correctness from native linear qualification.
    ASSERT_NO_FATAL_FAILURE(sample(texture, {input(strict.key, 0), input(preview.key, 0)}, results));
    for (const auto& result : results) {
        EXPECT_EQ(result.decision.validity, cv::SampleValidity::Complete);
        expectPixel(result, oracle(*source, desc, input(strict.key, 0)));
    }
}

TEST_F(WholeMipDevice, NativePointGradientNpotMatchesWrapperAndAnalyticalPixels) {
    auto source = std::make_shared<TypedImageSource>(makeFilteringMipSource(19, 19, true));
    TextureDesc desc;
    desc.filterMode = desc.mipmapFilterMode = hipFilterModePoint;
    wm::Texture texture(source, desc);
    const auto key = texture.addSampler(desc);
    ASSERT_TRUE(key.succeeded());
    ASSERT_EQ(texture.resize(0), cv::Outcome::Success);
    auto wrapped = input(key.key, .5f + 1.f / 64, WholeMipPath::Gradient);
    wrapped.u = 1.125f;
    wrapped.v = -.125f;
    wrapped.ddx.x = wrapped.ddy.y = std::exp2(wrapped.lod) / 19.f;
    auto native = wrapped;
    native.path = WholeMipPath::NativeGradient;
    std::vector<WholeMipResult> results;
    ASSERT_NO_FATAL_FAILURE(sample(texture, {wrapped, native}, results));
    ASSERT_EQ(results.size(), 2u);
    const auto actual = color(results[0].value), direct = color(results[1].value);
    const auto expected = oracle(*source, desc, wrapped);
    for (size_t channel = 0; channel < 4; ++channel) {
        EXPECT_NEAR(actual[channel], direct[channel], PixelTolerance);
        EXPECT_NEAR(direct[channel], expected.value[channel], PixelTolerance)
            << "Native full-chain tex2DGrad, no suffix rebasing; channel=" << channel;
    }
}

using Modes = std::tuple<hipTextureFilterMode, hipTextureFilterMode>;
class WholeMipNumerical : public WholeMipDevice, public testing::WithParamInterface<Modes> {
protected:
    TextureDesc descriptor() const {
        TextureDesc desc;
        desc.filterMode = std::get<0>(GetParam());
        desc.mipmapFilterMode = std::get<1>(GetParam());
        return desc;
    }
    void compare(const TypedImageSource& source, const TextureDesc& desc,
                 const std::vector<WholeMipInput>& inputs, const std::vector<WholeMipResult>& actual,
                 const std::vector<WholeMipResult>& full) {
        ASSERT_EQ(actual.size(), inputs.size());
        ASSERT_EQ(full.size(), inputs.size());
        size_t incorrect = 0, different = 0;
        float maximum = 0, maximumDifference = 0;
        size_t firstFailure = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            EXPECT_TRUE(actual[i].decision.contributesStrictSample());
            EXPECT_TRUE(full[i].decision.contributesStrictSample());
            const auto expected = oracle(source, desc, inputs[i]);
            const auto a = color(actual[i].value), b = color(full[i].value);
            for (size_t c = 0; c < 4; ++c) {
                const float error = std::abs(a[c] - expected.value[c]);
                const float difference = std::abs(a[c] - b[c]);
                maximum = std::max(maximum, error);
                maximumDifference = std::max(maximumDifference, difference);
                if (!std::isfinite(a[c]) || error > expected.tolerance[c]) {
                    if (!incorrect) firstFailure = i;
                    ++incorrect;
                }
                // Same fixed contrast/128 bound, not a doubled full/suffix
                // allowance: two 8-bit quantizations were already budgeted.
                if (!std::isfinite(b[c]) || difference > expected.tolerance[c])
                    ++different;
                if (!std::isfinite(b[c]) || std::abs(b[c] - expected.value[c]) > expected.tolerance[c])
                    ++incorrect;
            }
        }
        EXPECT_EQ(incorrect, 0u) << "Strict analytic oracle; max error=" << maximum
            << " first failing sample=" << firstFailure << " lod=" << inputs[firstFailure].lod;
        EXPECT_EQ(different, 0u) << "Full-chain versus suffix max difference=" << maximumDifference;
        std::cout << "Whole-mip numerical " << source.info.width << "x" << source.info.height
                  << " samples=" << inputs.size() << " max analytical error=" << maximum
                  << " full/suffix difference=" << maximumDifference << '\n';
    }
    void numerical(bool gradients, bool patterned) {
        const std::array<std::pair<uint32_t, uint32_t>, 9> shapes{{
            {16, 16}, {19, 19}, {16, 8}, {19, 11}, {16, 1}, {1, 16}, {19, 1}, {1, 19}, {1, 1}}};
        for (const auto shape : shapes) {
            for (auto address : {hipAddressModeWrap, hipAddressModeClamp}) {
                SCOPED_TRACE(testing::Message() << "shape=" << shape.first << "x" << shape.second
                    << " address=" << int(address) << " gradient=" << gradients << " patterned=" << patterned);
                auto source = std::make_shared<TypedImageSource>(
                    makeFilteringMipSource(shape.first, shape.second, patterned));
                auto reference = std::make_shared<TypedImageSource>(*source);
                auto desc = descriptor();
                desc.addressMode[0] = desc.addressMode[1] = address;
                wm::Texture texture(source, desc), full(reference, desc);
                const auto key = texture.addSampler(desc), referenceKey = full.addSampler(desc);
                ASSERT_TRUE(key.succeeded());
                ASSERT_TRUE(referenceKey.succeeded());
                ASSERT_EQ(full.resize(0), cv::Outcome::Success);
                const uint32_t last = source->info.numMipLevels - 1;
                const std::vector<uint32_t> phases{0, std::min(2u, last), 0, std::min(3u, last), 1u > last ? 0u : 1u, 0};
                uint64_t expectedUploaded = 0;
                for (uint32_t first : phases) {
                    SCOPED_TRACE(testing::Message() << "first=" << first);
                    wm::Status before;
                    ASSERT_EQ(texture.getStatus(before), cv::Outcome::Success);
                    const size_t priorReads = source->readLevels.size();
                    ASSERT_EQ(texture.resize(first), cv::Outcome::Success);
                    const bool changed = before.mips.firstResidentMip != first;
                    uint64_t payload = 0;
                    for (uint32_t level = first; level <= last; ++level) payload += source->mipPixels[level].size();
                    if (changed) expectedUploaded += payload;
                    wm::Status after;
                    ASSERT_EQ(texture.getStatus(after), cv::Outcome::Success);
                    EXPECT_EQ(after.uploadedBytes, expectedUploaded);
                    for (size_t i = priorReads; i < source->readLevels.size(); ++i)
                        EXPECT_GE(source->readLevels[i], first);
                    ASSERT_NO_FATAL_FAILURE(storage(texture, *source, first, {desc}));
                    std::vector<float> lods{float(first), float(last), float(last + 8)};
                    if (first == 0) lods.push_back(-4);
                    for (uint32_t level = first; level < last; ++level)
                        for (float fraction : {.25f, .5f - 1.f/64, .5f + 1.f/64, .75f})
                            lods.push_back(float(level) + fraction);
                    for (uint32_t axis = 0; axis < (gradients ? 4u : 1u); ++axis) {
                        std::vector<WholeMipInput> inputs;
                        for (float lod : lods) {
                            for (const auto uv : {std::pair<float, float>{0, 0}, {1, 1},
                                                {-.125f, .625f}, {1.125f, -.125f}, {.375f, .5625f}}) {
                                auto in = input(key.key, lod, gradients ? WholeMipPath::Gradient : WholeMipPath::Lod);
                                in.u = uv.first;
                                in.v = uv.second;
                                if (gradients) {
                                    const float extent = std::exp2(lod);
                                    if (axis == 0 || axis == 2) {
                                        in.ddx.x = extent / shape.first;
                                        in.ddy.y = extent / shape.second;
                                        if (axis == 2) {
                                            in.ddx.x = -in.ddx.x;
                                            in.ddy.y = -in.ddy.y;
                                        }
                                    } else {
                                        in.ddx.y = extent / shape.second;
                                        in.ddy.x = extent / shape.first;
                                        if (axis == 3) {
                                            in.ddx.y = -in.ddx.y;
                                            in.ddy.x = -in.ddy.x;
                                        }
                                    }
                                }
                                inputs.push_back(in);
                            }
                        }
                        std::vector<WholeMipResult> actual, baseline;
                        ASSERT_NO_FATAL_FAILURE(sample(texture, inputs, actual));
                        auto baselineInputs = inputs;
                        for (auto& in : baselineInputs) in.key = referenceKey.key;
                        ASSERT_NO_FATAL_FAILURE(sample(full, baselineInputs, baseline));
                        compare(*source, desc, inputs, actual, baseline);
                    }
                }
            }
        }
    }
};

TEST_P(WholeMipNumerical, AuthoredColorsExplicitFullCoarseGrowTrimRegrow) {
    ASSERT_NO_FATAL_FAILURE(numerical(false, false));
}
TEST_P(WholeMipNumerical, NonconstantExplicitFullCoarseGrowTrimRegrow) {
    ASSERT_NO_FATAL_FAILURE(numerical(false, true));
}
TEST_P(WholeMipNumerical, AuthoredGradientOriginalSpaceAcrossAllShapesAndAxes) {
    ASSERT_NO_FATAL_FAILURE(numerical(true, false));
}
TEST_P(WholeMipNumerical, NonconstantGradientOriginalSpaceAcrossAllShapesAndAxes) {
    ASSERT_NO_FATAL_FAILURE(numerical(true, true));
}

TEST_P(WholeMipNumerical, SharedVariantsReversedRegistrationRequestOrderAndReload) {
    for (bool reverseRegistration : {false, true}) {
        for (bool reverseRequests : {false, true}) {
            auto source = std::make_shared<TypedImageSource>(makeFilteringMipSource(19, 11, true));
            TextureDesc first = descriptor(), second = first;
            second.filterMode = first.filterMode == hipFilterModePoint ? hipFilterModeLinear : hipFilterModePoint;
            second.mipmapFilterMode = first.mipmapFilterMode == hipFilterModePoint ? hipFilterModeLinear : hipFilterModePoint;
            second.addressMode[0] = second.addressMode[1] = hipAddressModeClamp;
            std::vector<TextureDesc> descriptors = reverseRegistration
                ? std::vector<TextureDesc>{second, first} : std::vector<TextureDesc>{first, second};
            wm::Texture texture(source, first);
            const auto a = texture.addSampler(descriptors[0]), b = texture.addSampler(descriptors[1]);
            ASSERT_TRUE(a.succeeded());
            ASSERT_TRUE(b.succeeded());
            EXPECT_FALSE(a.key == b.key);
            EXPECT_TRUE(texture.addSampler(descriptors[0]).key == a.key);
            for (uint32_t cycle = 0; cycle < 2; ++cycle) {
                for (uint32_t requestedFirst : {2u, 0u, 3u, 1u}) {
                    ASSERT_EQ(texture.unload(), cv::Outcome::Success);
                    wm::DeviceContext context;
                    ASSERT_EQ(texture.prepare(harness.stream(), context), cv::Outcome::Success);
                    std::vector<WholeMipInput> requests{input(a.key, float(requestedFirst)),
                                                       input(b.key, float(requestedFirst))};
                    if (reverseRequests) std::reverse(requests.begin(), requests.end());
                    std::vector<WholeMipResult> missing;
                    ASSERT_EQ(harness.sample(context, requests, missing), hipSuccess);
                    for (const auto& result : missing) EXPECT_TRUE(result.decision.needsRequest());
                    ASSERT_EQ(texture.processRequests(), cv::Outcome::Success);
                    ASSERT_NO_FATAL_FAILURE(storage(texture, *source, requestedFirst, descriptors));
                    std::vector<WholeMipInput> samples;
                    for (const auto key : {a.key, b.key})
                        for (float lod : {float(requestedFirst), float(requestedFirst) + .25f, 4.f})
                            for (float u : {-.125f, .375f, 1.125f}) {
                                auto in = input(key, lod);
                                in.u = u;
                                samples.push_back(in);
                            }
                    std::vector<WholeMipResult> results;
                    ASSERT_NO_FATAL_FAILURE(sample(texture, samples, results));
                    for (size_t i = 0; i < samples.size(); ++i) {
                        SCOPED_TRACE(testing::Message() << "cycle=" << cycle << " requested=" << requestedFirst
                            << " reverse registrations=" << reverseRegistration << " requests=" << reverseRequests
                            << " sample=" << i);
                        const auto& desc = samples[i].key == a.key ? descriptors[0] : descriptors[1];
                        EXPECT_TRUE(results[i].decision.contributesStrictSample());
                        expectPixel(results[i], oracle(*source, desc, samples[i]));
                    }
                }
            }
        }
    }
}

TEST_P(WholeMipNumerical, ExactPointHalfTiesRequireBothLevelsWithoutBlending) {
    auto source = std::make_shared<TypedImageSource>(makeFilteringMipSource());
    const auto desc = descriptor();
    wm::Texture texture(source, desc);
    const auto registration = texture.addSampler(desc);
    ASSERT_TRUE(registration.succeeded());
    ASSERT_EQ(texture.resize(1), cv::Outcome::Success);
    std::vector<WholeMipResult> results;
    wm::DeviceContext context;
    ASSERT_EQ(texture.prepare(harness.stream(), context), cv::Outcome::Success);
    ASSERT_EQ(harness.sample(context, {input(registration.key, .5f)}, results), hipSuccess);
    EXPECT_EQ(results[0].decision.validity, cv::SampleValidity::Missing);
    EXPECT_EQ(results[0].decision.required.first, 0u);
    EXPECT_EQ(results[0].decision.required.last, 1u);
    ASSERT_EQ(texture.processRequests(), cv::Outcome::Success);
    ASSERT_NO_FATAL_FAILURE(sample(texture, {input(registration.key, .5f)}, results));
    if (desc.mipmapFilterMode == hipFilterModePoint) {
        const auto actual = color(results[0].value);
        EXPECT_TRUE(actual == filteringMipColors[0] || actual == filteringMipColors[1]);
    } else {
        expectPixel(results[0], oracle(*source, desc, input(registration.key, .5f)));
    }
}

TEST_P(WholeMipNumerical, PolicyLimitedAndUnnormalizedSingleLevelRetainOriginalClamps) {
    for (uint32_t limit : {1u, 3u}) {
        for (bool normalized : {false, true}) {
            if (!normalized && limit != 1) continue; // Multilevel unnormalized is explicitly unsupported.
            auto source = std::make_shared<TypedImageSource>(makeFilteringMipSource(19, 11, true));
            auto desc = descriptor();
            desc.maxMipLevel = limit;
            desc.normalizedCoords = normalized;
            desc.addressMode[0] = desc.addressMode[1] = hipAddressModeClamp;
            wm::Texture texture(source, desc);
            const auto registration = texture.addSampler(desc);
            ASSERT_TRUE(registration.succeeded());
            for (uint32_t first : {0u, limit - 1, 0u}) {
                ASSERT_EQ(texture.resize(first), cv::Outcome::Success);
                ASSERT_NO_FATAL_FAILURE(storage(texture, *source, first, {desc}));
                std::vector<WholeMipInput> inputs;
                for (auto path : {WholeMipPath::Lod, WholeMipPath::Gradient}) {
                    for (float lod : {float(first), 17.f}) {
                        auto in = input(registration.key, lod, path);
                        in.u = normalized ? .375f : 3.125f;
                        in.v = normalized ? .625f : 2.75f;
                        in.ddx.x = std::exp2(lod) / (normalized ? source->info.width : 1.f);
                        in.ddy.y = std::exp2(lod) / (normalized ? source->info.height : 1.f);
                        inputs.push_back(in);
                    }
                }
                std::vector<WholeMipResult> results;
                ASSERT_NO_FATAL_FAILURE(sample(texture, inputs, results));
                for (size_t i = 0; i < inputs.size(); ++i) {
                    EXPECT_TRUE(results[i].decision.contributesStrictSample());
                    expectPixel(results[i], oracle(*source, desc, inputs[i]));
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SpatialAndMip, WholeMipNumerical,
    testing::Combine(testing::Values(hipFilterModePoint, hipFilterModeLinear),
                     testing::Values(hipFilterModePoint, hipFilterModeLinear)),
    [](const testing::TestParamInfo<Modes>& info) {
        return std::string(std::get<0>(info.param) == hipFilterModePoint ? "SpatialPoint" : "SpatialLinear") +
            (std::get<1>(info.param) == hipFilterModePoint ? "MipPoint" : "MipLinear");
    });

} // namespace
} }
