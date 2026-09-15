// SPDX-License-Identifier: MIT
#include "TestUtils.h"
#include "TextureSamplingHarness.h"
#include <DemandLoading/Internal/HipCalls.h>
#include <ImageSource/ImageSource.h>
#include <ImageSource/TextureInfo.h>
#include <array>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <thread>

namespace hip_demand {
namespace test {
namespace {
using internal::HipOperation;
using internal::HipFaultState;

class RegistrationSource : public ImageSource {
public:
    RegistrationSource() {
        info.width = info.height = 2;
        info.numChannels = 4;
        info.numMipLevels = 1;
        info.isValid = true;
        pixels.fill(63);
    }
    void open(TextureInfo* output) override {
        if (throwOpen) throw std::runtime_error("Injected source open failure");
        if (throwAllocation) throw std::bad_alloc();
        if (throwNonstandard) throw 7;
        opened = !failOpen;
        if (output) *output = info;
    }
    void close() override { opened = false; }
    bool isOpen() const override { return opened; }
    const TextureInfo& getInfo() const override { return info; }
    bool readMipLevel(char* dest, unsigned int mip, unsigned int width,
                      unsigned int height, hipStream_t = nullptr) override {
        ++reads;
        if (failRead || mip != 0 || width != 2 || height != 2) return false;
        std::memcpy(dest, pixels.data(), pixels.size());
        return true;
    }
    bool readBaseColor(float4&) override { return false; }
    unsigned long long getNumBytesRead() const override { return reads * pixels.size(); }
    double getTotalReadTime() const override { return 0; }
    unsigned long long getHash(hipStream_t = nullptr) const override {
        if (throwHash) throw std::runtime_error("Injected source hash failure");
        return hash;
    }
    TextureInfo info;
    std::array<unsigned char, 16> pixels;
    unsigned long long hash = 0;
    bool opened = false, failOpen = false, failRead = false;
    bool throwOpen = false, throwHash = false, throwAllocation = false, throwNonstandard = false;
    unsigned int reads = 0;
};

struct ScopedFaults {
    std::shared_ptr<HipFaultState> state = std::make_shared<HipFaultState>();
    ScopedFaults() { internal::setHipFaultState(state); }
    ~ScopedFaults() { internal::setHipFaultState(nullptr); }
};

LoaderOptions smallOptions(size_t capacity = 4) {
    LoaderOptions options;
    options.maxTextures = capacity;
    options.maxRequestsPerLaunch = 8;
    options.maxThreads = 2;
    options.enableEviction = false;
    return options;
}

TextureDesc singleLevel() {
    TextureDesc desc;
    desc.generateMipmaps = false;
    desc.filterMode = hipFilterModePoint;
    return desc;
}

void expectFailure(const TextureHandle& handle, LoaderError error) {
    EXPECT_FALSE(handle.valid);
    EXPECT_EQ(handle.id, InvalidTextureId);
    EXPECT_EQ(handle.error, error);
    EXPECT_EQ(handle.width, 0);
    EXPECT_EQ(handle.height, 0);
    EXPECT_EQ(handle.channels, 0);
}

void request(DemandTextureLoader& loader, const std::vector<uint32_t>& ids, size_t expected) {
    loader.launchPrepare();
    const auto context = loader.getDeviceContext();
    const uint32_t count = static_cast<uint32_t>(ids.size());
    ASSERT_LE(count, context.maxRequests);
    ASSERT_EQ(hipMemcpy(context.requests, ids.data(), ids.size() * sizeof(uint32_t), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(context.requestCount, &count, sizeof(count), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(loader.processRequests(nullptr, context), expected);
    loader.launchPrepare();
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

hipTextureObject_t textureObject(DemandTextureLoader& loader, uint32_t id) {
    hipTextureObject_t object = 0;
    EXPECT_EQ(hipMemcpy(&object, loader.getDeviceContext().textures + id, sizeof(object), hipMemcpyDeviceToHost), hipSuccess);
    return object;
}

TEST(RegistrationResult, DefaultAndLegacyLayout) {
    expectFailure(TextureHandle{}, LoaderError::InvalidTextureId);
    static_assert(sizeof(TextureHandle) == 24);
    static_assert(offsetof(TextureHandle, id) == 0);
    static_assert(offsetof(TextureHandle, error) == 20);
    EXPECT_EQ(InvalidTextureId, UINT32_MAX);
}

TEST(RegistrationResult, FaultRulesValidateAndCountEveryInvocation) {
    HipFaultState state;
    EXPECT_THROW(state.fail(HipOperation::Upload, hipSuccess), std::invalid_argument);
    EXPECT_THROW(state.fail(HipOperation::Upload, hipErrorUnknown, 0), std::invalid_argument);
    state.fail(HipOperation::Upload, hipErrorOutOfMemory, 2);
    state.fail(HipOperation::Upload, hipErrorInvalidValue, 3);
    EXPECT_EQ(state.before(HipOperation::Upload), hipSuccess);
    EXPECT_EQ(state.before(HipOperation::Upload), hipErrorOutOfMemory);
    EXPECT_EQ(state.before(HipOperation::Upload), hipErrorInvalidValue);
    EXPECT_EQ(state.before(HipOperation::Upload), hipSuccess);
}

class RegistrationCapacity : public HipTestFixture,
                             public testing::WithParamInterface<std::tuple<int, size_t>> {};

TEST_P(RegistrationCapacity, ExhaustionAndDeduplicationPreserveZero) {
    const auto [kind, capacity] = GetParam();
    DemandTextureLoader loader(smallOptions(capacity));
    ASSERT_EQ(loader.getLastError(), LoaderError::Success);
    std::vector<std::shared_ptr<RegistrationSource>> sources;
    std::array<unsigned char, 4> pixel{63, 127, 191, 255};
    for (size_t i = 0; i < capacity; ++i) {
        SCOPED_TRACE(i + 1);
        TextureHandle handle;
        if (kind == 0) {
            handle = loader.createTexture("registration-missing-" + std::to_string(i), singleLevel());
        } else if (kind == 1) {
            sources.push_back(std::make_shared<RegistrationSource>());
            handle = loader.createTexture(sources.back(), singleLevel());
        } else {
            handle = loader.createTextureFromMemory(pixel.data(), 1, 1, 4, singleLevel());
        }
        ASSERT_TRUE(handle.valid);
        EXPECT_EQ(handle.error, LoaderError::Success);
        EXPECT_EQ(handle.id, i);
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
        if (kind == 0)
            expectFailure(loader.createTexture("one-too-many"), LoaderError::MaxTexturesExceeded);
        else if (kind == 1)
            expectFailure(loader.createTexture(std::make_shared<RegistrationSource>()), LoaderError::MaxTexturesExceeded);
        else
            expectFailure(loader.createTextureFromMemory(pixel.data(), 1, 1, 4), LoaderError::MaxTexturesExceeded);
    }
    if (kind != 2) {
        const auto reused = kind == 0 ? loader.createTexture("registration-missing-0", singleLevel()) :
                                       loader.createTexture(sources.front(), singleLevel());
        EXPECT_TRUE(reused.valid);
        EXPECT_EQ(reused.id, 0u);
        EXPECT_EQ(reused.error, LoaderError::Success);
        EXPECT_EQ(loader.getLastError(), LoaderError::MaxTexturesExceeded);
    }
    EXPECT_EQ(loader.getResidentTextureCount(), 0u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
}

INSTANTIATE_TEST_SUITE_P(AllOverloads, RegistrationCapacity,
    testing::Combine(testing::Values(0, 1, 2), testing::Values(size_t{1}, size_t{7}, size_t{4096})));

class RegistrationErrors : public HipTestFixture {};

TEST_F(RegistrationErrors, InvalidOptionsAreRejectedBeforeHipAndStayFailed) {
    ScopedFaults faults;
    for (size_t value : {size_t{0}, size_t{UINT32_MAX} + 1, SIZE_MAX}) {
        for (bool textures : {false, true}) {
            auto options = smallOptions();
            (textures ? options.maxTextures : options.maxRequestsPerLaunch) = value;
            DemandTextureLoader loader(options);
            EXPECT_EQ(loader.getLastError(), LoaderError::InvalidParameter);
            const unsigned char pixel = 1;
            expectFailure(loader.createTexture("lazy.png"), LoaderError::InvalidParameter);
            expectFailure(loader.createTexture(std::make_shared<RegistrationSource>()), LoaderError::InvalidParameter);
            expectFailure(loader.createTextureFromMemory(&pixel, 1, 1, 1), LoaderError::InvalidParameter);
            const auto context = loader.getDeviceContext();
            EXPECT_EQ(context.maxTextures, 0u);
            EXPECT_EQ(context.textures, nullptr);
            loader.launchPrepare();
            EXPECT_EQ(loader.processRequests(nullptr, context), 0u);
            loader.processRequestsAsync(nullptr, context).wait();
        }
    }
    EXPECT_TRUE(faults.state->records().empty());
}

TEST_F(RegistrationErrors, InitializationOperationalErrorIsNotCapacityOrOutOfMemory) {
    ScopedFaults faults;
    faults.state->fail(HipOperation::GetDevice, hipErrorInvalidDevice);
    DemandTextureLoader loader(smallOptions());
    EXPECT_EQ(loader.getLastError(), LoaderError::HipError);
    expectFailure(loader.createTexture("lazy"), LoaderError::HipError);
    ASSERT_EQ(faults.state->records().size(), 1u);
    EXPECT_EQ(faults.state->records().front().error, hipErrorInvalidDevice);
}

TEST_F(RegistrationErrors, InvalidDataMetadataAndDescriptorsDoNotConsumeCapacity) {
    DemandTextureLoader loader(smallOptions(1));
    unsigned char pixel[4]{};
    expectFailure(loader.createTexture(std::shared_ptr<ImageSource>{}), LoaderError::InvalidParameter);
    expectFailure(loader.createTexture(""), LoaderError::InvalidParameter);
    expectFailure(loader.createTexture(std::string("bad\0name", 8)), LoaderError::InvalidParameter);
    expectFailure(loader.createTextureFromMemory(nullptr, 1, 1, 4), LoaderError::InvalidParameter);
    for (const auto& dimensions : {std::array<int, 3>{0, 1, 1}, {-1, 1, 1}, {1, 0, 1},
                                  {1, -1, 1}, {1, 1, 0}, {1, 1, 5}, {INT32_MAX, INT32_MAX, 4}}) {
        expectFailure(loader.createTextureFromMemory(pixel, dimensions[0], dimensions[1], dimensions[2]),
                      LoaderError::InvalidParameter);
    }
    for (int invalid = 0; invalid < 7; ++invalid) {
        auto source = std::make_shared<RegistrationSource>();
        switch (invalid) {
        case 0: source->info.width = 0; break;
        case 1: source->info.height = UINT32_MAX; break;
        case 2: source->info.numChannels = 5; break;
        case 3: source->info.isValid = false; break;
        case 4: source->info.format = static_cast<hipArray_Format>(-1); break;
        case 5: source->info.width = source->info.height = INT32_MAX; break;
        case 6:
            source->info.width = source->info.height = INT32_MAX;
            source->info.format = HIP_AD_FORMAT_FLOAT;
            break;
        }
        expectFailure(loader.createTexture(source), LoaderError::InvalidParameter);
    }
    for (int invalid = 0; invalid < 4; ++invalid) {
        TextureDesc desc;
        if (invalid == 0) desc.filterMode = static_cast<hipTextureFilterMode>(-1);
        if (invalid == 1) desc.mipmapFilterMode = static_cast<hipTextureFilterMode>(-1);
        if (invalid == 2) desc.addressMode[1] = static_cast<hipTextureAddressMode>(-1);
        if (invalid == 3) desc.evictionPriority = static_cast<EvictionPriority>(-1);
        expectFailure(loader.createTexture("lazy", desc), LoaderError::InvalidParameter);
        expectFailure(loader.createTexture(std::make_shared<RegistrationSource>(), desc), LoaderError::InvalidParameter);
        expectFailure(loader.createTextureFromMemory(pixel, 1, 1, 4, desc), LoaderError::InvalidParameter);
    }
    const auto zero = loader.createTextureFromMemory(pixel, 1, 1, 4);
    ASSERT_TRUE(zero.valid);
    EXPECT_EQ(zero.id, 0u);
}

TEST_F(RegistrationErrors, SourceExceptionsAndClosedSourcesRollBack) {
    DemandTextureLoader loader(smallOptions(1));
    auto source = std::make_shared<RegistrationSource>();
    source->hash = 123;
    source->throwHash = true;
    expectFailure(loader.createTexture(source), LoaderError::ImageLoadFailed);
    source->throwHash = false;
    source->throwOpen = true;
    expectFailure(loader.createTexture(source), LoaderError::ImageLoadFailed);
    source->throwOpen = false;
    source->throwAllocation = true;
    expectFailure(loader.createTexture(source), LoaderError::OutOfMemory);
    source->throwAllocation = false;
    source->throwNonstandard = true;
    EXPECT_THROW(loader.createTexture(source), int);
    source->throwNonstandard = false;
    source->failOpen = true;
    expectFailure(loader.createTexture(source), LoaderError::ImageLoadFailed);
    source->failOpen = false;
    const auto result = loader.createTexture(source);
    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.id, 0u);
    EXPECT_EQ(source->reads, 0u);
}

TEST_F(RegistrationErrors, LazyFilenameFailureIsNotRegistrationFailure) {
    DemandTextureLoader loader(smallOptions(1));
    const auto handle = loader.createTexture("registration-deliberately-missing-file.exr");
    ASSERT_TRUE(handle.valid);
    EXPECT_EQ(handle.error, LoaderError::Success);
    EXPECT_EQ(handle.id, 0u);
    request(loader, {handle.id}, 0);
    EXPECT_EQ(loader.getLastError(), LoaderError::ImageLoadFailed);
    EXPECT_EQ(loader.getResidentTextureCount(), 0u);
    const auto reused = loader.createTexture("registration-deliberately-missing-file.exr");
    EXPECT_TRUE(reused.valid);
    EXPECT_EQ(reused.id, handle.id);
    EXPECT_EQ(reused.error, LoaderError::Success);
}

TEST_F(RegistrationErrors, LaterSourceReadFailurePreservesRegistrationAndRetry) {
    DemandTextureLoader loader(smallOptions(1));
    auto source = std::make_shared<RegistrationSource>();
    source->failRead = true;
    const auto handle = loader.createTexture(source, singleLevel());
    ASSERT_TRUE(handle.valid);
    EXPECT_EQ(source->reads, 0u);
    request(loader, {handle.id}, 0);
    EXPECT_EQ(loader.getLastError(), LoaderError::ImageLoadFailed);
    EXPECT_EQ(loader.getResidentTextureCount(), 0u);
    source->failRead = false;
    const auto reused = loader.createTexture(source, singleLevel());
    EXPECT_EQ(reused.id, handle.id);
    EXPECT_EQ(reused.error, LoaderError::Success);
    request(loader, {handle.id}, 1);
}

TEST_F(RegistrationErrors, PointerAndContentNamespacesAndUnretainedAliases) {
    alignas(RegistrationSource) unsigned char storage[sizeof(RegistrationSource)];
    DemandTextureLoader loader(smallOptions(4));
    const std::string filename = "registration-hash-namespace";
    const auto file = loader.createTexture(filename);
    auto source = std::make_shared<RegistrationSource>();
    source->hash = std::hash<std::string>{}(filename);
    const auto image = loader.createTexture(source);
    ASSERT_TRUE(image.valid);
    EXPECT_NE(image.id, file.id);
    auto alias = std::make_shared<RegistrationSource>();
    alias->hash = source->hash;
    EXPECT_EQ(loader.createTexture(alias).id, image.id);
    std::weak_ptr<RegistrationSource> weak = alias;
    alias.reset();
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(loader.createTexture(filename).id, file.id);
    EXPECT_EQ(loader.createTexture(source).id, image.id);

    const auto recycleSource = [&] {
        return std::shared_ptr<RegistrationSource>(new (storage) RegistrationSource,
            [](RegistrationSource* p) { p->~RegistrationSource(); });
    };
    auto reusedAddress = recycleSource();
    reusedAddress->hash = source->hash;
    EXPECT_EQ(loader.createTexture(reusedAddress).id, image.id);
    reusedAddress.reset();
    reusedAddress = recycleSource();
    const auto distinct = loader.createTexture(reusedAddress);
    EXPECT_TRUE(distinct.valid);
    EXPECT_NE(distinct.id, image.id);
}

TEST_F(RegistrationErrors, InvalidIdOperationsPreserveResidentAndColdZero) {
    DemandTextureLoader loader(smallOptions(1));
    auto source = std::make_shared<RegistrationSource>();
    const auto zero = loader.createTexture(source, singleLevel());
    ASSERT_TRUE(zero.valid);
    const auto failed = loader.createTexture(std::make_shared<RegistrationSource>());
    expectFailure(failed, LoaderError::MaxTexturesExceeded);
    for (bool resident : {false, true}) {
        if (resident) request(loader, {zero.id}, 1);
        const auto object = textureObject(loader, 0);
        const auto memory = loader.getTotalTextureMemory();
        for (uint32_t id : {failed.id, 1u, 32u, UINT32_MAX - 1}) {
            loader.unloadTexture(id);
            EXPECT_EQ(loader.getLastError(), LoaderError::InvalidTextureId);
            loader.updateEvictionPriority(id, EvictionPriority::KeepResident);
            EXPECT_EQ(loader.getLastError(), LoaderError::InvalidTextureId);
        }
        request(loader, {failed.id, 1u, UINT32_MAX - 1}, 0);
        EXPECT_EQ(textureObject(loader, 0), object);
        EXPECT_EQ(loader.getResidentTextureCount(), resident ? 1u : 0u);
        EXPECT_EQ(loader.getTotalTextureMemory(), memory);
        EXPECT_EQ(source->reads, resident ? 1u : 0u);
    }
}

TEST_F(RegistrationErrors, FailedHandlesNeverSampleOrRequestTextureZeroOnDevice) {
    DemandTextureLoader loader(smallOptions(1));
    const std::array<unsigned char, 4> pixel{63, 127, 191, 255};
    const auto zero = loader.createTextureFromMemory(pixel.data(), 1, 1, 4, singleLevel());
    ASSERT_TRUE(zero.valid);
    ASSERT_EQ(zero.id, 0u);
    const std::array<TextureHandle, 3> failed{
        loader.createTexture("capacity-failed-file"),
        loader.createTexture(std::make_shared<RegistrationSource>()),
        loader.createTextureFromMemory(pixel.data(), 1, 1, 4)
    };
    TextureSamplingHarness harness;
    ASSERT_EQ(harness.open(samplingModulePath()), hipSuccess);
    for (bool resident : {false, true}) {
        if (resident) request(loader, {zero.id}, 1);
        loader.launchPrepare();
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const auto context = loader.getDeviceContext();
        const std::array<uint32_t, 8> canary{17, 18, 19, 20, 21, 22, 23, 24};
        ASSERT_EQ(hipMemcpy(context.requests, canary.data(), sizeof(canary), hipMemcpyHostToDevice), hipSuccess);
        std::vector<SamplingInput> inputs;
        for (const auto& handle : failed) {
            expectFailure(handle, LoaderError::MaxTexturesExceeded);
            for (SamplingPath path : {SamplingPath::Implicit, SamplingPath::Lod,
                                      SamplingPath::Gradient, SamplingPath::RecordRequest}) {
                SamplingInput input;
                input.textureId = handle.id;
                input.path = path;
                input.lod = 2;
                input.ddx = {.5f, 0};
                input.ddy = {0, .5f};
                input.defaultColor = {-2, .125f, 9, .5f};
                inputs.push_back(input);
            }
        }
        std::vector<SamplingResult> results;
        ASSERT_EQ(harness.sample(context, inputs, results), hipSuccess);
        ASSERT_EQ(results.size(), inputs.size());
        for (const auto& result : results) {
            EXPECT_EQ(result.resident, 0u);
            EXPECT_FLOAT_EQ(result.value.x, -2);
            EXPECT_FLOAT_EQ(result.value.y, .125f);
            EXPECT_FLOAT_EQ(result.value.z, 9);
            EXPECT_FLOAT_EQ(result.value.w, .5f);
        }
        uint32_t count = UINT32_MAX, overflow = UINT32_MAX;
        std::array<uint32_t, 8> requests{};
        ASSERT_EQ(hipMemcpy(&count, context.requestCount, sizeof(count), hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipMemcpy(&overflow, context.requestOverflow, sizeof(overflow), hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipMemcpy(requests.data(), context.requests, sizeof(requests), hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(count, 0u);
        EXPECT_EQ(overflow, 0u);
        EXPECT_EQ(requests, canary);
        EXPECT_EQ(loader.getResidentTextureCount(), resident ? 1u : 0u);
        if (resident) {
            inputs.clear();
            for (SamplingPath path : {SamplingPath::Implicit, SamplingPath::Lod, SamplingPath::Gradient}) {
                SamplingInput input;
                input.textureId = zero.id;
                input.path = path;
                inputs.push_back(input);
            }
            ASSERT_EQ(harness.sample(context, inputs, results), hipSuccess);
            for (const auto& result : results) {
                EXPECT_EQ(result.resident, 1u);
                EXPECT_NEAR(result.value.x, 63.f / 255, 1e-6f);
                EXPECT_NEAR(result.value.y, 127.f / 255, 1e-6f);
                EXPECT_NEAR(result.value.z, 191.f / 255, 1e-6f);
                EXPECT_NEAR(result.value.w, 1, 1e-6f);
            }
        }
    }
}

TEST_F(RegistrationErrors, ConcurrentRegistrationDeduplicatesAndExhaustsExactly) {
    DemandTextureLoader loader(smallOptions(7));
    auto source = std::make_shared<RegistrationSource>();
    std::array<TextureHandle, 16> results;
    std::vector<std::thread> workers;
    for (size_t i = 0; i < results.size(); ++i)
        workers.emplace_back([&, i] { results[i] = loader.createTexture(source, singleLevel()); });
    for (auto& worker : workers) worker.join();
    for (const auto& handle : results) {
        EXPECT_TRUE(handle.valid);
        EXPECT_EQ(handle.id, 0u);
    }
    workers.clear();
    const unsigned char pixel = 7;
    for (size_t i = 0; i < results.size(); ++i)
        workers.emplace_back([&, i] { results[i] = loader.createTextureFromMemory(&pixel, 1, 1, 1); });
    for (auto& worker : workers) worker.join();
    size_t successes = 0;
    std::set<uint32_t> ids;
    for (const auto& handle : results) {
        if (handle.valid) { ++successes; ids.insert(handle.id); }
        else expectFailure(handle, LoaderError::MaxTexturesExceeded);
    }
    EXPECT_EQ(successes, 6u);
    EXPECT_EQ(ids.size(), 6u);
}

class RegistrationInitialization : public HipTestFixture,
    public testing::WithParamInterface<std::tuple<HipOperation, size_t>> {};

TEST_P(RegistrationInitialization, PartialFailureNeverPublishesAndCanBeDestroyed) {
    ScopedFaults faults;
    const auto [operation, invocation] = GetParam();
    faults.state->fail(operation, hipErrorOutOfMemory, invocation);
    DemandTextureLoader loader(smallOptions());
    EXPECT_EQ(loader.getLastError(), LoaderError::OutOfMemory);
    expectFailure(loader.createTexture("lazy"), LoaderError::OutOfMemory);
    expectFailure(loader.createTexture(std::make_shared<RegistrationSource>()), LoaderError::OutOfMemory);
    unsigned char pixel = 1;
    expectFailure(loader.createTextureFromMemory(&pixel, 1, 1, 1), LoaderError::OutOfMemory);
    loader.unloadTexture(InvalidTextureId);
    expectFailure(loader.createTexture("still-failed"), LoaderError::OutOfMemory);
    EXPECT_EQ(loader.getDeviceContext().textures, nullptr);
    loader.launchPrepare();
    EXPECT_EQ(loader.processRequests(nullptr, loader.getDeviceContext()), 0u);
    loader.processRequestsAsync(nullptr, loader.getDeviceContext()).wait();
}

INSTANTIATE_TEST_SUITE_P(Faults, RegistrationInitialization, testing::Values(
    std::make_tuple(HipOperation::GetDevice, 1u),
    std::make_tuple(HipOperation::DeviceAllocation, 1u),
    std::make_tuple(HipOperation::DeviceAllocation, 2u),
    std::make_tuple(HipOperation::DeviceAllocation, 3u),
    std::make_tuple(HipOperation::DeviceAllocation, 4u),
    std::make_tuple(HipOperation::HostAllocation, 1u),
    std::make_tuple(HipOperation::HostAllocation, 2u),
    std::make_tuple(HipOperation::HostAllocation, 3u),
    std::make_tuple(HipOperation::HostAllocation, 4u),
    std::make_tuple(HipOperation::Initialize, 1u),
    std::make_tuple(HipOperation::Initialize, 2u),
    std::make_tuple(HipOperation::Initialize, 3u)));

TEST_F(RegistrationErrors, RegistrationAllocationFailuresLeaveMapsAndIdsUnchanged) {
    for (auto operation : {HipOperation::CacheData, HipOperation::FilenameMap,
                           HipOperation::SourceMap, HipOperation::ContentMap}) {
        ScopedFaults faults;
        DemandTextureLoader loader(smallOptions(1));
        faults.state->fail(operation, hipErrorOutOfMemory);
        auto source = std::make_shared<RegistrationSource>();
        source->hash = 9876;
        const unsigned char pixel = 7;
        const auto create = [&] {
            if (operation == HipOperation::CacheData)
                return loader.createTextureFromMemory(&pixel, 1, 1, 1);
            if (operation == HipOperation::FilenameMap)
                return loader.createTexture("transaction-lazy-file");
            return loader.createTexture(source);
        };
        expectFailure(create(), LoaderError::OutOfMemory);
        EXPECT_EQ(loader.getResidentTextureCount(), 0u);
        const auto success = create();
        ASSERT_TRUE(success.valid);
        EXPECT_EQ(success.id, 0u);
        EXPECT_EQ(success.error, LoaderError::Success);
    }
}

#ifdef USE_OIIO
TEST_F(RegistrationErrors, FilenameMapRollsBackWhenSourceMapAllocationFails) {
    ScopedFaults faults;
    DemandTextureLoader loader(smallOptions(1));
    faults.state->fail(HipOperation::SourceMap, hipErrorOutOfMemory);
    const auto path = (std::filesystem::path(getTestImagesPath()) / "checker.exr").string();
    ASSERT_TRUE(std::filesystem::exists(path));
    expectFailure(loader.createTexture(path), LoaderError::OutOfMemory);
    const auto success = loader.createTexture(path);
    ASSERT_TRUE(success.valid);
    EXPECT_EQ(success.id, 0u);
    EXPECT_GT(success.width, 0);
    EXPECT_EQ(loader.createTexture(path).id, success.id);
}
#endif

TEST_F(RegistrationErrors, FaultStateIsIsolatedResetAndRetainsRawErrors) {
    std::unique_ptr<DemandTextureLoader> first;
    std::shared_ptr<HipFaultState> state;
    {
        ScopedFaults faults;
        state = faults.state;
        first = std::make_unique<DemandTextureLoader>(smallOptions());
    }
    DemandTextureLoader second(smallOptions());
    state->fail(HipOperation::AllocateArray, hipErrorInvalidValue);
    auto a = first->createTexture(std::make_shared<RegistrationSource>(), singleLevel());
    auto b = second.createTexture(std::make_shared<RegistrationSource>(), singleLevel());
    request(second, {b.id}, 1);
    request(*first, {a.id}, 0);
    EXPECT_EQ(first->getLastError(), LoaderError::HipError);
    const auto records = state->records();
    ASSERT_FALSE(records.empty());
    EXPECT_EQ(records.back().operation, HipOperation::AllocateArray);
    EXPECT_EQ(records.back().error, hipErrorInvalidValue);
    EXPECT_TRUE(records.back().injected);
    request(*first, {a.id}, 1);
    EXPECT_EQ(first->getResidentTextureCount(), 1u);
}

TEST_F(RegistrationErrors, FaultStateFollowsParallelWorkerLoads) {
    ScopedFaults faults;
    DemandTextureLoader loader(smallOptions());
    const auto a = loader.createTexture(std::make_shared<RegistrationSource>(), singleLevel());
    const auto b = loader.createTexture(std::make_shared<RegistrationSource>(), singleLevel());
    faults.state->fail(HipOperation::Upload, hipErrorOutOfMemory);
    request(loader, {a.id, b.id}, 1);
    EXPECT_EQ(loader.getLastError(), LoaderError::OutOfMemory);
    EXPECT_EQ(loader.getResidentTextureCount(), 1u);
    size_t injected = 0;
    for (const auto& record : faults.state->records())
        if (record.operation == HipOperation::Upload && record.injected) ++injected;
    EXPECT_EQ(injected, 1u);
    request(loader, {a.id, b.id}, 1);
    EXPECT_EQ(loader.getResidentTextureCount(), 2u);
}

class RegistrationUploadFailure : public HipTestFixture,
    public testing::WithParamInterface<std::tuple<bool, HipOperation>> {};

TEST_P(RegistrationUploadFailure, FailureRollbackCleanupAndRetry) {
    const auto [mipmapped, operation] = GetParam();
    ScopedFaults faults;
    DemandTextureLoader loader(smallOptions());
    TextureDesc desc = singleLevel();
    desc.generateMipmaps = mipmapped;
    const auto texture = loader.createTexture(std::make_shared<RegistrationSource>(), desc);
    ASSERT_TRUE(texture.valid);
    const auto allocation = mipmapped ? HipOperation::AllocateMipmapped : HipOperation::AllocateArray;
    const auto cleanup = mipmapped ? HipOperation::FreeMipmapped : HipOperation::FreeArray;
    const auto failureOperation = operation == HipOperation::AllocateArray ? allocation : operation;
    faults.state->fail(failureOperation, hipErrorInvalidValue);
    if (failureOperation != allocation)
        faults.state->fail(cleanup, hipErrorUnknown);
    request(loader, {texture.id}, 0);
    EXPECT_EQ(loader.getLastError(), LoaderError::HipError);
    EXPECT_EQ(loader.getResidentTextureCount(), 0u);
    EXPECT_EQ(loader.getTotalTextureMemory(), failureOperation == allocation ? 0u : (mipmapped ? 20u : 16u));
    EXPECT_EQ(textureObject(loader, texture.id), hipTextureObject_t{});
    bool primary = false, secondary = false;
    for (const auto& record : faults.state->records()) {
        if (record.operation == failureOperation && record.injected)
            primary = record.error == hipErrorInvalidValue;
        if (record.operation == cleanup && record.injected)
            secondary = record.error == hipErrorUnknown;
    }
    EXPECT_TRUE(primary) << "Required multilevel allocation must execute, not silently fall back";
    EXPECT_EQ(secondary, failureOperation != allocation);
    request(loader, {texture.id}, 1);
    EXPECT_EQ(loader.getResidentTextureCount(), 1u);
    loader.unloadTexture(texture.id);
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
}

INSTANTIATE_TEST_SUITE_P(Faults, RegistrationUploadFailure,
    testing::Combine(testing::Bool(),
                     testing::Values(HipOperation::AllocateArray, HipOperation::Upload, HipOperation::CreateSampler)));

TEST_F(RegistrationErrors, CleanupFailureRetainsBackingUntilRetry) {
    for (auto operation : {HipOperation::DestroySampler, HipOperation::FreeArray, HipOperation::FreeMipmapped}) {
        ScopedFaults faults;
        DemandTextureLoader loader(smallOptions());
        auto desc = singleLevel();
        desc.generateMipmaps = operation == HipOperation::FreeMipmapped;
        auto handle = loader.createTexture(std::make_shared<RegistrationSource>(), desc);
        request(loader, {handle.id}, 1);
        const auto memory = loader.getTotalTextureMemory();
        faults.state->fail(operation, hipErrorUnknown);
        loader.unloadTexture(handle.id);
        EXPECT_EQ(loader.getLastError(), LoaderError::HipError);
        EXPECT_EQ(loader.getResidentTextureCount(), 0u);
        EXPECT_EQ(loader.getTotalTextureMemory(), memory);
        loader.unloadTexture(handle.id);
        EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    }
}
} // namespace
} // namespace test
} // namespace hip_demand
