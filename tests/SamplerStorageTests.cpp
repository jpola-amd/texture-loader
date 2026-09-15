// SPDX-License-Identifier: MIT
#include "TestUtils.h"
#include "ImageDataTestUtils.h"
#include "TextureSamplingHarness.h"
#include <DemandLoading/Internal/HipCalls.h>
#include <DemandLoading/Internal/TextureIdentity.h>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <unordered_map>

namespace hip_demand { namespace test {
namespace {
using internal::HipFaultState;
using internal::HipOperation;
constexpr float SharedPixelTolerance = 1e-6f;

class SharedSource : public TypedImageSource {
public:
    explicit SharedSource(TypedImageSource source = makeBoundaryPatternSource(false),
                          unsigned long long identity = 0)
        : TypedImageSource(std::move(source)), hash(identity) {}
    unsigned long long getHash(hipStream_t = nullptr) const override { return hash; }
    unsigned long long hash;
};

class BlockedSharedSource : public SharedSource {
public:
    bool readMipLevel(char* dest, unsigned int level, unsigned int width,
                      unsigned int height, hipStream_t stream) override {
        {
            std::unique_lock<std::mutex> lock(mutex);
            entered = true;
            cv.notify_all();
            cv.wait(lock, [&] { return released; });
        }
        return SharedSource::readMipLevel(dest, level, width, height, stream);
    }
    bool waitForRead() {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, std::chrono::seconds(10), [&] { return entered; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }
private:
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, released = false;
};

class ThrowingSharedSource : public SharedSource {
public:
    explicit ThrowingSharedSource(bool allocation)
        : SharedSource(makeAuthoredMipSource(false)), allocationFailure(allocation) {}
    bool readMipLevel(char* dest, unsigned int level, unsigned int width,
                      unsigned int height, hipStream_t stream) override {
        if (fail && level == 1) {
            if (allocationFailure) throw std::bad_alloc();
            throw std::runtime_error("Injected authored mip read exception");
        }
        return SharedSource::readMipLevel(dest, level, width, height, stream);
    }
    bool fail = true;
    bool allocationFailure;
};

struct FaultScope {
    std::shared_ptr<HipFaultState> state = std::make_shared<HipFaultState>();
    FaultScope() { internal::setHipFaultState(state); }
    ~FaultScope() { internal::setHipFaultState(nullptr); }
    size_t calls(HipOperation operation, bool onlySuccess = true) const {
        size_t count = 0;
        for (const auto& record : state->records())
            if (record.operation == operation && (!onlySuccess || record.error == hipSuccess))
                ++count;
        return count;
    }
};

LoaderOptions sharingOptions(size_t capacity = 32) {
    LoaderOptions options;
    options.maxTextures = capacity;
    options.maxRequestsPerLaunch = 64;
    options.maxThreads = 4;
    options.enableEviction = false;
    options.minResidentFrames = 0;
    return options;
}

TextureDesc sharingDesc(bool mips = false) {
    TextureDesc desc;
    desc.generateMipmaps = mips;
    desc.filterMode = hipFilterModePoint;
    desc.mipmapFilterMode = hipFilterModePoint;
    return desc;
}

struct ConstantHash {
    template<class T> size_t operator()(const T&) const { return 0; }
};

TEST(SamplerStorageKeys, DescriptorHashIgnoresPaddingAndEqualityResolvesCollisions) {
    alignas(TextureDesc) unsigned char firstBytes[sizeof(TextureDesc)];
    alignas(TextureDesc) unsigned char secondBytes[sizeof(TextureDesc)];
    std::memset(firstBytes, 0x55, sizeof(firstBytes));
    std::memset(secondBytes, 0xaa, sizeof(secondBytes));
    const auto* first = new (firstBytes) TextureDesc;
    const auto* second = new (secondBytes) TextureDesc;
    EXPECT_TRUE(*first == *second);
    EXPECT_EQ(internal::TextureDescHash{}(*first), internal::TextureDescHash{}(*second));
    std::unordered_map<TextureDesc, unsigned int, ConstantHash> keys;
    keys.emplace(*first, 0);
    auto variant = *first;
    variant.addressMode[0] = hipAddressModeClamp;
    keys.emplace(variant, 1);
    EXPECT_EQ(keys.size(), 2u);
    EXPECT_EQ(keys.at(*second), 0u);
    EXPECT_EQ(keys.at(variant), 1u);
}

TEST(SamplerStorageKeys, StorageEqualityResolvesCollisionsAcrossEveryField) {
    internal::StorageKey base;
    base.metadata = makeAuthoredMipSource(false).info;
    std::vector<internal::StorageKey> keys{base};
    SharedSource source;
    for (unsigned int field = 0; field < 14; ++field) {
        auto changed = base;
        switch (field) {
        case 0: changed.identity = internal::StorageIdentity::Content; break;
        case 1: changed.filename = "different"; break;
        case 2: changed.source = &source; break;
        case 3: changed.contentHash = 123; break;
        case 4: changed.metadata.width *= 2; break;
        case 5: changed.metadata.height *= 2; break;
        case 6: changed.metadata.format = HIP_AD_FORMAT_FLOAT; break;
        case 7: changed.metadata.numChannels = 3; break;
        case 8: changed.metadata.numMipLevels = 1; break;
        case 9: changed.metadata.isValid = false; break;
        case 10: changed.metadata.isTiled = true; break;
        case 11: changed.sRGB = true; break;
        case 12: changed.generateMipmaps = false; break;
        case 13: changed.maxMipLevel = 2; break;
        }
        EXPECT_FALSE(changed == base);
        keys.push_back(changed);
    }
    std::unordered_map<internal::StorageKey, size_t, ConstantHash> collisions;
    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(collisions.emplace(keys[i], i).second);
        const auto copy = keys[i];
        EXPECT_EQ(internal::StorageKeyHash{}(copy), internal::StorageKeyHash{}(keys[i]));
    }
    EXPECT_EQ(collisions.size(), keys.size());
    for (size_t i = 0; i < keys.size(); ++i)
        EXPECT_EQ(collisions.at(keys[i]), i);
}

class SamplerStorage : public HipTestFixture {
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
    hipTextureObject_t object(DemandTextureLoader& loader, uint32_t id) {
        hipTextureObject_t value = 0;
        EXPECT_EQ(hipMemcpy(&value, loader.getDeviceContext().textures + id, sizeof(value),
                            hipMemcpyDeviceToHost), hipSuccess);
        return value;
    }
    hipResourceDesc resource(DemandTextureLoader& loader, uint32_t id) {
        hipResourceDesc result{};
        const auto texture = object(loader, id);
        EXPECT_NE(texture, hipTextureObject_t{});
        if (texture)
            EXPECT_EQ(hipGetTextureObjectResourceDesc(&result, texture), hipSuccess);
        return result;
    }
    void expectShared(DemandTextureLoader& loader, uint32_t a, uint32_t b, bool mips) {
        EXPECT_NE(object(loader, a), object(loader, b));
        const auto first = resource(loader, a), second = resource(loader, b);
        ASSERT_EQ(first.resType, mips ? hipResourceTypeMipmappedArray : hipResourceTypeArray);
        ASSERT_EQ(second.resType, first.resType);
        if (mips) EXPECT_EQ(first.res.mipmap.mipmap, second.res.mipmap.mipmap);
        else EXPECT_EQ(first.res.array.array, second.res.array.array);
    }
    void expectSamePixel(const SamplingResult& actual, const SamplingResult& expected) {
        ASSERT_EQ(actual.resident, expected.resident);
        const std::array<float, 4> a{actual.value.x, actual.value.y, actual.value.z, actual.value.w};
        const std::array<float, 4> b{expected.value.x, expected.value.y, expected.value.z, expected.value.w};
        for (size_t c = 0; c < a.size(); ++c)
            EXPECT_NEAR(a[c], b[c], SharedPixelTolerance) << "channel=" << c;
    }
};

class SamplerStorageIdentity : public SamplerStorage, public testing::WithParamInterface<unsigned int> {};

TEST_P(SamplerStorageIdentity, EveryDescriptorFieldHasIndependentRegistration) {
    DemandTextureLoader loader(sharingOptions());
    auto source = std::make_shared<SharedSource>();
    const TextureDesc base;
    TextureDesc changed = base;
    switch (GetParam()) {
    case 0: changed.addressMode[0] = hipAddressModeClamp; break;
    case 1: changed.addressMode[1] = hipAddressModeMirror; break;
    case 2: changed.filterMode = hipFilterModePoint; break;
    case 3: changed.mipmapFilterMode = hipFilterModePoint; break;
    case 4: changed.normalizedCoords = false; break;
    case 5: changed.sRGB = true; break;
    case 6: changed.generateMipmaps = false; break;
    case 7: changed.maxMipLevel = 1; break;
    case 8: changed.evictionPriority = EvictionPriority::KeepResident; break;
    }
    for (bool filename : {false, true}) {
        const auto create = [&](const TextureDesc& desc) {
            return filename ? loader.createTexture("sampler-storage-lazy-file", desc)
                            : loader.createTexture(source, desc);
        };
        const auto first = create(base), variant = create(changed);
        ASSERT_TRUE(first.valid);
        ASSERT_TRUE(variant.valid);
        EXPECT_NE(first.id, variant.id);
        EXPECT_EQ(create(base).id, first.id);
        EXPECT_EQ(create(changed).id, variant.id);
    }
}
INSTANTIATE_TEST_SUITE_P(Fields, SamplerStorageIdentity, testing::Range(0u, 9u));

TEST_F(SamplerStorage, CapacityCountsVariantsAndPriorityMutationPreservesLiveIds) {
    DemandTextureLoader loader(sharingOptions(2));
    auto source = std::make_shared<SharedSource>();
    auto normal = sharingDesc(), high = normal;
    high.evictionPriority = EvictionPriority::High;
    const auto a = loader.createTexture(source, normal), b = loader.createTexture(source, high);
    ASSERT_TRUE(a.valid);
    ASSERT_TRUE(b.valid);
    ASSERT_NE(a.id, b.id);
    auto third = normal;
    third.addressMode[0] = hipAddressModeClamp;
    EXPECT_EQ(loader.createTexture(source, third).error, LoaderError::MaxTexturesExceeded);
    EXPECT_EQ(loader.createTexture(source, high).id, b.id);
    loader.updateEvictionPriority(a.id, EvictionPriority::High);
    const auto converged = loader.createTexture(source, high);
    EXPECT_TRUE(converged.id == a.id || converged.id == b.id);
    EXPECT_EQ(loader.createTexture(source, normal).error, LoaderError::MaxTexturesExceeded);
    loader.updateEvictionPriority(a.id, EvictionPriority::Normal);
    EXPECT_EQ(loader.createTexture(source, normal).id, a.id);
    EXPECT_EQ(loader.createTexture(source, high).id, b.id);
    loader.updateEvictionPriority(a.id, static_cast<EvictionPriority>(99));
    EXPECT_EQ(loader.getLastError(), LoaderError::InvalidParameter);
    EXPECT_EQ(loader.createTexture(source, normal).id, a.id);
    request(loader, {a.id, b.id}, 2);
    expectShared(loader, a.id, b.id, false);
}

TEST_F(SamplerStorage, ContentAliasesRetainOnlyOwningSourceAndSeparateNamespacesAndFormats) {
    FaultScope faults;
    std::weak_ptr<SharedSource> retained;
    {
        DemandTextureLoader loader(sharingOptions());
        const std::string filename = "sampler-storage-namespace";
        const auto hash = std::hash<std::string>{}(filename);
        auto source = std::make_shared<SharedSource>(makeBoundaryPatternSource(false), hash);
        retained = source;
        auto desc = sharingDesc(), variant = desc;
        variant.addressMode[0] = hipAddressModeClamp;
        const auto a = loader.createTexture(source, desc);
        const auto file = loader.createTexture(filename, desc);
        EXPECT_NE(a.id, file.id);
        auto alias = std::make_shared<SharedSource>(makeBoundaryPatternSource(false), hash);
        std::weak_ptr<SharedSource> incoming = alias;
        EXPECT_EQ(loader.createTexture(alias, desc).id, a.id);
        const auto b = loader.createTexture(alias, variant);
        ASSERT_TRUE(b.valid);
        alias.reset();
        EXPECT_TRUE(incoming.expired());
        auto floating = std::make_shared<SharedSource>(makeBoundaryPatternSource(true), hash);
        const auto c = loader.createTexture(floating, desc);
        ASSERT_TRUE(c.valid);
        EXPECT_NE(c.id, a.id);
        source.reset();
        EXPECT_FALSE(retained.expired());
        request(loader, {b.id, a.id, c.id}, 3);
        expectShared(loader, a.id, b.id, false);
        EXPECT_NE(resource(loader, a.id).res.array.array, resource(loader, c.id).res.array.array);
        EXPECT_EQ(loader.getTotalTextureMemory(), 64u + 256u);
        EXPECT_EQ(retained.lock()->reads, 1u);
        EXPECT_EQ(floating->reads, 1u);
        loader.unloadAll();
        EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    }
    EXPECT_TRUE(retained.expired());
    EXPECT_EQ(faults.calls(HipOperation::AllocateArray), 2u);
    EXPECT_EQ(faults.calls(HipOperation::FreeArray), 2u);
    EXPECT_EQ(faults.calls(HipOperation::DestroySampler), 3u);
}

TEST_F(SamplerStorage, ContentMetadataAndStoragePoliciesAreConservative) {
    DemandTextureLoader loader(sharingOptions());
    const auto desc = sharingDesc(true);
    auto source = std::make_shared<SharedSource>(makeAuthoredMipSource(true), 789);
    const auto original = loader.createTexture(source, desc);
    std::vector<uint32_t> ids{original.id};
    for (unsigned int field = 0; field < 3; ++field) {
        auto changed = desc;
        if (field == 0) changed.sRGB = true;
        if (field == 1) changed.generateMipmaps = false;
        if (field == 2) changed.maxMipLevel = 2;
        const auto variant = loader.createTexture(source, changed);
        ASSERT_TRUE(variant.valid);
        EXPECT_NE(original.id, variant.id);
        ids.push_back(variant.id);
    }
    auto differentSize = std::make_shared<SharedSource>(makeBoundaryPatternSource(true), 789);
    const auto sizeVariant = loader.createTexture(differentSize, desc);
    ASSERT_TRUE(sizeVariant.valid);
    EXPECT_NE(original.id, sizeVariant.id);
    ids.push_back(sizeVariant.id);
    request(loader, ids, ids.size());
    std::set<hipMipmappedArray_t> arrays;
    for (uint32_t id : {ids[0], ids[1], ids[3], ids[4]}) {
        const auto backing = resource(loader, id);
        ASSERT_EQ(backing.resType, hipResourceTypeMipmappedArray);
        EXPECT_TRUE(arrays.insert(backing.res.mipmap.mipmap).second);
    }
    EXPECT_EQ(resource(loader, ids[2]).resType, hipResourceTypeArray);
    EXPECT_EQ(loader.getTotalTextureMemory(), 1360u * 2 + 1024u + 1280u + 336u);
}

TEST_F(SamplerStorage, ContentHashDoesNotOverrideNativeChannelsOrAuthoredMipCount) {
    DemandTextureLoader loader(sharingOptions());
    auto complete = std::make_shared<SharedSource>(makeAuthoredMipSource(true), 321);
    auto fewerMips = std::make_shared<SharedSource>(makeAuthoredMipSource(true), 321);
    fewerMips->info.numMipLevels = 1;
    const auto a = loader.createTexture(complete), b = loader.createTexture(fewerMips);
    ASSERT_TRUE(a.valid);
    ASSERT_TRUE(b.valid);
    EXPECT_NE(a.id, b.id);
    auto rgba = std::make_shared<SharedSource>(makeSource(0, 4), 654);
    auto rgb = std::make_shared<SharedSource>(makeSource(0, 3), 654);
    const auto c = loader.createTexture(rgba), d = loader.createTexture(rgb);
    ASSERT_TRUE(c.valid);
    ASSERT_TRUE(d.valid);
    EXPECT_NE(c.id, d.id);
}

#ifdef USE_OIIO
TEST_F(SamplerStorage, FilenameVariantsShareProductionReaderBacking) {
    FaultScope faults;
    DemandTextureLoader loader(sharingOptions());
    const auto filename = (std::filesystem::path(getTestImagesPath()) / "checker.exr").string();
    ASSERT_TRUE(std::filesystem::exists(filename));
    auto desc = sharingDesc(), variant = desc;
    variant.filterMode = hipFilterModeLinear;
    const auto a = loader.createTexture(filename, desc), b = loader.createTexture(filename, variant);
    ASSERT_TRUE(a.valid);
    ASSERT_TRUE(b.valid);
    ASSERT_NE(a.id, b.id);
    request(loader, {b.id, a.id}, 2);
    expectShared(loader, a.id, b.id, false);
    EXPECT_EQ(faults.calls(HipOperation::AllocateArray), 1u);
    EXPECT_EQ(faults.calls(HipOperation::Upload), 1u);
    EXPECT_EQ(loader.getTotalTextureMemory(), size_t(a.width) * a.height * 16);
}
#endif

class SamplerStoragePixels : public SamplerStorage,
    public testing::WithParamInterface<std::tuple<bool, bool, bool, bool>> {};

TEST_P(SamplerStoragePixels, ReversedCreationAndDemandMatchIsolatedPixelsAndShareOneUpload) {
    const auto [floating, mips, reverseCreation, reverseDemand] = GetParam();
    FaultScope faults;
    DemandTextureLoader loader(sharingOptions());
    auto source = std::make_shared<SharedSource>(makeBoundaryPatternSource(floating));
    auto point = sharingDesc(mips), linear = point;
    linear.addressMode[0] = linear.addressMode[1] = hipAddressModeClamp;
    linear.filterMode = hipFilterModeLinear;
    linear.mipmapFilterMode = hipFilterModeLinear;
    TextureHandle a, b;
    if (reverseCreation) {
        b = loader.createTexture(source, linear);
        a = loader.createTexture(source, point);
    } else {
        a = loader.createTexture(source, point);
        b = loader.createTexture(source, linear);
    }
    ASSERT_TRUE(a.valid);
    ASSERT_TRUE(b.valid);
    ASSERT_NE(a.id, b.id);
    const auto first = reverseDemand ? b : a, second = reverseDemand ? a : b;
    request(loader, {first.id}, 1);
    request(loader, {second.id}, 1);
    expectShared(loader, a.id, b.id, mips);
    EXPECT_EQ(source->reads, 1u);
    EXPECT_EQ(faults.calls(mips ? HipOperation::AllocateMipmapped : HipOperation::AllocateArray), 1u);
    EXPECT_EQ(faults.calls(HipOperation::Upload), mips ? 3u : 1u);
    EXPECT_EQ(faults.calls(HipOperation::CreateSampler), 2u);
    const size_t bytes = (mips ? 21u : 16u) * (floating ? 16u : 4u);
    EXPECT_EQ(loader.getTotalTextureMemory(), bytes);

    const auto isolatedA = loader.createTexture(std::make_shared<SharedSource>(makeBoundaryPatternSource(floating)), point);
    const auto isolatedB = loader.createTexture(std::make_shared<SharedSource>(makeBoundaryPatternSource(floating)), linear);
    request(loader, {isolatedA.id, isolatedB.id}, 2);
    TextureSamplingHarness harness;
    ASSERT_EQ(harness.open(samplingModulePath()), hipSuccess);
    for (unsigned int pass = 0; pass < 2; ++pass) {
        for (SamplingPath path : {SamplingPath::Implicit, SamplingPath::Lod, SamplingPath::Gradient}) {
            std::vector<SamplingInput> inputs;
            for (const auto& uv : {std::pair<float, float>{-.125f, .375f}, {1.125f, .625f},
                                   {.25f, .5f}, {.875f, -.125f}, {.375f, 1.125f}}) {
                for (uint32_t id : {a.id, b.id, isolatedA.id, isolatedB.id}) {
                    SamplingInput input;
                    input.textureId = id;
                    input.path = path;
                    input.u = uv.first;
                    input.v = uv.second;
                    input.ddx = {.0625f, 0};
                    input.ddy = {0, .0625f};
                    inputs.push_back(input);
                }
            }
            std::vector<SamplingResult> results;
            ASSERT_EQ(harness.sample(loader.getDeviceContext(), inputs, results), hipSuccess);
            ASSERT_EQ(results.size(), inputs.size());
            bool distinctPixels = false;
            for (size_t i = 0; i < results.size(); i += 4) {
                ASSERT_EQ(results[i].resident, 1u);
                ASSERT_EQ(results[i + 1].resident, 1u);
                expectSamePixel(results[i], results[i + 2]);
                expectSamePixel(results[i + 1], results[i + 3]);
                distinctPixels |= std::abs(results[i].value.x - results[i + 1].value.x) > .001f;
            }
            EXPECT_TRUE(distinctPixels);
        }
        if (pass == 0) {
            const auto siblingObject = object(loader, b.id);
            loader.unloadTexture(a.id);
            EXPECT_EQ(loader.getResidentTextureCount(), 3u);
            EXPECT_EQ(object(loader, a.id), hipTextureObject_t{});
            EXPECT_EQ(object(loader, b.id), siblingObject);
            EXPECT_EQ(loader.getTotalTextureMemory(), bytes * 3);
            request(loader, {a.id}, 1);
            expectShared(loader, a.id, b.id, mips);
            EXPECT_EQ(source->reads, 1u);
        }
    }
    loader.unloadAll();
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    EXPECT_EQ(loader.getResidentTextureCount(), 0u);
}
INSTANTIATE_TEST_SUITE_P(FormatsAndOrders, SamplerStoragePixels,
    testing::Combine(testing::Bool(), testing::Bool(), testing::Bool(), testing::Bool()));

class SamplerStorageBatch : public SamplerStorage, public testing::WithParamInterface<bool> {};

TEST_P(SamplerStorageBatch, ConcurrentRegistrationAndBatchDemandCompleteEveryVariantOnce) {
    const bool mips = GetParam();
    FaultScope faults;
    DemandTextureLoader loader(sharingOptions(8));
    auto source = std::make_shared<SharedSource>(makeAuthoredMipSource(true));
    std::array<TextureHandle, 16> handles{};
    std::vector<std::thread> workers;
    for (unsigned int i = 0; i < handles.size(); ++i) {
        workers.emplace_back([&, i] {
            auto desc = sharingDesc(mips);
            if (i % 2) desc.filterMode = hipFilterModeLinear;
            handles[i] = loader.createTexture(source, desc);
        });
    }
    for (auto& worker : workers) worker.join();
    std::vector<uint32_t> ids;
    for (size_t i = 0; i < handles.size(); ++i) {
        ASSERT_TRUE(handles[i].valid);
        EXPECT_EQ(handles[i].id, handles[i % 2].id);
        ids.push_back(handles[i].id);
    }
    EXPECT_NE(handles[0].id, handles[1].id);
    queue(loader, ids);
    auto ticket = loader.processRequestsAsync(nullptr, loader.getDeviceContext());
    ASSERT_EQ(ticket.numTasksTotal(), 1);
    ticket.wait();
    EXPECT_EQ(ticket.numTasksRemaining(), 0);
    loader.launchPrepare();
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(loader.getResidentTextureCount(), 2u);
    expectShared(loader, handles[0].id, handles[1].id, mips);
    EXPECT_EQ(source->reads, mips ? 4u : 1u);
    EXPECT_EQ(faults.calls(mips ? HipOperation::AllocateMipmapped : HipOperation::AllocateArray), 1u);
    EXPECT_EQ(faults.calls(HipOperation::Upload), mips ? 4u : 1u);
    EXPECT_EQ(loader.getTotalTextureMemory(), mips ? 1360u : 1024u);
}

TEST_P(SamplerStorageBatch, LiveSiblingSurvivesFailedSamplerAndCleanupRetry) {
    const bool mips = GetParam();
    FaultScope faults;
    DemandTextureLoader loader(sharingOptions());
    auto source = std::make_shared<SharedSource>();
    auto first = sharingDesc(mips), second = first;
    second.filterMode = hipFilterModeLinear;
    const auto a = loader.createTexture(source, first), b = loader.createTexture(source, second);
    request(loader, {a.id}, 1);
    const auto old = object(loader, a.id);
    const auto bytes = loader.getTotalTextureMemory();
    faults.state->fail(HipOperation::CreateSampler, hipErrorNotSupported);
    request(loader, {b.id}, 0);
    EXPECT_EQ(loader.getResidentTextureCount(), 1u);
    EXPECT_EQ(loader.getLastError(), LoaderError::HipError);
    EXPECT_EQ(object(loader, a.id), old);
    EXPECT_EQ(object(loader, b.id), hipTextureObject_t{});
    EXPECT_EQ(loader.getTotalTextureMemory(), bytes);
    EXPECT_EQ(faults.calls(mips ? HipOperation::FreeMipmapped : HipOperation::FreeArray), 0u);
    request(loader, {b.id}, 1);
    EXPECT_EQ(source->reads, 1u);
    faults.state->fail(HipOperation::DestroySampler, hipErrorUnknown);
    loader.unloadTexture(a.id);
    EXPECT_EQ(loader.getLastError(), LoaderError::HipError);
    EXPECT_EQ(object(loader, a.id), hipTextureObject_t{});
    EXPECT_NE(object(loader, b.id), hipTextureObject_t{});
    EXPECT_EQ(loader.getResidentTextureCount(), 1u);
    loader.unloadTexture(b.id);
    EXPECT_EQ(loader.getTotalTextureMemory(), bytes);
    EXPECT_EQ(faults.calls(mips ? HipOperation::FreeMipmapped : HipOperation::FreeArray), 0u);
    loader.unloadTexture(a.id);
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    EXPECT_EQ(faults.calls(mips ? HipOperation::FreeMipmapped : HipOperation::FreeArray), 1u);
}

TEST_P(SamplerStorageBatch, BatchSamplerFailureLeavesSuccessfulSiblingPublished) {
    const bool mips = GetParam();
    for (size_t failedSibling : {1u, 2u}) {
        SCOPED_TRACE(failedSibling);
        FaultScope faults;
        DemandTextureLoader loader(sharingOptions());
        auto source = std::make_shared<SharedSource>();
        auto desc = sharingDesc(mips), variant = desc;
        variant.filterMode = hipFilterModeLinear;
        const auto a = loader.createTexture(source, desc), b = loader.createTexture(source, variant);
        faults.state->fail(HipOperation::CreateSampler, hipErrorNotSupported, failedSibling);
        request(loader, {a.id, b.id}, 1);
        EXPECT_EQ(loader.getResidentTextureCount(), 1u);
        EXPECT_NE(object(loader, a.id) != 0, object(loader, b.id) != 0);
        EXPECT_EQ(loader.getTotalTextureMemory(), mips ? 84u : 64u);
        EXPECT_EQ(faults.calls(mips ? HipOperation::AllocateMipmapped : HipOperation::AllocateArray), 1u);
        EXPECT_EQ(faults.calls(mips ? HipOperation::FreeMipmapped : HipOperation::FreeArray), 0u);
        request(loader, {a.id, b.id}, 1);
        expectShared(loader, a.id, b.id, mips);
        EXPECT_EQ(source->reads, 1u);
    }
}

TEST_P(SamplerStorageBatch, FailedGroupBackingCleanupRetainsChargeUntilRetry) {
    const bool mips = GetParam();
    FaultScope faults;
    DemandTextureLoader loader(sharingOptions());
    auto source = std::make_shared<SharedSource>();
    auto desc = sharingDesc(mips), variant = desc;
    variant.filterMode = hipFilterModeLinear;
    const auto a = loader.createTexture(source, desc), b = loader.createTexture(source, variant);
    request(loader, {a.id, b.id}, 2);
    const auto cleanup = mips ? HipOperation::FreeMipmapped : HipOperation::FreeArray;
    faults.state->fail(cleanup, hipErrorUnknown);
    loader.unloadTexture(a.id);
    loader.unloadTexture(b.id);
    EXPECT_EQ(loader.getResidentTextureCount(), 0u);
    EXPECT_EQ(loader.getLastError(), LoaderError::HipError);
    EXPECT_EQ(object(loader, a.id), hipTextureObject_t{});
    EXPECT_EQ(object(loader, b.id), hipTextureObject_t{});
    EXPECT_EQ(loader.getTotalTextureMemory(), mips ? 84u : 64u);
    EXPECT_EQ(faults.calls(cleanup), 0u);
    loader.unloadAll();
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    EXPECT_EQ(faults.calls(cleanup), 1u);
    request(loader, {a.id, b.id}, 2);
    expectShared(loader, a.id, b.id, mips);
    EXPECT_EQ(source->reads, 2u);
}

TEST_P(SamplerStorageBatch, GroupEvictionAggregatesProtectionAndInvalidatesEveryMapping) {
    const bool mips = GetParam();
    auto options = sharingOptions();
    options.enableEviction = true;
    options.maxTextureMemory = mips ? 84 : 64;
    DemandTextureLoader loader(options);
    auto source = std::make_shared<SharedSource>();
    auto desc = sharingDesc(mips), protectedDesc = desc;
    protectedDesc.evictionPriority = EvictionPriority::KeepResident;
    const auto a = loader.createTexture(source, desc), protectedId = loader.createTexture(source, protectedDesc);
    const auto other = loader.createTexture(std::make_shared<SharedSource>(), desc);
    request(loader, {a.id}, 1);
    request(loader, {other.id}, 1);
    EXPECT_NE(object(loader, a.id), hipTextureObject_t{}) << "A registered KeepResident sibling protects backing before demand";
    EXPECT_EQ(loader.getResidentTextureCount(), 2u);
    request(loader, {protectedId.id}, 1);
    loader.updateEvictionPriority(protectedId.id, EvictionPriority::Low);
    loader.updateEvictionPriority(a.id, EvictionPriority::Low);
    loader.unloadTexture(other.id);
    request(loader, {other.id}, 1);
    EXPECT_EQ(object(loader, a.id), hipTextureObject_t{});
    EXPECT_EQ(object(loader, protectedId.id), hipTextureObject_t{});
    EXPECT_NE(object(loader, other.id), hipTextureObject_t{});
    EXPECT_EQ(loader.getResidentTextureCount(), 1u);
    EXPECT_EQ(loader.getTotalTextureMemory(), options.maxTextureMemory);
}
INSTANTIATE_TEST_SUITE_P(ResourceKinds, SamplerStorageBatch, testing::Bool());

TEST_F(SamplerStorage, EvictionUsesHighestSiblingPriorityRatherThanFirstRegistration) {
    auto options = sharingOptions();
    options.enableEviction = true;
    options.maxTextureMemory = 128;
    DemandTextureLoader loader(options);
    auto source = std::make_shared<SharedSource>();
    auto low = sharingDesc(), high = low, normal = low;
    low.evictionPriority = EvictionPriority::Low;
    high.evictionPriority = EvictionPriority::High;
    const auto a = loader.createTexture(source, low), b = loader.createTexture(source, high);
    const auto other = loader.createTexture(std::make_shared<SharedSource>(), normal);
    const auto incoming = loader.createTexture(std::make_shared<SharedSource>(), normal);
    request(loader, {a.id, b.id}, 2);
    request(loader, {other.id}, 1);
    request(loader, {incoming.id}, 1);
    EXPECT_NE(object(loader, a.id), hipTextureObject_t{});
    EXPECT_NE(object(loader, b.id), hipTextureObject_t{});
    EXPECT_EQ(object(loader, other.id), hipTextureObject_t{});
    EXPECT_EQ(loader.getResidentTextureCount(), 3u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 128u);
    loader.updateEvictionPriority(b.id, EvictionPriority::Low);
    request(loader, {other.id}, 1);
    EXPECT_EQ(object(loader, a.id), hipTextureObject_t{});
    EXPECT_EQ(object(loader, b.id), hipTextureObject_t{});
    EXPECT_EQ(loader.getResidentTextureCount(), 2u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 128u);
}

TEST_F(SamplerStorage, CurrentBatchProtectsAlreadyResidentSharedBacking) {
    auto options = sharingOptions();
    options.enableEviction = true;
    options.maxTextureMemory = 128;
    DemandTextureLoader loader(options);
    auto low = sharingDesc(), high = low, siblingDesc = low;
    low.evictionPriority = EvictionPriority::Low;
    high.evictionPriority = EvictionPriority::High;
    siblingDesc.filterMode = hipFilterModeLinear;
    auto source = std::make_shared<SharedSource>();
    const auto a = loader.createTexture(source, low), sibling = loader.createTexture(source, siblingDesc);
    const auto other = loader.createTexture(std::make_shared<SharedSource>(), high);
    const auto incoming = loader.createTexture(std::make_shared<SharedSource>(), sharingDesc());
    request(loader, {a.id, sibling.id}, 2);
    request(loader, {other.id}, 1);
    const auto original = object(loader, a.id);
    request(loader, {a.id, incoming.id}, 1);
    EXPECT_EQ(object(loader, a.id), original);
    EXPECT_NE(object(loader, sibling.id), hipTextureObject_t{});
    EXPECT_EQ(object(loader, other.id), hipTextureObject_t{});
    EXPECT_EQ(source->reads, 1u);
    EXPECT_EQ(loader.getResidentTextureCount(), 3u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 128u);
}

TEST_F(SamplerStorage, AuthoredMipVariantsPreservePixelsAndIndependentMipSelection) {
    DemandTextureLoader loader(sharingOptions());
    auto source = std::make_shared<SharedSource>(makeAuthoredMipSource(true));
    auto point = sharingDesc(true), linear = point;
    linear.mipmapFilterMode = hipFilterModeLinear;
    const auto a = loader.createTexture(source, point), b = loader.createTexture(source, linear);
    request(loader, {b.id, a.id}, 2);
    expectShared(loader, a.id, b.id, true);
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{0, 1, 2, 3}));
    const auto backing = resource(loader, a.id);
    for (unsigned int level = 0; level < source->info.numMipLevels; ++level) {
        hipArray_t array{};
        ASSERT_EQ(hipGetMipmappedArrayLevel(&array, backing.res.mipmap.mipmap, level), hipSuccess);
        const size_t rowBytes = (8u >> level) * 16;
        std::vector<unsigned char> pixels(source->mipPixels[level].size());
        ASSERT_EQ(hipMemcpy2DFromArray(pixels.data(), rowBytes, array, 0, 0, rowBytes,
                                      8u >> level, hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(pixels, source->mipPixels[level]);
    }
    TextureSamplingHarness harness;
    ASSERT_EQ(harness.open(samplingModulePath()), hipSuccess);
    for (float lod : {0.f, .5f, 1.f, 1.5f, 2.f, 2.5f, 3.f}) {
        std::vector<SamplingInput> inputs(2);
        inputs[0].textureId = a.id;
        inputs[1].textureId = b.id;
        for (auto& input : inputs) { input.path = SamplingPath::Lod; input.lod = lod; }
        std::vector<SamplingResult> results;
        ASSERT_EQ(harness.sample(loader.getDeviceContext(), inputs, results), hipSuccess);
        ASSERT_EQ(results.size(), 2u);
        const auto lower = static_cast<unsigned int>(lod);
        const auto upper = std::min(3u, lower + 1);
        const auto& lo = authoredFloatMipColors[lower];
        const auto& hi = authoredFloatMipColors[upper];
        const float fraction = lod - lower;
        const std::array<float, 4> p{results[0].value.x, results[0].value.y,
                                     results[0].value.z, results[0].value.w};
        const std::array<float, 4> l{results[1].value.x, results[1].value.y,
                                     results[1].value.z, results[1].value.w};
        bool selectsLower = true, selectsUpper = fraction != 0;
        for (size_t c = 0; c < 4; ++c) {
            selectsLower &= std::abs(p[c] - lo[c]) <= SharedPixelTolerance;
            selectsUpper &= std::abs(p[c] - hi[c]) <= SharedPixelTolerance;
            EXPECT_NEAR(l[c], lo[c] + fraction * (hi[c] - lo[c]), SharedPixelTolerance);
        }
        EXPECT_EQ(results[0].resident, 1u);
        EXPECT_EQ(results[1].resident, 1u);
        EXPECT_TRUE(selectsLower || selectsUpper) << "Point mip ties must select one complete authored color";
    }
}

class SamplerStorageFault : public SamplerStorage,
    public testing::WithParamInterface<std::tuple<bool, HipOperation>> {};

TEST_P(SamplerStorageFault, StorageFailuresRollbackAndExplicitRetryCompletesBothSamplers) {
    const auto [mips, operation] = GetParam();
    FaultScope faults;
    DemandTextureLoader loader(sharingOptions());
    auto source = std::make_shared<SharedSource>();
    auto desc = sharingDesc(mips), variant = desc;
    variant.filterMode = hipFilterModeLinear;
    const auto a = loader.createTexture(source, desc), b = loader.createTexture(source, variant);
    const auto allocation = mips ? HipOperation::AllocateMipmapped : HipOperation::AllocateArray;
    const auto failing = operation == HipOperation::AllocateArray ? allocation : operation;
    faults.state->fail(failing, hipErrorOutOfMemory);
    request(loader, {a.id}, 0);
    EXPECT_EQ(loader.getLastError(), LoaderError::OutOfMemory);
    EXPECT_EQ(loader.getResidentTextureCount(), 0u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    EXPECT_EQ(object(loader, a.id), hipTextureObject_t{});
    request(loader, {b.id, a.id}, 2);
    expectShared(loader, a.id, b.id, mips);
    EXPECT_EQ(loader.getResidentTextureCount(), 2u);
    EXPECT_EQ(loader.getTotalTextureMemory(), mips ? 84u : 64u);
}
INSTANTIATE_TEST_SUITE_P(StorageFailures, SamplerStorageFault,
    testing::Combine(testing::Bool(), testing::Values(HipOperation::AllocateArray, HipOperation::Upload,
                                                     HipOperation::CreateSampler)));

TEST_F(SamplerStorage, SourceReadFailureDoesNotPublishAndCanRetry) {
    DemandTextureLoader loader(sharingOptions());
    auto source = std::make_shared<SharedSource>();
    source->failRead = true;
    auto desc = sharingDesc(), variant = desc;
    variant.filterMode = hipFilterModeLinear;
    const auto a = loader.createTexture(source, desc), b = loader.createTexture(source, variant);
    request(loader, {a.id, b.id}, 0);
    EXPECT_EQ(source->reads, 1u);
    EXPECT_EQ(loader.getLastError(), LoaderError::ImageLoadFailed);
    EXPECT_EQ(loader.getResidentTextureCount(), 0u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    source->failRead = false;
    request(loader, {b.id, a.id}, 2);
    expectShared(loader, a.id, b.id, false);
}

TEST_F(SamplerStorage, AuthoredMipExceptionsRollbackAllSiblingsAndPreserveErrorCategory) {
    for (bool allocationFailure : {false, true}) {
        DemandTextureLoader loader(sharingOptions());
        auto source = std::make_shared<ThrowingSharedSource>(allocationFailure);
        auto desc = sharingDesc(true), variant = desc;
        variant.filterMode = hipFilterModeLinear;
        const auto a = loader.createTexture(source, desc), b = loader.createTexture(source, variant);
        request(loader, {a.id, b.id}, 0);
        EXPECT_EQ(loader.getLastError(), allocationFailure ? LoaderError::OutOfMemory : LoaderError::ImageLoadFailed);
        EXPECT_EQ(loader.getResidentTextureCount(), 0u);
        EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
        source->fail = false;
        request(loader, {b.id, a.id}, 2);
        expectShared(loader, a.id, b.id, true);
    }
}

TEST_F(SamplerStorage, DestructionRetriesTransientCleanupWithoutLosingSharedOwnership) {
    for (HipOperation operation : {HipOperation::DestroySampler, HipOperation::FreeArray,
                                   HipOperation::InvalidateMappings, HipOperation::SynchronizeConsumers}) {
        FaultScope faults;
        {
            DemandTextureLoader loader(sharingOptions());
            auto source = std::make_shared<SharedSource>();
            auto desc = sharingDesc(), variant = desc;
            variant.filterMode = hipFilterModeLinear;
            const auto a = loader.createTexture(source, desc), b = loader.createTexture(source, variant);
            request(loader, {a.id, b.id}, 2);
            faults.state->fail(operation, hipErrorUnknown);
        }
        EXPECT_EQ(faults.calls(HipOperation::DestroySampler), 2u);
        EXPECT_EQ(faults.calls(HipOperation::FreeArray), 1u);
    }
}

TEST_F(SamplerStorage, UnloadFencesQueuedSnapshotAndMakesStaleContextUnavailable) {
    DemandTextureLoader loader(sharingOptions());
    auto source = std::make_shared<SharedSource>();
    auto desc = sharingDesc(), variant = desc;
    variant.addressMode[0] = hipAddressModeClamp;
    const auto a = loader.createTexture(source, desc), b = loader.createTexture(source, variant);
    request(loader, {a.id, b.id}, 2);
    const auto oldContext = loader.getDeviceContext();
    TextureSamplingHarness harness;
    ASSERT_EQ(harness.open(samplingModulePath()), hipSuccess);
    SamplingInput input;
    input.textureId = a.id;
    input.path = SamplingPath::Lod;
    std::vector<SamplingResult> reference, inFlight, after;
    ASSERT_EQ(harness.sample(oldContext, {input}, reference), hipSuccess);
    input.path = SamplingPath::DelayedSnapshot;
    ASSERT_EQ(harness.sample(oldContext, {input}, inFlight, [&] { loader.unloadAll(); }), hipSuccess);
    ASSERT_EQ(inFlight.size(), 1u);
    expectSamePixel(inFlight[0], reference[0]);
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    input.path = SamplingPath::Lod;
    ASSERT_EQ(harness.sample(oldContext, {input}, after), hipSuccess);
    ASSERT_EQ(after.size(), 1u);
    EXPECT_EQ(after[0].resident, 0u);
    EXPECT_EQ(object(loader, a.id), hipTextureObject_t{});
    EXPECT_EQ(object(loader, b.id), hipTextureObject_t{});
}

class SamplerStorageOrdering : public SamplerStorage, public testing::WithParamInterface<bool> {};

TEST_P(SamplerStorageOrdering, BlockedSourceStaysAliveAndRetirementCannotRepublish) {
    DemandTextureLoader loader(sharingOptions());
    auto source = std::make_shared<BlockedSharedSource>();
    const auto a = loader.createTexture(source, sharingDesc());
    ASSERT_TRUE(a.valid);
    queue(loader, {a.id});
    auto ticket = loader.processRequestsAsync(nullptr, loader.getDeviceContext());
    const bool entered = source->waitForRead();
    if (!entered) source->release();
    ASSERT_TRUE(entered);
    std::weak_ptr<BlockedSharedSource> weak = source;
    std::promise<void> started;
    auto startedFuture = started.get_future();
    auto retire = std::async(std::launch::async, [&] {
        started.set_value();
        if (GetParam()) loader.abort();
        else loader.unloadTexture(a.id);
    });
    startedFuture.wait();
    EXPECT_EQ(retire.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    EXPECT_FALSE(weak.expired());
    source->release();
    source.reset();
    ticket.wait();
    retire.get();
    EXPECT_EQ(loader.getResidentTextureCount(), 0u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    EXPECT_EQ(object(loader, a.id), hipTextureObject_t{});
    EXPECT_FALSE(weak.expired()) << "Unload is not registration release";
}
INSTANTIATE_TEST_SUITE_P(UnloadAndAbort, SamplerStorageOrdering, testing::Bool());

TEST_F(SamplerStorage, CancellingOneCapturedRequestPreservesItsLiveSibling) {
    DemandTextureLoader loader(sharingOptions());
    auto source = std::make_shared<BlockedSharedSource>();
    auto desc = sharingDesc(), variant = desc;
    variant.filterMode = hipFilterModeLinear;
    const auto a = loader.createTexture(source, desc), b = loader.createTexture(source, variant);
    queue(loader, {a.id, b.id});
    auto ticket = loader.processRequestsAsync(nullptr, loader.getDeviceContext());
    const bool entered = source->waitForRead();
    if (!entered) source->release();
    ASSERT_TRUE(entered);
    auto unload = std::async(std::launch::async, [&] { loader.unloadTexture(a.id); });
    EXPECT_EQ(unload.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    source->release();
    ticket.wait();
    unload.get();
    loader.launchPrepare();
    EXPECT_EQ(object(loader, a.id), hipTextureObject_t{});
    EXPECT_NE(object(loader, b.id), hipTextureObject_t{});
    EXPECT_EQ(loader.getResidentTextureCount(), 1u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 64u);
    EXPECT_EQ(source->reads, 1u);
}

class SamplerStorageRetirementFault : public SamplerStorage,
    public testing::WithParamInterface<std::tuple<bool, HipOperation, size_t>> {};

TEST_P(SamplerStorageRetirementFault, FailedFenceOrInvalidationRetainsSafeObjectsAndCharges) {
    const auto [group, operation, invocation] = GetParam();
    FaultScope faults;
    DemandTextureLoader loader(sharingOptions());
    auto source = std::make_shared<SharedSource>();
    auto desc = sharingDesc(), variant = desc;
    variant.filterMode = hipFilterModeLinear;
    const auto a = loader.createTexture(source, desc), b = loader.createTexture(source, variant);
    request(loader, {a.id, b.id}, 2);
    faults.state->fail(operation, hipErrorUnknown, invocation);
    if (group) loader.unloadAll();
    else loader.unloadTexture(a.id);
    EXPECT_EQ(loader.getLastError(), LoaderError::HipError);
    EXPECT_EQ(loader.getTotalTextureMemory(), 64u);
    EXPECT_EQ(faults.calls(HipOperation::DestroySampler), 0u);
    EXPECT_EQ(faults.calls(HipOperation::FreeArray), 0u);
    uint32_t flags = 0;
    ASSERT_EQ(hipMemcpy(&flags, loader.getDeviceContext().residentFlags, sizeof(flags),
                        hipMemcpyDeviceToHost), hipSuccess);
    for (uint32_t id : {a.id, b.id}) {
        if (flags & (1u << id))
            ASSERT_NE(object(loader, id), hipTextureObject_t{});
    }
    TextureSamplingHarness harness;
    ASSERT_EQ(harness.open(samplingModulePath()), hipSuccess);
    SamplingInput input;
    input.textureId = a.id;
    std::vector<SamplingResult> result;
    ASSERT_EQ(harness.sample(loader.getDeviceContext(), {input}, result), hipSuccess);
    loader.unloadAll();
    EXPECT_EQ(loader.getTotalTextureMemory(), 0u);
    EXPECT_EQ(faults.calls(HipOperation::DestroySampler), 2u);
    EXPECT_EQ(faults.calls(HipOperation::FreeArray), 1u);
}
INSTANTIATE_TEST_SUITE_P(RetirementFailures, SamplerStorageRetirementFault, testing::Values(
    std::make_tuple(false, HipOperation::SynchronizeConsumers, 1u),
    std::make_tuple(true, HipOperation::SynchronizeConsumers, 1u),
    std::make_tuple(false, HipOperation::InvalidateMappings, 1u),
    std::make_tuple(true, HipOperation::InvalidateMappings, 1u),
    std::make_tuple(false, HipOperation::InvalidateMappings, 2u),
    std::make_tuple(true, HipOperation::InvalidateMappings, 2u),
    std::make_tuple(false, HipOperation::InvalidateMappings, 3u),
    std::make_tuple(true, HipOperation::InvalidateMappings, 3u)));

class SamplerStoragePublicationFault : public SamplerStorage, public testing::WithParamInterface<size_t> {};

TEST_P(SamplerStoragePublicationFault, PartialPublicationNeverExposesInvalidObjectAndCanRetry) {
    FaultScope faults;
    DemandTextureLoader loader(sharingOptions());
    auto source = std::make_shared<SharedSource>();
    auto desc = sharingDesc(), variant = desc;
    variant.filterMode = hipFilterModeLinear;
    const auto a = loader.createTexture(source, desc), b = loader.createTexture(source, variant);
    queue(loader, {a.id, b.id});
    ASSERT_EQ(loader.processRequests(nullptr, loader.getDeviceContext()), 2u);
    faults.state->fail(HipOperation::PublishMappings, hipErrorUnknown, GetParam());
    loader.launchPrepare();
    EXPECT_EQ(loader.getLastError(), LoaderError::HipError);
    uint32_t flags = UINT32_MAX;
    ASSERT_EQ(hipMemcpy(&flags, loader.getDeviceContext().residentFlags, sizeof(flags),
                        hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(flags, 0u);
    EXPECT_EQ(loader.getTotalTextureMemory(), 64u);
    loader.launchPrepare();
    expectShared(loader, a.id, b.id, false);
    ASSERT_EQ(hipMemcpy(&flags, loader.getDeviceContext().residentFlags, sizeof(flags),
                        hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(flags & 3u, 3u);
}
INSTANTIATE_TEST_SUITE_P(PublicationFailures, SamplerStoragePublicationFault, testing::Values(1u, 2u, 3u));

} // namespace
} } // namespace hip_demand::test
