// SPDX-License-Identifier: MIT
#include "TestUtils.h"
#include "ImageDataTestUtils.h"
#include "TextureSamplingHarness.h"
#include <cmath>
#include <iomanip>

namespace hip_demand { namespace test {
namespace {

// Fixed before device execution. Normalized bytes and authored float constants
// allow 1e-6 absolute error, including exactly representable spatial/LOD weights.
constexpr float PixelTolerance = 1e-6f;
// Fractional gradient LOD may incur approximate log2 and fixed-point mip weights.
// Allow two 8-bit fractional quantizations, scaled by adjacent-level contrast.
constexpr float GradientWeightTolerance = 1.f / 128;

std::array<float, 4> components(float4 value) {
    return {value.x, value.y, value.z, value.w};
}

std::array<float, 4> sourcePixel(const TypedImageSource& source, unsigned int level, size_t pixel) {
    const auto& bytes = source.mipPixels.empty() ? source.pixels : source.mipPixels.at(level);
    std::array<float, 4> result{};
    for (size_t channel = 0; channel < 4; ++channel)
        result[channel] = source.info.format == HIP_AD_FORMAT_FLOAT
            ? internal::readChannel<float>(bytes.data() + (pixel * 4 + channel) * sizeof(float))
            : bytes.at(pixel * 4 + channel) / 255.f;
    return result;
}

void expectSample(const SamplingResult& result, const std::array<float, 4>& expected,
                  const std::array<float, 4>& tolerance = {PixelTolerance, PixelTolerance,
                                                          PixelTolerance, PixelTolerance}) {
    EXPECT_EQ(result.resident, 1u);
    const auto actual = components(result.value);
    for (size_t channel = 0; channel < 4; ++channel) {
        EXPECT_TRUE(std::isfinite(actual[channel])) << "channel=" << channel;
        EXPECT_NEAR(actual[channel], expected[channel], tolerance[channel]) << "channel=" << channel;
    }
}

SamplingInput inputFor(uint32_t id, SamplingPath path, float u, float v, float lod = 0) {
    SamplingInput input;
    input.textureId = id;
    input.path = path;
    input.u = u;
    input.v = v;
    input.lod = lod;
    return input;
}

struct NativeTexture {
    hipTextureObject_t object{};
    TextureObject* table = nullptr;
    ~NativeTexture() {
        if (table) EXPECT_EQ(hipFree(table), hipSuccess);
        if (object) EXPECT_EQ(hipDestroyTextureObject(object), hipSuccess);
    }
};

class LegacyTextureSamplingTest : public LoaderTestFixture, public testing::WithParamInterface<bool> {
protected:
    void SetUp() override {
        LoaderTestFixture::SetUp();
        if (HasFatalFailure())
            return;
        std::filesystem::path path;
        ASSERT_NO_THROW(path = samplingModulePath());
        std::cout << "Sampling module=" << path.string() << " pixel tolerance=" << PixelTolerance
                  << " fractional gradient weight tolerance=" << GradientWeightTolerance << '\n';
        ASSERT_EQ(harness_.open(path), hipSuccess) << "Required HIP sampling module could not be loaded";
    }
    void TearDown() override {
        EXPECT_EQ(harness_.close(), hipSuccess);
        EXPECT_TRUE(harness_.isClosed());
        LoaderTestFixture::TearDown();
    }
    void requestAndLoad(const std::shared_ptr<TypedImageSource>& source, const TextureDesc& desc,
                        TextureHandle& handle, hipTextureObject_t& texture) {
        handle = loader_->createTexture(source, desc);
        ASSERT_TRUE(handle.valid) << static_cast<int>(handle.error);
        ASSERT_EQ(handle.error, LoaderError::Success);
        loader_->launchPrepare();
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const auto context = loader_->getDeviceContext();
        std::vector<SamplingResult> miss;
        ASSERT_EQ(harness_.sample(context, {inputFor(handle.id, SamplingPath::Implicit, .5f, .5f)}, miss),
                  hipSuccess);
        ASSERT_EQ(miss.size(), 1u);
        EXPECT_EQ(miss[0].resident, 0u);
        EXPECT_EQ(components(miss[0].value), (std::array<float, 4>{1, 0, 1, 1}));
        uint32_t requests = 0;
        ASSERT_EQ(hipMemcpy(&requests, context.requestCount, sizeof(requests), hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(requests, 1u);
        ASSERT_EQ(loader_->processRequests(nullptr, context), 1u) << static_cast<int>(loader_->getLastError());
        loader_->launchPrepare();
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(hipMemcpy(&texture, context.textures + handle.id, sizeof(texture), hipMemcpyDeviceToHost),
                  hipSuccess);
        ASSERT_NE(texture, hipTextureObject_t{});
    }
    void verifyStorage(const TypedImageSource& source, hipTextureObject_t texture, bool mipmapped) {
        hipResourceDesc resource{};
        ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, texture), hipSuccess);
        ASSERT_EQ(resource.resType, mipmapped ? hipResourceTypeMipmappedArray : hipResourceTypeArray)
            << "Required authored multilevel storage is unavailable; base-only fallback is NOT a sampling pass";
        size_t expectedMemory = 0;
        for (unsigned int level = 0; level < source.info.numMipLevels; ++level) {
            SCOPED_TRACE(level);
            hipArray_t array{};
            if (mipmapped) {
                ASSERT_EQ(hipGetMipmappedArrayLevel(&array, resource.res.mipmap.mipmap, level), hipSuccess)
                    << "Every authored mip must exist in the real device allocation";
            } else {
                array = resource.res.array.array;
            }
            ASSERT_NE(array, nullptr);
            hipChannelFormatDesc channel{};
            hipExtent extent{};
            unsigned int flags = 0;
            ASSERT_EQ(hipArrayGetInfo(&channel, &extent, &flags, array), hipSuccess);
            const unsigned int width = std::max(1u, source.info.width >> level);
            const unsigned int height = std::max(1u, source.info.height >> level);
            EXPECT_EQ(extent.width, width);
            EXPECT_EQ(extent.height, height);
            EXPECT_EQ(channel.x, source.info.format == HIP_AD_FORMAT_FLOAT ? 32 : 8);
            EXPECT_EQ(channel.y, channel.x);
            EXPECT_EQ(channel.z, channel.x);
            EXPECT_EQ(channel.w, channel.x);
            const auto& expected = source.mipPixels.empty() ? source.pixels : source.mipPixels[level];
            const size_t rowBytes = expected.size() / height;
            std::vector<unsigned char> actual(expected.size());
            ASSERT_EQ(hipMemcpy2DFromArray(actual.data(), rowBytes, array, 0, 0, rowBytes, height,
                                           hipMemcpyDeviceToHost), hipSuccess);
            EXPECT_EQ(actual, expected) << "Authored storage must not be replaced by generated pixels";
            expectedMemory += expected.size();
        }
        EXPECT_EQ(loader_->getTotalTextureMemory(), expectedMemory);
    }
    TextureSamplingHarness harness_;
};

TEST_P(LegacyTextureSamplingTest, AuthoredMipExplicitLodAndGradient) {
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(GetParam()));
    TextureDesc desc;
    desc.generateMipmaps = true;
    desc.filterMode = hipFilterModeLinear;
    desc.mipmapFilterMode = hipFilterModeLinear; // Qualify current linear mode, not item 02.
    TextureHandle handle;
    hipTextureObject_t texture{};
    ASSERT_NO_FATAL_FAILURE(requestAndLoad(source, desc, handle, texture));
    ASSERT_NO_FATAL_FAILURE(verifyStorage(*source, texture, true));
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{0, 1, 2, 3}));
    EXPECT_EQ(source->baseColorReads, 0u);

    std::vector<SamplingInput> inputs{inputFor(handle.id, SamplingPath::Implicit, .5f, .5f)};
    for (float lod : {-2.f, 0.f, .25f, .5f, 1.f, 1.5f, 2.f, 2.5f, 3.f, 5.f})
        inputs.push_back(inputFor(handle.id, SamplingPath::Lod, .5f, .5f, lod));
    for (float lod : {0.f, .5f, 1.f, 1.5f, 2.f, 2.5f, 3.f}) {
        auto input = inputFor(handle.id, SamplingPath::Gradient, .5f, .5f, lod);
        input.ddx = {std::exp2(lod) / source->info.width, 0};
        input.ddy = {0, std::exp2(lod) / source->info.height};
        inputs.push_back(input);
    }
    std::vector<SamplingResult> results;
    ASSERT_EQ(harness_.sample(loader_->getDeviceContext(), inputs, results), hipSuccess);
    ASSERT_EQ(results.size(), inputs.size());
    hipDeviceProp_t properties{};
    ASSERT_EQ(hipGetDeviceProperties(&properties, device_), hipSuccess);
#ifdef _WIN32
    const bool allowKnownIssue = hasKnownMipBlendIssue(true, properties.gcnArchName);
#else
    const bool allowKnownIssue = hasKnownMipBlendIssue(false, properties.gcnArchName);
#endif
    bool observedKnownIssue = false;
    for (size_t index = 0; index < inputs.size(); ++index) {
        const auto& input = inputs[index];
        SCOPED_TRACE(::testing::Message() << "path=" << static_cast<int>(input.path) << " lod=" << input.lod);
        const float lod = std::clamp(input.lod, 0.f, 3.f);
        const auto lower = static_cast<unsigned int>(std::floor(lod));
        const auto upper = std::min(lower + 1, 3u);
        const float fraction = lod - lower;
        const auto first = sourcePixel(*source, lower, 0);
        const auto second = sourcePixel(*source, upper, 0);
        std::array<float, 4> expected{}, tolerance{};
        for (size_t channel = 0; channel < 4; ++channel) {
            expected[channel] = first[channel] + fraction * (second[channel] - first[channel]);
            tolerance[channel] = PixelTolerance;
            if (input.path == SamplingPath::Gradient && fraction != 0)
                tolerance[channel] += GradientWeightTolerance * std::abs(second[channel] - first[channel]);
        }
        const auto comparison = compareMipBlend(components(results[index].value), first, second,
                                                 fraction, tolerance, allowKnownIssue);
        if (comparison == MipBlendComparison::KnownIncorrect) {
            EXPECT_EQ(results[index].resident, 1u);
            observedKnownIssue = true;
        } else {
            expectSample(results[index], expected, tolerance);
        }
    }
    if (observedKnownIssue && !HasFailure()) {
        GTEST_SKIP() << "Expected incorrect behavior (upstream issue pending): Windows gfx1201 native mip "
                       "blending matches clamp(1.25 * fractional LOD - 0.125, 0, 1), e.g. LOD 0.25 "
                       "uses weight 0.1875 instead of 0.25. All samples were checked; this is not a "
                       "filtering qualification pass. A corrected runtime will pass normally.";
    }
}

TEST_P(LegacyTextureSamplingTest, SingleLevelPointWrapPreservesSpatialPixels) {
    auto source = std::make_shared<TypedImageSource>(makeBoundaryPatternSource(GetParam()));
    TextureDesc desc;
    desc.generateMipmaps = false;
    desc.filterMode = hipFilterModePoint;
    TextureHandle handle;
    hipTextureObject_t texture{};
    ASSERT_NO_FATAL_FAILURE(requestAndLoad(source, desc, handle, texture));
    ASSERT_NO_FATAL_FAILURE(verifyStorage(*source, texture, false));
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{0}));
    std::vector<SamplingInput> inputs;
    std::vector<std::array<float, 4>> expected;
    for (SamplingPath path : {SamplingPath::Implicit, SamplingPath::Lod, SamplingPath::Gradient}) {
        for (float offset : {-1.f, 0.f, 1.f}) {
            for (unsigned int y = 0; y < 4; ++y) {
                for (unsigned int x = 0; x < 4; ++x) {
                    auto input = inputFor(handle.id, path, (x + .5f) / 4 + offset,
                                          (y + .5f) / 4 - offset, 3);
                    input.ddx = {.5f, 0};
                    input.ddy = {0, .5f};
                    inputs.push_back(input);
                    expected.push_back(sourcePixel(*source, 0, y * 4 + x));
                }
            }
        }
    }
    std::vector<SamplingResult> results;
    ASSERT_EQ(harness_.sample(loader_->getDeviceContext(), inputs, results), hipSuccess);
    ASSERT_EQ(results.size(), expected.size());
    for (size_t index = 0; index < results.size(); ++index) {
        SCOPED_TRACE(::testing::Message() << "sample=" << index << " u=" << inputs[index].u << " v=" << inputs[index].v);
        expectSample(results[index], expected[index]);
    }
}

TEST_P(LegacyTextureSamplingTest, NativeExplicitLodMatchesLoaderWrapper) {
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(GetParam()));
    TextureDesc desc;
    TextureHandle handle;
    hipTextureObject_t texture{};
    ASSERT_NO_FATAL_FAILURE(requestAndLoad(source, desc, handle, texture));
    ASSERT_NO_FATAL_FAILURE(verifyStorage(*source, texture, true));
    hipResourceDesc resource{};
    hipTextureDesc returned{};
    ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, texture), hipSuccess);
    ASSERT_EQ(hipGetTextureObjectTextureDesc(&returned, texture), hipSuccess);
    EXPECT_EQ(returned.mipmapFilterMode, hipFilterModeLinear);
    EXPECT_EQ(returned.filterMode, hipFilterModeLinear);
    EXPECT_EQ(returned.normalizedCoords, 1);
    EXPECT_FLOAT_EQ(returned.minMipmapLevelClamp, 0);
    EXPECT_FLOAT_EQ(returned.maxMipmapLevelClamp, 3);
    EXPECT_FLOAT_EQ(returned.mipmapLevelBias, 0);
    std::cout << "Native comparison: resource=" << resource.resType
              << " mipFilter=" << returned.mipmapFilterMode
              << " maxAnisotropy=" << returned.maxAnisotropy << '\n';

    NativeTexture native;
    hipTextureDesc sampler{};
    sampler.addressMode[0] = sampler.addressMode[1] = hipAddressModeWrap;
    sampler.filterMode = sampler.mipmapFilterMode = hipFilterModeLinear;
    sampler.normalizedCoords = 1;
    sampler.readMode = GetParam() ? hipReadModeElementType : hipReadModeNormalizedFloat;
    sampler.maxMipmapLevelClamp = 3;
    ASSERT_EQ(hipCreateTextureObject(&native.object, &resource, &sampler, nullptr), hipSuccess);
    ASSERT_EQ(hipMalloc(&native.table, sizeof(TextureObject)), hipSuccess);
    ASSERT_EQ(hipMemcpy(native.table, &native.object, sizeof(native.object), hipMemcpyHostToDevice), hipSuccess);
    DeviceContext context{};
    context.textures = native.table;
    std::vector<SamplingInput> inputs;
    for (float lod : {0.f, .25f, .5f, .75f, 1.f, 1.25f, 2.25f, 3.f})
        inputs.push_back(inputFor(handle.id, SamplingPath::Lod, .5f, .5f, lod));
    std::vector<SamplingResult> wrapped, direct;
    ASSERT_EQ(harness_.sample(loader_->getDeviceContext(), inputs, wrapped), hipSuccess);
    for (auto& input : inputs) {
        input.textureId = 0;
        input.path = SamplingPath::NativeLod;
    }
    ASSERT_EQ(harness_.sample(context, inputs, direct), hipSuccess);
    ASSERT_EQ(direct.size(), wrapped.size());
    for (size_t i = 0; i < direct.size(); ++i) {
        expectSample(direct[i], components(wrapped[i].value));
        std::cout << "Native LOD=" << inputs[i].lod << " red=" << direct[i].value.x
                  << " wrapped red=" << wrapped[i].value.x << '\n';
    }
}

TEST_P(LegacyTextureSamplingTest, NativeMipWeightSweepDiagnostics) {
    auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(GetParam()));
    TextureHandle handle;
    hipTextureObject_t texture{};
    ASSERT_NO_FATAL_FAILURE(requestAndLoad(source, TextureDesc{}, handle, texture));
    ASSERT_NO_FATAL_FAILURE(verifyStorage(*source, texture, true));
    hipResourceDesc resource{};
    ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, texture), hipSuccess);

    for (auto spatial : {hipFilterModePoint, hipFilterModeLinear}) {
        for (auto mip : {hipFilterModePoint, hipFilterModeLinear}) {
            for (unsigned int anisotropy : {0u, 1u, 16u}) {
                SCOPED_TRACE(::testing::Message() << "spatial=" << spatial << " mip=" << mip
                                                 << " anisotropy=" << anisotropy);
                NativeTexture native;
                hipTextureDesc sampler{};
                sampler.addressMode[0] = sampler.addressMode[1] = hipAddressModeWrap;
                sampler.filterMode = spatial;
                sampler.mipmapFilterMode = mip;
                sampler.maxAnisotropy = anisotropy;
                sampler.normalizedCoords = 1;
                sampler.readMode = GetParam() ? hipReadModeElementType : hipReadModeNormalizedFloat;
                sampler.maxMipmapLevelClamp = 3;
                const hipError_t creation = hipCreateTextureObject(&native.object, &resource, &sampler, nullptr);
                if (creation == hipErrorNotSupported) {
                    std::cout << "LOD_CONFIG format=" << (GetParam() ? "Float" : "Byte")
                              << " spatial=" << spatial << " mip=" << mip << " aniso=" << anisotropy
                              << " unsupported=" << static_cast<int>(creation) << '\n';
                    continue;
                }
                ASSERT_EQ(creation, hipSuccess);
                ASSERT_EQ(hipMalloc(&native.table, sizeof(TextureObject)), hipSuccess);
                ASSERT_EQ(hipMemcpy(native.table, &native.object, sizeof(native.object), hipMemcpyHostToDevice), hipSuccess);
                auto context = loader_->getDeviceContext();
                context.textures = native.table;
                std::vector<SamplingResult> descriptor;
                ASSERT_EQ(harness_.sample(context, {inputFor(0, SamplingPath::NativeSamplerWords, 0, 0)},
                                          descriptor), hipSuccess);
                ASSERT_EQ(descriptor.size(), 1u);
                std::array<uint32_t, 4> words{};
                std::memcpy(words.data(), &descriptor[0].value, sizeof(words));
                ASSERT_EQ(harness_.sample(context, {inputFor(0, SamplingPath::NativeImageControlWord, 0, 0)},
                                          descriptor), hipSuccess);
                ASSERT_EQ(descriptor.size(), 1u);
                uint32_t imageWord5 = 0;
                std::memcpy(&imageWord5, &descriptor[0].value.x, sizeof(imageWord5));
                std::cout << "LOD_SAMPLER format=" << (GetParam() ? "Float" : "Byte")
                          << " spatial=" << spatial << " mip=" << mip << " aniso=" << anisotropy
                          << " words=" << std::hex << words[0] << ',' << words[1] << ','
                          << words[2] << ',' << words[3] << " imageWord5=" << imageWord5 << std::dec << '\n';
                std::vector<SamplingInput> inputs;
                for (SamplingPath path : {SamplingPath::NativeLod, SamplingPath::Gradient}) {
                    for (unsigned int step = 0; step <= 48; ++step) {
                        const float lod = step / 16.f;
                        auto input = inputFor(0, path, .5f, .5f, lod);
                        input.ddx = {std::exp2(lod) / source->info.width, 0};
                        input.ddy = {0, std::exp2(lod) / source->info.height};
                        inputs.push_back(input);
                    }
                }
                std::vector<SamplingResult> results;
                ASSERT_EQ(harness_.sample(context, inputs, results), hipSuccess);
                ASSERT_EQ(results.size(), inputs.size());
                for (size_t i = 0; i < inputs.size(); ++i) {
                    const float lod = inputs[i].lod;
                    const unsigned int lower = static_cast<unsigned int>(lod);
                    const unsigned int upper = std::min(lower + 1, 3u);
                    const auto first = sourcePixel(*source, lower, 0);
                    const auto second = sourcePixel(*source, upper, 0);
                    const float fraction = lod - lower;
                    EXPECT_EQ(results[i].resident, 1u);
                    for (float channel : components(results[i].value))
                        EXPECT_TRUE(std::isfinite(channel));
                    if (fraction == 0)
                        expectSample(results[i], first);
                    const float weight = lower == upper ? 0 :
                        (results[i].value.x - first[0]) / (second[0] - first[0]);
                    std::cout << std::setprecision(9) << "LOD_SWEEP format=" << (GetParam() ? "Float" : "Byte")
                              << " spatial=" << spatial << " mip=" << mip << " aniso=" << anisotropy
                              << " path=" << (inputs[i].path == SamplingPath::NativeLod ? "Lod" : "Grad")
                              << " lod=" << lod << " red=" << results[i].value.x << " weight=" << weight
                              << " linearError=" << results[i].value.x -
                                  (first[0] + fraction * (second[0] - first[0])) << '\n';
                }
            }
        }
    }
}

std::array<float, 4> wrappedBilinear(const TypedImageSource& source, float u, float v) {
    const float x = u * 4 - .5f, y = v * 4 - .5f;
    const int ix = static_cast<int>(std::floor(x)), iy = static_cast<int>(std::floor(y));
    const float fx = x - ix, fy = y - iy;
    const auto wrap = [](int index) { return static_cast<unsigned int>((index % 4 + 4) % 4); };
    const auto a = sourcePixel(source, 0, wrap(iy) * 4 + wrap(ix));
    const auto b = sourcePixel(source, 0, wrap(iy) * 4 + wrap(ix + 1));
    const auto c = sourcePixel(source, 0, wrap(iy + 1) * 4 + wrap(ix));
    const auto d = sourcePixel(source, 0, wrap(iy + 1) * 4 + wrap(ix + 1));
    std::array<float, 4> result{};
    for (size_t channel = 0; channel < 4; ++channel)
        result[channel] = (1-fy) * ((1-fx)*a[channel] + fx*b[channel]) +
                          fy * ((1-fx)*c[channel] + fx*d[channel]);
    return result;
}

TEST_P(LegacyTextureSamplingTest, SingleLevelLinearWrapPreservesBoundaryInterpolation) {
    auto source = std::make_shared<TypedImageSource>(makeBoundaryPatternSource(GetParam()));
    TextureDesc desc;
    desc.generateMipmaps = false;
    desc.filterMode = hipFilterModeLinear;
    TextureHandle handle;
    hipTextureObject_t texture{};
    ASSERT_NO_FATAL_FAILURE(requestAndLoad(source, desc, handle, texture));
    ASSERT_NO_FATAL_FAILURE(verifyStorage(*source, texture, false));
    const std::array<std::array<float, 2>, 12> coordinates{{
        {0, .125f}, {1, .125f}, {.125f, 0}, {.125f, 1}, {0, 0}, {1, 1},
        {-.0625f, .375f}, {1.0625f, .375f}, {.375f, -.0625f}, {.375f, 1.0625f},
        {.1875f, .375f}, {.625f, .5625f}
    }};
    std::vector<SamplingInput> inputs;
    for (SamplingPath path : {SamplingPath::Implicit, SamplingPath::Lod, SamplingPath::Gradient}) {
        for (const auto& uv : coordinates) {
            auto input = inputFor(handle.id, path, uv[0], uv[1], 2.5f);
            input.ddx = {.5f, 0};
            input.ddy = {0, .5f};
            inputs.push_back(input);
        }
    }
    std::vector<SamplingResult> results;
    ASSERT_EQ(harness_.sample(loader_->getDeviceContext(), inputs, results), hipSuccess);
    ASSERT_EQ(results.size(), inputs.size());
    for (size_t index = 0; index < results.size(); ++index) {
        SCOPED_TRACE(::testing::Message() << "path=" << static_cast<int>(inputs[index].path)
                                        << " u=" << inputs[index].u << " v=" << inputs[index].v);
        expectSample(results[index], wrappedBilinear(*source, inputs[index].u, inputs[index].v));
    }
}

INSTANTIATE_TEST_SUITE_P(ByteAndFloat, LegacyTextureSamplingTest, testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) { return info.param ? "Float" : "Byte"; });

class SamplingHarnessDeviceTest : public HipTestFixture {};

TEST_F(SamplingHarnessDeviceTest, SuccessfulCleanupAndReopen) {
    TextureSamplingHarness harness;
    std::filesystem::path module;
    ASSERT_NO_THROW(module = samplingModulePath());
    for (unsigned int pass = 0; pass < 2; ++pass) {
        ASSERT_EQ(harness.open(module), hipSuccess);
        EXPECT_FALSE(harness.isClosed());
        std::vector<SamplingInput> inputs;
        for (SamplingPath path : {SamplingPath::Implicit, SamplingPath::Lod, SamplingPath::Gradient,
                                  SamplingPath::RecordRequest}) {
            auto input = inputFor(UINT32_MAX, path, .5f, .5f);
            input.defaultColor = {-2, .25f, 7, .5f};
            inputs.push_back(input);
        }
        std::vector<SamplingResult> results;
        ASSERT_EQ(harness.sample(DeviceContext{}, inputs, results), hipSuccess);
        ASSERT_EQ(results.size(), inputs.size());
        for (const auto& result : results) {
            EXPECT_EQ(result.resident, 0u);
            EXPECT_EQ(components(result.value), (std::array<float, 4>{-2, .25f, 7, .5f}));
        }
        ASSERT_EQ(harness.close(), hipSuccess);
        EXPECT_TRUE(harness.isClosed());
        EXPECT_EQ(harness.close(), hipSuccess);
    }
}

TEST_F(SamplingHarnessDeviceTest, PartialSetupFailureRollsBackAndNextLaunchSucceeds) {
    TextureSamplingHarness harness;
    std::filesystem::path module;
    ASSERT_NO_THROW(module = samplingModulePath());
    // Symbol lookup follows module/stream/buffer acquisition, exercising partial setup rollback.
    EXPECT_NE(harness.open(module, "deliberatelyMissingSamplingEntryPoint"), hipSuccess);
    EXPECT_TRUE(harness.isClosed());
    EXPECT_NE(harness.open(module.parent_path() / "deliberately-missing-sampling-module.co"), hipSuccess);
    EXPECT_TRUE(harness.isClosed());
    ASSERT_EQ(harness.open(module), hipSuccess);
    std::vector<SamplingResult> results;
    ASSERT_EQ(harness.sample(DeviceContext{}, {SamplingInput{}}, results), hipSuccess);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].resident, 0u);
    EXPECT_EQ(components(results[0].value), (std::array<float, 4>{1, 0, 1, 1}));
    EXPECT_EQ(harness.close(), hipSuccess);
    EXPECT_TRUE(harness.isClosed());
}

TEST_F(SamplingHarnessDeviceTest, InvalidBatchDoesNotMutateOutputAndFixtureRemainsUsable) {
    TextureSamplingHarness harness;
    std::filesystem::path module;
    ASSERT_NO_THROW(module = samplingModulePath());
    ASSERT_EQ(harness.open(module), hipSuccess);
    std::vector<SamplingResult> results(1);
    results[0].resident = 123;
    EXPECT_EQ(harness.sample(DeviceContext{}, {}, results), hipErrorInvalidValue);
    EXPECT_EQ(harness.sample(DeviceContext{}, std::vector<SamplingInput>(TextureSamplingHarness::MaxSamples + 1),
                             results), hipErrorInvalidValue);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].resident, 123u);
    ASSERT_EQ(harness.sample(DeviceContext{}, std::vector<SamplingInput>(TextureSamplingHarness::MaxSamples),
                             results), hipSuccess);
    ASSERT_EQ(results.size(), TextureSamplingHarness::MaxSamples);
    for (const auto& result : results) {
        EXPECT_EQ(result.resident, 0u);
        EXPECT_EQ(components(result.value), (std::array<float, 4>{1, 0, 1, 1}));
    }
    ASSERT_EQ(harness.close(), hipSuccess);
    EXPECT_EQ(harness.sample(DeviceContext{}, {SamplingInput{}}, results), hipErrorInvalidValue);
    EXPECT_TRUE(harness.isClosed());
}

} // namespace
} }
