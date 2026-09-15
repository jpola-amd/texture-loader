// SPDX-License-Identifier: MIT
#include "TestUtils.h"
#include "ImageDataTestUtils.h"
#include "TextureSamplingHarness.h"
#include <DemandLoading/Internal/HipCalls.h>
#include <cmath>
#include <iomanip>
#include <tuple>

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

class TextureSamplingTest : public LoaderTestFixture {
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
    void verifyStorage(const TypedImageSource& source, hipTextureObject_t texture, bool mipmapped,
                       unsigned int levels = 0) {
        if (levels == 0)
            levels = source.info.numMipLevels;
        hipResourceDesc resource{};
        ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, texture), hipSuccess);
        ASSERT_EQ(resource.resType, mipmapped ? hipResourceTypeMipmappedArray : hipResourceTypeArray)
            << "Required authored multilevel storage is unavailable; base-only fallback is NOT a sampling pass";
        size_t expectedMemory = 0;
        for (unsigned int level = 0; level < levels; ++level) {
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
        if (mipmapped) {
            hipArray_t extra{};
            EXPECT_NE(hipGetMipmappedArrayLevel(&extra, resource.res.mipmap.mipmap, levels), hipSuccess)
                << "The allocation must not contain levels beyond the actual chain";
        }
    }
    TextureSamplingHarness harness_;
};

class LegacyTextureSamplingTest : public TextureSamplingTest, public testing::WithParamInterface<bool> {};


/**
 * Test case for authored mipmaps with explicit LOD and gradient sampling paths.
 * This is not working as expected on Windows. 
 * The mixing betwen the authored mipmaps and the explicit LOD sampling seems to cause incorrect results.
 * At least it is not linear as expected 
 * SWDEV-608984
 * 
 */
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

std::array<float, 4> spatialPixel(const TypedImageSource& source, unsigned int level,
                                float u, float v, const TextureDesc& desc) {
    const int width = std::max(1u, source.info.width >> level);
    const int height = std::max(1u, source.info.height >> level);
    const auto address = [](int index, int size, hipTextureAddressMode mode) {
        return mode == hipAddressModeWrap ? (index % size + size) % size
                                         : std::clamp(index, 0, size - 1);
    };
    const auto pixel = [&](int x, int y) {
        return sourcePixel(source, level, address(y, height, desc.addressMode[1]) * width +
                                         address(x, width, desc.addressMode[0]));
    };
    if (desc.normalizedCoords) {
        u *= width;
        v *= height;
    }
    if (desc.filterMode == hipFilterModePoint)
        return pixel(static_cast<int>(std::floor(u)), static_cast<int>(std::floor(v)));
    const float x = u - .5f, y = v - .5f;
    const int ix = static_cast<int>(std::floor(x)), iy = static_cast<int>(std::floor(y));
    const float fx = x - ix, fy = y - iy;
    const auto a = pixel(ix, iy), b = pixel(ix + 1, iy);
    const auto c = pixel(ix, iy + 1), d = pixel(ix + 1, iy + 1);
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
        expectSample(results[index], spatialPixel(*source, 0, inputs[index].u, inputs[index].v, desc));
    }
}

INSTANTIATE_TEST_SUITE_P(ByteAndFloat, LegacyTextureSamplingTest, testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) { return info.param ? "Float" : "Byte"; });

using FilterModes = std::tuple<hipTextureFilterMode, hipTextureFilterMode>;

class MipFilteringTest : public TextureSamplingTest, public testing::WithParamInterface<FilterModes> {
protected:
    void SetUp() override {
        faults_ = std::make_shared<internal::HipFaultState>();
        internal::setHipFaultState(faults_);
        TextureSamplingTest::SetUp();
        internal::setHipFaultState(nullptr);
        if (HasFatalFailure())
            return;
        hipDeviceProp_t properties{};
        ASSERT_EQ(hipGetDeviceProperties(&properties, device_), hipSuccess);
#ifdef _WIN32
        allowKnownIssue_ = hasKnownMipBlendIssue(true, properties.gcnArchName);
#else
        allowKnownIssue_ = hasKnownMipBlendIssue(false, properties.gcnArchName);
#endif
    }
    TextureDesc descriptor() const {
        TextureDesc desc;
        desc.filterMode = std::get<0>(GetParam());
        desc.mipmapFilterMode = std::get<1>(GetParam());
        desc.sRGB = false;
        return desc;
    }
    void verifySampler(hipTextureObject_t texture, const TextureDesc& desc, unsigned int levels,
                       bool mipmapped = true) {
        hipTextureDesc returned{};
        ASSERT_EQ(hipGetTextureObjectTextureDesc(&returned, texture), hipSuccess);
        EXPECT_EQ(returned.filterMode, desc.filterMode);
        EXPECT_EQ(returned.addressMode[0], desc.addressMode[0]);
        EXPECT_EQ(returned.addressMode[1], desc.addressMode[1]);
        EXPECT_EQ(returned.normalizedCoords, desc.normalizedCoords ? 1 : 0);
        EXPECT_EQ(returned.readMode, hipReadModeElementType);
        EXPECT_EQ(returned.sRGB, 0);
        EXPECT_EQ(returned.maxAnisotropy, 0u);
        EXPECT_FLOAT_EQ(returned.mipmapLevelBias, 0);
        if (mipmapped) {
            EXPECT_EQ(returned.mipmapFilterMode, desc.mipmapFilterMode);
            EXPECT_FLOAT_EQ(returned.minMipmapLevelClamp, 0);
            EXPECT_FLOAT_EQ(returned.maxMipmapLevelClamp, float(levels - 1));
        }
    }
    void checkSamples(const TypedImageSource& source, const TextureDesc& desc,
                      const std::vector<SamplingInput>& inputs, const std::vector<SamplingResult>& results,
                      unsigned int levels = 5) {
        ASSERT_EQ(results.size(), inputs.size());
        float maxError = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            SCOPED_TRACE(testing::Message() << "sample=" << i << " size=" << source.info.width << "x"
                << source.info.height << " path=" << static_cast<int>(input.path) << " lod=" << input.lod
                << " uv=" << input.u << "," << input.v);
            const float lod = std::clamp(input.lod, 0.f, float(levels - 1));
            const bool point = desc.mipmapFilterMode == hipFilterModePoint;
            const auto lower = static_cast<unsigned int>(std::floor(lod + (point ? .5f : 0)));
            const auto upper = point ? lower : std::min(lower + 1, levels - 1);
            const float fraction = point ? 0 : lod - lower;
            const auto first = spatialPixel(source, lower, input.u, input.v, desc);
            const auto second = spatialPixel(source, upper, input.u, input.v, desc);
            std::array<float, 4> expected{}, tolerance{};
            const auto actual = components(results[i].value);
            for (size_t c = 0; c < 4; ++c) {
                expected[c] = first[c] + fraction * (second[c] - first[c]);
                tolerance[c] = PixelTolerance;
                if (input.path == SamplingPath::Gradient && fraction != 0)
                    tolerance[c] += GradientWeightTolerance * std::abs(second[c] - first[c]);
                maxError = std::max(maxError, std::abs(actual[c] - expected[c]));
            }
            // Known native Windows gfx1201 linear-mip defect (SWDEV-608984).
            // As in the legacy mip test, only the measured curve is eligible;
            // unexpected errors still fail and a corrected runtime passes.
            const auto comparison = point ? MipBlendComparison::Unexpected :
                compareMipBlend(actual, first, second, fraction, tolerance, allowKnownIssue_);
            if (comparison == MipBlendComparison::KnownIncorrect) {
                EXPECT_EQ(results[i].resident, 1u);
                observedKnownIssue_ = true;
            } else {
                expectSample(results[i], expected, tolerance);
            }
        }
        std::cout << "Mip filtering samples=" << inputs.size() << " size=" << source.info.width << "x"
                  << source.info.height << " spatial=" << desc.filterMode << " mip=" << desc.mipmapFilterMode
                  << " max absolute channel error=" << maxError << '\n';
    }
    void reportKnownIssue() {
        if (observedKnownIssue_ && !HasFailure()) {
            GTEST_SKIP() << "Expected incorrect behavior (upstream issue pending, SWDEV-608984): Windows "
                           "gfx1201 native linear-mip blending matches clamp(1.25 * fractional LOD - "
                           "0.125, 0, 1), e.g. LOD 0.25 uses weight 0.1875 instead of 0.25. All samples "
                           "were checked; this is not a filtering qualification pass.";
        }
    }
    std::shared_ptr<internal::HipFaultState> faults_;
    bool allowKnownIssue_ = false;
    bool observedKnownIssue_ = false;
};

TEST_P(MipFilteringTest, GeneratedByteRoundingKeepsSpatialAndMipModesIndependent) {
    auto source = std::make_shared<TypedImageSource>(makeBoundaryPatternSource(false));
    TypedImageSource reference = *source;
    reference.info.numMipLevels = calculateNumMipLevels(reference.info.width, reference.info.height);
    reference.mipPixels.push_back(reference.pixels);
    auto image = internal::decodeImagePixels(source->pixels.data(), source->info);
    for (unsigned int level = 1; level < reference.info.numMipLevels; ++level) {
        image = internal::downsampleImage(image);
        reference.mipPixels.push_back(image.bytes);
    }
    const auto desc = descriptor();
    TextureHandle handle;
    hipTextureObject_t texture{};
    ASSERT_NO_FATAL_FAILURE(requestAndLoad(source, desc, handle, texture));
    ASSERT_NO_FATAL_FAILURE(verifyStorage(reference, texture, true));
    const bool point = desc.mipmapFilterMode == hipFilterModePoint;
    // Half-way linear weights are exact and distinct from the known quarter-LOD
    // runtime defect covered by the existing qualification tests below.
    const std::vector<float> lods = point ? std::vector<float>{0, .25f, .75f, 1, 1.25f, 1.75f, 2}
                                          : std::vector<float>{0, .5f, 1, 1.5f, 2};
    std::vector<SamplingInput> inputs;
    std::vector<std::array<float, 4>> expected;
    for (float lod : lods) {
        for (float u : {-.125f, .1875f, .5f, .875f, 1.125f}) {
            for (float v : {.25f, .5625f}) {
                inputs.push_back(inputFor(handle.id, SamplingPath::Lod, u, v, lod));
                const auto lower = static_cast<unsigned int>(std::floor(lod + (point ? .5f : 0)));
                const auto upper = point ? lower : std::min(lower + 1, reference.info.numMipLevels - 1);
                const float fraction = point ? 0 : lod - lower;
                const auto first = spatialPixel(reference, lower, u, v, desc);
                const auto second = spatialPixel(reference, upper, u, v, desc);
                std::array<float, 4> value{};
                for (unsigned int c = 0; c < 4; ++c)
                    value[c] = first[c] + fraction * (second[c] - first[c]);
                expected.push_back(value);
            }
        }
    }
    std::vector<SamplingResult> results;
    ASSERT_EQ(harness_.sample(loader_->getDeviceContext(), inputs, results), hipSuccess);
    ASSERT_EQ(results.size(), expected.size());
    for (size_t i = 0; i < results.size(); ++i) {
        SCOPED_TRACE(i);
        expectSample(results[i], expected[i]);
    }
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{0}));
}

TEST_P(MipFilteringTest, AuthoredChainSamplerStateAndReload) {
    for (unsigned int limit : {0u, 1u, 3u, std::numeric_limits<unsigned int>::max()}) {
        auto source = std::make_shared<TypedImageSource>(makeFilteringMipSource());
        auto desc = descriptor();
        desc.maxMipLevel = limit;
        desc.addressMode[1] = hipAddressModeClamp;
        const unsigned int levels = limit == 0 ? 5 : std::min(limit, 5u);
        uint32_t registeredId = InvalidTextureId;
        for (unsigned int pass = 0; pass < 3; ++pass) {
            SCOPED_TRACE(testing::Message() << "limit=" << limit << " pass=" << pass);
            TextureHandle handle;
            hipTextureObject_t texture{};
            ASSERT_NO_FATAL_FAILURE(requestAndLoad(source, desc, handle, texture));
            if (pass == 0)
                registeredId = handle.id;
            EXPECT_EQ(handle.id, registeredId);
            // An explicit one-level policy uses an intentional ordinary array,
            // not a capability fallback. Keep every authored-byte/pixel check.
            ASSERT_NO_FATAL_FAILURE(verifyStorage(*source, texture, levels > 1, levels));
            ASSERT_NO_FATAL_FAILURE(verifySampler(texture, desc, levels, levels > 1));
            std::vector<SamplingInput> inputs;
            for (unsigned int level = 0; level < levels; ++level)
                inputs.push_back(inputFor(handle.id, SamplingPath::Lod, .5f, .5f, float(level)));
            inputs.push_back(inputFor(handle.id, SamplingPath::Lod, .5f, .5f, -2));
            inputs.push_back(inputFor(handle.id, SamplingPath::Lod, .5f, .5f, 6));
            std::vector<SamplingResult> results;
            ASSERT_EQ(harness_.sample(loader_->getDeviceContext(), inputs, results), hipSuccess);
            ASSERT_NO_FATAL_FAILURE(checkSamples(*source, desc, inputs, results, levels));
            std::vector<unsigned int> expectedReads;
            for (unsigned int repeat = 0; repeat <= pass; ++repeat)
                for (unsigned int level = 0; level < levels; ++level)
                    expectedReads.push_back(level);
            EXPECT_EQ(source->readLevels, expectedReads);
            EXPECT_EQ(source->baseColorReads, 0u);
            loader_->unloadTexture(handle.id);
            EXPECT_EQ(loader_->getTotalTextureMemory(), 0u);
            EXPECT_EQ(loader_->getResidentTextureCount(), 0u);
        }
    }
}

TEST_P(MipFilteringTest, ExplicitSelectionTransitionsAndClamps) {
    auto source = std::make_shared<TypedImageSource>(makeFilteringMipSource());
    const auto desc = descriptor();
    TextureHandle handle;
    hipTextureObject_t texture{};
    ASSERT_NO_FATAL_FAILURE(requestAndLoad(source, desc, handle, texture));
    ASSERT_NO_FATAL_FAILURE(verifyStorage(*source, texture, true));
    std::vector<SamplingInput> inputs;
    for (float lod : {-2.f, 0.f, 4.f, 6.f})
        inputs.push_back(inputFor(handle.id, SamplingPath::Lod, .5f, .5f, lod));
    for (unsigned int level = 0; level < 4; ++level)
        for (float fraction : {0.f, .25f, .5f - 1.f/64, .5f + 1.f/64, .75f, 1.f - 1.f/64})
            inputs.push_back(inputFor(handle.id, SamplingPath::Lod, .5f, .5f, level + fraction));
    std::vector<SamplingResult> results;
    ASSERT_EQ(harness_.sample(loader_->getDeviceContext(), inputs, results), hipSuccess);
    checkSamples(*source, desc, inputs, results);
    reportKnownIssue();
}

TEST_P(MipFilteringTest, HalfwayTiesAreCharacterizedSeparately) {
    auto source = std::make_shared<TypedImageSource>(makeFilteringMipSource());
    const auto desc = descriptor();
    TextureHandle handle;
    hipTextureObject_t texture{};
    ASSERT_NO_FATAL_FAILURE(requestAndLoad(source, desc, handle, texture));
    ASSERT_NO_FATAL_FAILURE(verifyStorage(*source, texture, true));
    std::vector<SamplingInput> inputs;
    for (unsigned int level = 0; level < 4; ++level)
        inputs.push_back(inputFor(handle.id, SamplingPath::Lod, .5f, .5f, level + .5f));
    std::vector<SamplingResult> results;
    ASSERT_EQ(harness_.sample(loader_->getDeviceContext(), inputs, results), hipSuccess);
    if (desc.mipmapFilterMode == hipFilterModeLinear) {
        checkSamples(*source, desc, inputs, results);
    } else {
        ASSERT_EQ(results.size(), inputs.size());
        for (size_t i = 0; i < results.size(); ++i) {
            const auto actual = components(results[i].value);
            const auto lower = sourcePixel(*source, static_cast<unsigned int>(i), 0);
            const auto upper = sourcePixel(*source, static_cast<unsigned int>(i + 1), 0);
            const auto matches = [&](const std::array<float, 4>& expected) {
                for (size_t c = 0; c < 4; ++c)
                    if (!std::isfinite(actual[c]) || std::abs(actual[c] - expected[c]) > PixelTolerance)
                        return false;
                return true;
            };
            EXPECT_EQ(results[i].resident, 1u);
            EXPECT_TRUE(matches(lower) || matches(upper)) << "A point tie must never blend levels";
            std::cout << "Point tie lod=" << inputs[i].lod << " selected="
                      << (matches(lower) ? "lower" : matches(upper) ? "upper" : "neither") << '\n';
        }
    }
}

TEST_P(MipFilteringTest, NonconstantSpatialAndMipModesAreIndependent) {
    for (auto address : {hipAddressModeWrap, hipAddressModeClamp}) {
        auto source = std::make_shared<TypedImageSource>(makeFilteringMipSource(16, 16, true));
        auto desc = descriptor();
        desc.addressMode[0] = desc.addressMode[1] = address;
        TextureHandle handle;
        hipTextureObject_t texture{};
        ASSERT_NO_FATAL_FAILURE(requestAndLoad(source, desc, handle, texture));
        ASSERT_NO_FATAL_FAILURE(verifyStorage(*source, texture, true));
        ASSERT_NO_FATAL_FAILURE(verifySampler(texture, desc, 5));
        std::vector<SamplingInput> inputs;
        for (float lod : {0.f, .25f, .75f, 1.f, 1.25f, 1.75f, 2.f, 2.25f, 2.75f, 3.f, 4.f})
            for (const auto& uv : {std::array<float, 2>{0, 0}, {1, 1}, {-.03125f, .375f},
                                   {1.03125f, .375f}, {.1875f, .375f}, {.375f, .1875f}})
                inputs.push_back(inputFor(handle.id, SamplingPath::Lod, uv[0], uv[1], lod));
        std::vector<SamplingResult> results;
        ASSERT_EQ(harness_.sample(loader_->getDeviceContext(), inputs, results), hipSuccess);
        checkSamples(*source, desc, inputs, results);
        loader_->unloadTexture(handle.id);
        EXPECT_EQ(loader_->getTotalTextureMemory(), 0u);
    }
    reportKnownIssue();
}

TEST_P(MipFilteringTest, IsotropicGradientsMatchOriginalLodAcrossShapesAndAxes) {
    for (const auto& dimensions : {std::array<unsigned int, 2>{16, 16}, {19, 19}, {16, 8}, {19, 11},
                                   {16, 1}, {1, 16}, {19, 1}, {1, 19}}) {
        auto source = std::make_shared<TypedImageSource>(makeFilteringMipSource(dimensions[0], dimensions[1]));
        const auto desc = descriptor();
        TextureHandle handle;
        hipTextureObject_t texture{};
        ASSERT_NO_FATAL_FAILURE(requestAndLoad(source, desc, handle, texture));
        ASSERT_NO_FATAL_FAILURE(verifyStorage(*source, texture, true));
        for (unsigned int orientation = 0; orientation < 4; ++orientation) {
            SCOPED_TRACE(testing::Message() << "derivative orientation=" << orientation);
            std::vector<SamplingInput> inputs;
            for (float lod : {-2.f, 0.f, .25f, .75f, 1.f, 1.25f, 1.75f, 2.f, 2.25f,
                               2.75f, 3.f, 3.25f, 3.75f, 4.f, 6.f}) {
                for (const auto& uv : {std::array<float, 2>{0, 0}, {1, 1}, {-.125f, 1.125f}}) {
                    auto input = inputFor(handle.id, SamplingPath::Gradient, uv[0], uv[1], lod);
                    const float rho = std::exp2(lod);
                    input.ddx = {rho / dimensions[0], 0};
                    input.ddy = {0, rho / dimensions[1]};
                    if (orientation & 1)
                        std::swap(input.ddx, input.ddy);
                    if (orientation & 2) {
                        input.ddx.x = -input.ddx.x;
                        input.ddx.y = -input.ddx.y;
                        input.ddy.x = -input.ddy.x;
                        input.ddy.y = -input.ddy.y;
                    }
                    inputs.push_back(input);
                    input.path = SamplingPath::Lod;
                    inputs.push_back(input);
                }
            }
            std::vector<SamplingResult> results;
            ASSERT_EQ(harness_.sample(loader_->getDeviceContext(), inputs, results), hipSuccess);
            checkSamples(*source, desc, inputs, results);
            ASSERT_EQ(results.size(), inputs.size());
            for (size_t i = 0; i < results.size(); i += 2) {
                const float lod = std::clamp(inputs[i].lod, 0.f, 4.f);
                const auto lower = static_cast<unsigned int>(std::floor(lod));
                const auto first = sourcePixel(*source, lower, 0);
                const auto second = sourcePixel(*source, std::min(lower + 1, 4u), 0);
                std::array<float, 4> tolerance{};
                for (size_t c = 0; c < 4; ++c)
                    tolerance[c] = PixelTolerance +
                        (desc.mipmapFilterMode == hipFilterModeLinear && lod != lower
                            ? GradientWeightTolerance * std::abs(second[c] - first[c]) : 0);
                expectSample(results[i], components(results[i + 1].value), tolerance);
            }
        }
        loader_->unloadTexture(handle.id);
        EXPECT_EQ(loader_->getTotalTextureMemory(), 0u);
    }
    reportKnownIssue();
}

TEST_P(MipFilteringTest, IntentionalSingleLevelIsNotMipCapabilityFallback) {
    for (bool singleton : {false, true}) {
        for (bool normalized : {false, true}) {
            auto source = std::make_shared<TypedImageSource>(
                makeFilteringMipSource(singleton ? 1 : 16, singleton ? 1 : 16, true));
            auto desc = descriptor();
            desc.generateMipmaps = singleton;
            desc.normalizedCoords = normalized;
            desc.addressMode[0] = desc.addressMode[1] = hipAddressModeClamp;
            TextureHandle handle;
            hipTextureObject_t texture{};
            ASSERT_NO_FATAL_FAILURE(requestAndLoad(source, desc, handle, texture));
            ASSERT_NO_FATAL_FAILURE(verifyStorage(*source, texture, false, 1));
            ASSERT_NO_FATAL_FAILURE(verifySampler(texture, desc, 1, false));
            EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{0}));
            std::vector<SamplingInput> inputs;
            for (SamplingPath path : {SamplingPath::Lod, SamplingPath::Gradient})
                for (float lod : {-2.f, 0.f, .25f, .75f, 4.f, 8.f}) {
                    auto input = inputFor(handle.id, path, .375f, .1875f, lod);
                    input.ddx = {4, 0};
                    input.ddy = {0, 4};
                    inputs.push_back(input);
                }
            std::vector<SamplingResult> results;
            ASSERT_EQ(harness_.sample(loader_->getDeviceContext(), inputs, results), hipSuccess);
            checkSamples(*source, desc, inputs, results, 1);
            loader_->unloadTexture(handle.id);
            EXPECT_EQ(loader_->getTotalTextureMemory(), 0u);
        }
    }
}

TEST_P(MipFilteringTest, FailedMultilevelLoadNeverPublishesAndRetryRetainsMode) {
    using internal::HipOperation;
    for (const auto& fault : {std::pair<HipOperation, size_t>{HipOperation::AllocateMipmapped, 1},
                              {HipOperation::Upload, 1}, {HipOperation::Upload, 3},
                              {HipOperation::Upload, 5}, {HipOperation::CreateSampler, 1}}) {
        SCOPED_TRACE(testing::Message() << "operation=" << static_cast<int>(fault.first)
                                      << " invocation=" << fault.second);
        auto source = std::make_shared<TypedImageSource>(makeFilteringMipSource());
        const auto desc = descriptor();
        auto handle = loader_->createTexture(source, desc);
        ASSERT_TRUE(handle.valid);
        const auto error = fault.first == HipOperation::AllocateMipmapped ? hipErrorOutOfMemory
                                                                         : hipErrorNotSupported;
        faults_->fail(fault.first, error, fault.second);
        loader_->launchPrepare();
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        std::vector<SamplingResult> results;
        const auto miss = inputFor(handle.id, SamplingPath::Gradient, .5f, .5f, 2);
        ASSERT_EQ(harness_.sample(loader_->getDeviceContext(), {miss}, results), hipSuccess);
        ASSERT_EQ(results.size(), 1u);
        EXPECT_EQ(results[0].resident, 0u);
        EXPECT_EQ(loader_->processRequests(nullptr, loader_->getDeviceContext()), 0u);
        EXPECT_EQ(loader_->getLastError(), error == hipErrorOutOfMemory ? LoaderError::OutOfMemory
                                                                      : LoaderError::HipError);
        EXPECT_EQ(loader_->getTotalTextureMemory(), 0u);
        EXPECT_EQ(loader_->getResidentTextureCount(), 0u);
        const auto records = faults_->records();
        EXPECT_TRUE(std::any_of(records.begin(), records.end(), [&](const internal::HipCallRecord& record) {
            return record.operation == fault.first && record.error == error && record.injected;
        }));
        loader_->launchPrepare();
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        hipTextureObject_t texture{};
        ASSERT_EQ(hipMemcpy(&texture, loader_->getDeviceContext().textures + handle.id, sizeof(texture),
                            hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(texture, hipTextureObject_t{});
        ASSERT_NO_FATAL_FAILURE(requestAndLoad(source, desc, handle, texture));
        ASSERT_NO_FATAL_FAILURE(verifyStorage(*source, texture, true));
        ASSERT_NO_FATAL_FAILURE(verifySampler(texture, desc, 5));
        const auto input = inputFor(handle.id, SamplingPath::Lod, .5f, .5f,
                                    desc.mipmapFilterMode == hipFilterModePoint ? .75f : 2.f);
        ASSERT_EQ(harness_.sample(loader_->getDeviceContext(), {input}, results), hipSuccess);
        checkSamples(*source, desc, {input}, results);
        loader_->unloadTexture(handle.id);
        EXPECT_EQ(loader_->getTotalTextureMemory(), 0u);
    }
}

INSTANTIATE_TEST_SUITE_P(SpatialAndMip, MipFilteringTest,
    testing::Combine(testing::Values(hipFilterModePoint, hipFilterModeLinear),
                     testing::Values(hipFilterModePoint, hipFilterModeLinear)),
    [](const testing::TestParamInfo<FilterModes>& info) {
        return std::string(std::get<0>(info.param) == hipFilterModePoint ? "SpatialPoint" : "SpatialLinear") +
               (std::get<1>(info.param) == hipFilterModePoint ? "MipPoint" : "MipLinear");
    });

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
