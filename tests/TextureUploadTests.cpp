// SPDX-License-Identifier: MIT
#include "TestUtils.h"
#include "ImageDataTestUtils.h"
#include "TextureSamplingHarness.h"
#include <DemandLoading/DeviceContext.h>
#include <chrono>
#include <fstream>
#include <tuple>

namespace hip_demand {
namespace test {
namespace {

void loadRequestedTexture(DemandTextureLoader& loader, const TextureHandle& handle,
                            hipTextureObject_t& texture) {
    ASSERT_TRUE(handle.valid);
    auto context = loader.getDeviceContext();
    const uint32_t count = 1;
    ASSERT_EQ(hipMemcpy(context.requests, &handle.id, sizeof(handle.id), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(context.requestCount, &count, sizeof(count), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(loader.processRequests(nullptr, context), 1u);
    loader.launchPrepare(nullptr);
    ASSERT_EQ(hipMemcpy(&texture, context.textures + handle.id, sizeof(texture), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_NE(texture, hipTextureObject_t{});
}

TEST_F(LoaderTestFixture, ImageSourceFormatsUseCorrectGpuStorage) {
    size_t expectedMemory = 0;
    for (unsigned int formatIndex = 0; formatIndex < formats.size(); ++formatIndex) {
        for (unsigned int channels = 1; channels <= 4; ++channels) {
            SCOPED_TRACE(::testing::Message() << "format=" << formatIndex << " channels=" << channels);
            auto source = std::make_shared<TypedImageSource>(makeSource(formatIndex, channels));
            TextureDesc desc;
            desc.generateMipmaps = false;
            desc.filterMode = hipFilterModePoint;
            const auto handle = loader_->createTexture(source, desc);
            hipTextureObject_t texture = 0;
            loadRequestedTexture(*loader_, handle, texture);
            ASSERT_NE(texture, hipTextureObject_t{});

            hipResourceDesc resource{};
            hipTextureDesc sampler{};
            ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, texture), hipSuccess);
            ASSERT_EQ(hipGetTextureObjectTextureDesc(&sampler, texture), hipSuccess);
            ASSERT_EQ(resource.resType, hipResourceTypeArray);
            hipChannelFormatDesc channelDesc{};
            ASSERT_EQ(hipGetChannelDesc(&channelDesc, resource.res.array.array), hipSuccess);
            const bool floating = formatIndex != 0;
            EXPECT_EQ(channelDesc.x, floating ? 32 : 8);
            EXPECT_EQ(channelDesc.y, channelDesc.x);
            EXPECT_EQ(channelDesc.z, channelDesc.x);
            EXPECT_EQ(channelDesc.w, channelDesc.x);
            EXPECT_EQ(channelDesc.f, floating ? hipChannelFormatKindFloat : hipChannelFormatKindUnsigned);
            EXPECT_EQ(sampler.readMode, floating ? hipReadModeElementType : hipReadModeNormalizedFloat);

            const size_t rowBytes = 3 * 4 * (floating ? sizeof(float) : 1);
            std::vector<unsigned char> pixels(rowBytes * 2);
            ASSERT_EQ(hipMemcpy2DFromArray(pixels.data(), rowBytes, resource.res.array.array,
                                              0, 0, rowBytes, 2, hipMemcpyDeviceToHost), hipSuccess);
            for (size_t i = 0; i < 6; ++i) {
                for (size_t c = 0; c < 4; ++c) {
                    const float expected = c < channels ? formats[formatIndex].expected[(i + c) % 4]
                                                        : (c == 3 ? 1.f : 0.f);
                    const float actual = floating ? internal::readChannel<float>(pixels.data() + (i * 4 + c) * 4)
                                                  : pixels[i * 4 + c] / 255.f;
                    EXPECT_FLOAT_EQ(actual, expected) << "pixel=" << i << " channel=" << c;
                }
            }
            expectedMemory += pixels.size();
            EXPECT_EQ(loader_->getTotalTextureMemory(), expectedMemory);
        }
    }
    EXPECT_EQ(loader_->getResidentTextureCount(), 32u);
}

TEST_F(LoaderTestFixture, FloatMipUploadPreservesHdrValues) {
    auto source = std::make_shared<TypedImageSource>(makeSource(7, 4));
    source->info.width = 4;
    source->info.height = 1;
    const std::vector<float> colors = {-2, 2, 6, .5f, 2, 6, 10, .5f, 6, 10, 14, .5f, 10, 14, 18, .5f};
    source->pixels.resize(colors.size() * sizeof(float));
    std::memcpy(source->pixels.data(), colors.data(), source->pixels.size());
    TextureDesc desc;
    desc.generateMipmaps = true;
    auto handle = loader_->createTexture(source, desc);
    hipTextureObject_t texture = 0;
    loadRequestedTexture(*loader_, handle, texture);
    ASSERT_NE(texture, hipTextureObject_t{});
    hipResourceDesc resource{};
    ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, texture), hipSuccess);
    if (resource.resType == hipResourceTypeArray) {
        // Keep coverage of the supported non-mip fallback on older devices.
        EXPECT_EQ(loader_->getTotalTextureMemory(), 4u * 16);
        std::vector<float> pixels(colors.size());
        ASSERT_EQ(hipMemcpy2DFromArray(pixels.data(), 4 * 16, resource.res.array.array,
                                          0, 0, 4 * 16, 1, hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(pixels, colors);
    } else {
        ASSERT_EQ(resource.resType, hipResourceTypeMipmappedArray);
        hipArray_t level{};
        ASSERT_EQ(hipGetMipmappedArrayLevel(&level, resource.res.mipmap.mipmap, 1), hipSuccess);
        std::vector<float> pixels(8);
        ASSERT_EQ(hipMemcpy2DFromArray(pixels.data(), 2 * 16, level, 0, 0, 2 * 16, 1,
                                          hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(pixels, (std::vector<float>{0, 4, 8, .5f, 8, 12, 16, .5f}));
        EXPECT_EQ(loader_->getTotalTextureMemory(), (4u + 2u + 1u) * 16);
    }
}

TEST_F(LoaderTestFixture, NativeChannelMemoryTextureCanReloadAfterUnload) {
    for (unsigned int channels = 1; channels <= 3; ++channels) {
        SCOPED_TRACE(channels);
        auto source = makeSource(0, channels);
        TextureDesc desc;
        desc.generateMipmaps = false;
        auto handle = loader_->createTextureFromMemory(source.pixels.data(), 3, 2, channels, desc);
        for (unsigned int pass = 0; pass < 2; ++pass) {
            hipTextureObject_t texture{};
            loadRequestedTexture(*loader_, handle, texture);
            hipResourceDesc resource{};
            ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, texture), hipSuccess);
            ASSERT_EQ(resource.resType, hipResourceTypeArray);
            std::vector<unsigned char> pixels(6 * 4);
            ASSERT_EQ(hipMemcpy2DFromArray(pixels.data(), 3 * 4, resource.res.array.array,
                                              0, 0, 3 * 4, 2, hipMemcpyDeviceToHost), hipSuccess);
            for (size_t i = 0; i < 6; ++i)
                for (size_t c = 0; c < 4; ++c)
                    EXPECT_EQ(pixels[i * 4 + c], c < channels ? source.pixels[i * channels + c]
                                                             : (c == 3 ? 255 : 0));
            loader_->unloadTexture(handle.id);
            EXPECT_EQ(loader_->getTotalTextureMemory(), 0u);
        }
    }
}

TEST_F(LoaderTestFixture, SuppliedMipLevelsSurviveUploadAndLargeMipLimit) {
    auto source = std::make_shared<TypedImageSource>(makeSource(7, 4));
    source->info.width = source->info.height = 4;
    source->info.numMipLevels = 3;
    source->mipPixels = {std::vector<unsigned char>(16 * 16),
                         std::vector<unsigned char>(4 * 16), packed<float>({42, -3, .5f, .75f})};
    const auto authored = packed<float>({9, -2, .125f, .25f});
    for (size_t i = 0; i < 4; ++i)
        std::memcpy(source->mipPixels[1].data() + i * 16, authored.data(), 16);
    TextureDesc desc;
    desc.generateMipmaps = true;
    desc.maxMipLevel = std::numeric_limits<unsigned int>::max();
    auto handle = loader_->createTexture(source, desc);
    hipTextureObject_t texture{};
    loadRequestedTexture(*loader_, handle, texture);
    hipResourceDesc resource{};
    ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, texture), hipSuccess);
    if (resource.resType == hipResourceTypeArray)
        GTEST_SKIP() << "Device uses the supported non-mipmapped fallback";
    ASSERT_EQ(resource.resType, hipResourceTypeMipmappedArray);
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{0, 1, 2}));
    for (unsigned int index = 1; index < 3; ++index) {
        hipArray_t level{};
        ASSERT_EQ(hipGetMipmappedArrayLevel(&level, resource.res.mipmap.mipmap, index), hipSuccess);
        const unsigned int width = 4 >> index;
        std::vector<unsigned char> pixels(width * width * 16);
        ASSERT_EQ(hipMemcpy2DFromArray(pixels.data(), width * 16, level, 0, 0,
                                          width * 16, width, hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(pixels, source->mipPixels[index]);
    }
    EXPECT_EQ(loader_->getTotalTextureMemory(), (16u + 4u + 1u) * 16);
}

TEST_F(LoaderTestFixture, FilenameHdrAnd16BitPixelsKeepTheirPrecision) {
    const auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    for (bool hdr : {true, false}) {
#ifdef USE_OIIO
        if (!hdr)
            GTEST_SKIP() << "16-bit PGM precision check disabled pending an OIIO decoder fix: "
                           "OIIO 3.1.14.0 overflows signed integer normalization (49152 decodes as 49151).";
#endif
        const auto path = std::filesystem::temp_directory_path() /
            ("hip-demand-typed-" + suffix + (hdr ? ".hdr" : ".pgm"));
        struct RemoveFile {
            std::filesystem::path path;
            ~RemoveFile() { std::error_code error; std::filesystem::remove(path, error); }
        } remove{path};
        {
            std::ofstream file(path, std::ios::binary);
            ASSERT_TRUE(file.good());
            if (hdr) {
                file << "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n";
                const unsigned char values[] = {128, 64, 32, 131, 64, 128, 32, 129};
                file.write(reinterpret_cast<const char*>(values), sizeof(values));
            } else {
                file << "P5\n2 1\n65535\n";
                const unsigned char values[] = {0x40, 0, 0xc0, 0};
                file.write(reinterpret_cast<const char*>(values), sizeof(values));
            }
        }
        TextureDesc desc;
        desc.generateMipmaps = false;
        auto handle = loader_->createTexture(path.string(), desc);
        hipTextureObject_t texture{};
        loadRequestedTexture(*loader_, handle, texture);
        hipResourceDesc resource{};
        ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, texture), hipSuccess);
        ASSERT_EQ(resource.resType, hipResourceTypeArray);
        hipChannelFormatDesc channelDesc{};
        ASSERT_EQ(hipGetChannelDesc(&channelDesc, resource.res.array.array), hipSuccess);
        EXPECT_EQ(channelDesc.f, hipChannelFormatKindFloat);
        std::vector<float> pixels(8);
        ASSERT_EQ(hipMemcpy2DFromArray(pixels.data(), 2 * 16, resource.res.array.array,
                                          0, 0, 2 * 16, 1, hipMemcpyDeviceToHost), hipSuccess);
        if (hdr) {
            EXPECT_EQ(pixels, (std::vector<float>{4, 2, 1, 1, .5f, 1, .25f, 1}));
        } else {
            EXPECT_FLOAT_EQ(pixels[0], 16384.f / 65535);
            EXPECT_FLOAT_EQ(pixels[4], 49152.f / 65535);
            EXPECT_FLOAT_EQ(pixels[3], 1);
            EXPECT_FLOAT_EQ(pixels[7], 1);
        }
    }
}

std::vector<internal::ImageData> expectedMipChain(const TypedImageSource& source, bool srgb) {
    std::vector<internal::ImageData> chain;
    const unsigned int count = calculateNumMipLevels(source.info.width, source.info.height);
    for (unsigned int level = 0; level < count; ++level) {
        if (level < source.info.numMipLevels) {
            TextureInfo info = source.info;
            info.width = std::max(1u, info.width >> level);
            info.height = std::max(1u, info.height >> level);
            const auto& pixels = source.mipPixels.empty() ? source.pixels : source.mipPixels.at(level);
            chain.push_back(internal::decodeImagePixels(pixels.data(), info));
            if (srgb)
                internal::linearizeFloatSRGB(chain.back());
        } else {
            chain.push_back(internal::downsampleImage(chain.back(), srgb));
        }
    }
    return chain;
}

class MipRoundingUploadTest : public LoaderTestFixture,
                              public ::testing::WithParamInterface<std::tuple<bool, bool>> {
protected:
    void SetUp() override {
        LoaderTestFixture::SetUp();
        if (HasFatalFailure())
            return;
        ASSERT_EQ(harness_.open(samplingModulePath()), hipSuccess);
    }
    void TearDown() override {
        EXPECT_EQ(harness_.close(), hipSuccess);
        LoaderTestFixture::TearDown();
    }
    TextureDesc descriptor(bool srgb = false) const {
        TextureDesc desc;
        desc.filterMode = std::get<0>(GetParam()) ? hipFilterModeLinear : hipFilterModePoint;
        desc.mipmapFilterMode = std::get<1>(GetParam()) ? hipFilterModeLinear : hipFilterModePoint;
        desc.addressMode[0] = desc.addressMode[1] = hipAddressModeClamp;
        desc.sRGB = srgb;
        return desc;
    }
    void verifyChain(const std::vector<internal::ImageData>& chain, const TextureDesc& desc,
                     const TextureHandle& handle, hipTextureObject_t texture) {
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        hipResourceDesc resource{};
        hipTextureDesc sampler{};
        ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, texture), hipSuccess);
        ASSERT_EQ(hipGetTextureObjectTextureDesc(&sampler, texture), hipSuccess);
        ASSERT_EQ(resource.resType, chain.size() > 1 ? hipResourceTypeMipmappedArray : hipResourceTypeArray)
            << "Base-only fallback cannot pass generated-mip preservation";
        EXPECT_EQ(sampler.filterMode, desc.filterMode);
        EXPECT_EQ(sampler.sRGB, desc.sRGB && !chain.front().isFloat() ? 1 : 0);
        if (chain.size() > 1)
            EXPECT_EQ(sampler.mipmapFilterMode, desc.mipmapFilterMode);
        size_t bytes = 0;
        float maximumError = 0;
        for (unsigned int level = 0; level < chain.size(); ++level) {
            SCOPED_TRACE(level);
            const auto& expected = chain[level];
            bytes += expected.sizeBytes();
            hipArray_t array{};
            if (chain.size() > 1)
                ASSERT_EQ(hipGetMipmappedArrayLevel(&array, resource.res.mipmap.mipmap, level), hipSuccess);
            else
                array = resource.res.array.array;
            hipChannelFormatDesc channel{};
            hipExtent extent{};
            unsigned int flags = 0;
            ASSERT_EQ(hipArrayGetInfo(&channel, &extent, &flags, array), hipSuccess);
            EXPECT_EQ(extent.width, expected.width);
            EXPECT_EQ(extent.height, expected.height);
            EXPECT_EQ(channel.f, expected.isFloat() ? hipChannelFormatKindFloat : hipChannelFormatKindUnsigned);
            EXPECT_EQ(channel.x, expected.isFloat() ? 32 : 8);
            internal::ImageData uploaded;
            uploaded.reset(expected.width, expected.height, expected.isFloat());
            void* destination = uploaded.isFloat() ? static_cast<void*>(uploaded.floats.data())
                                                   : static_cast<void*>(uploaded.bytes.data());
            ASSERT_EQ(hipMemcpy2DFromArray(destination, expected.rowBytes(), array, 0, 0,
                                           expected.rowBytes(), expected.height, hipMemcpyDeviceToHost), hipSuccess);
            EXPECT_EQ(uploaded.bytes, expected.bytes);
            EXPECT_EQ(uploaded.floats, expected.floats);

            std::vector<SamplingInput> inputs;
            for (unsigned int y = 0; y < expected.height; ++y) {
                for (unsigned int x = 0; x < expected.width; ++x) {
                    SamplingInput input;
                    input.textureId = handle.id;
                    input.path = SamplingPath::Lod;
                    input.lod = static_cast<float>(level);
                    input.u = (x + .5f) / expected.width;
                    input.v = (y + .5f) / expected.height;
                    inputs.push_back(input);
                }
            }
            ASSERT_LE(inputs.size(), TextureSamplingHarness::MaxSamples);
            std::vector<SamplingResult> results;
            ASSERT_EQ(harness_.sample(loader_->getDeviceContext(), inputs, results), hipSuccess);
            ASSERT_EQ(results.size(), inputs.size());
            for (size_t pixel = 0; pixel < results.size(); ++pixel) {
                EXPECT_EQ(results[pixel].resident, 1u);
                const auto& value = results[pixel].value;
                const std::array<float, 4> actual{value.x, value.y, value.z, value.w};
                for (unsigned int c = 0; c < 4; ++c) {
                    float reference = expected.isFloat() ? expected.floats[pixel * 4 + c]
                                                        : expected.bytes[pixel * 4 + c] / 255.f;
                    const bool hardwareSRGB = desc.sRGB && !expected.isFloat() && c < 3;
                    if (hardwareSRGB)
                        reference = internal::srgbToLinear(reference);
                    // Hardware sRGB conversion is approximate; alpha and all other
                    // formats retain the existing harness's 1e-6 absolute tolerance.
                    const float tolerance = hardwareSRGB ? 1.f / 512 : 1e-6f;
                    EXPECT_NEAR(actual[c], reference, tolerance) << "pixel=" << pixel << " channel=" << c;
                    maximumError = std::max(maximumError, std::abs(actual[c] - reference));
                }
            }
        }
        EXPECT_EQ(loader_->getTotalTextureMemory(), bytes);
        std::cout << "Mip rounding payload=" << bytes << " max channel error=" << maximumError << '\n';
    }
    void checkReloads(const std::shared_ptr<TypedImageSource>& source, bool srgb = false) {
        const auto desc = descriptor(srgb);
        const auto chain = expectedMipChain(*source, srgb);
        const auto handle = loader_->createTexture(source, desc);
        ASSERT_TRUE(handle.valid);
        std::vector<unsigned int> reads;
        for (unsigned int pass = 0; pass < 3; ++pass) {
            SCOPED_TRACE(pass);
            hipTextureObject_t texture{};
            ASSERT_NO_FATAL_FAILURE(loadRequestedTexture(*loader_, handle, texture));
            ASSERT_NO_FATAL_FAILURE(verifyChain(chain, desc, handle, texture));
            for (unsigned int level = 0; level < source->info.numMipLevels; ++level)
                reads.push_back(level);
            EXPECT_EQ(source->readLevels, reads);
            EXPECT_EQ(source->baseColorReads, 0u);
            loader_->unloadTexture(handle.id);
            EXPECT_EQ(loader_->getResidentTextureCount(), 0u);
            EXPECT_EQ(loader_->getTotalTextureMemory(), 0u);
        }
    }
    TextureSamplingHarness harness_;
};

TEST_P(MipRoundingUploadTest, GeneratedByteOddThinAndRecursiveChains) {
    for (const auto& shape : {std::pair{4u, 4u}, {7u, 4u}, {5u, 3u}, {9u, 1u}, {1u, 9u}, {1u, 1u}}) {
        SCOPED_TRACE(::testing::Message() << shape.first << "x" << shape.second);
        auto source = std::make_shared<TypedImageSource>(makeSource(0, 4));
        source->info.width = shape.first;
        source->info.height = shape.second;
        source->pixels.resize(shape.first * shape.second * 4);
        for (size_t i = 0; i < source->pixels.size(); ++i)
            source->pixels[i] = shape.first == 4 ? (i < 16 ? 1 : 0)
                : static_cast<unsigned char>((i * 47 + i / 4 * 13) % 256);
        ASSERT_NO_FATAL_FAILURE(checkReloads(source));
    }
}

TEST_P(MipRoundingUploadTest, AuthoredAndMixedChainsRetainTheirPixels) {
    for (bool floating : {false, true}) {
        for (bool mixed : {false, true}) {
            for (bool srgb : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "float=" << floating << " mixed=" << mixed << " srgb=" << srgb);
                auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(floating));
                if (mixed) {
                    source->info.numMipLevels = 2;
                    source->mipPixels.resize(2);
                    if (!floating) {
                        std::fill(source->mipPixels[1].begin(), source->mipPixels[1].end(), 0);
                        std::fill_n(source->mipPixels[1].begin(), 16, 1);
                    }
                }
                ASSERT_NO_FATAL_FAILURE(checkReloads(source, srgb));
            }
        }
    }
}

TEST_P(MipRoundingUploadTest, FloatHalfIntegerHDRAndSRGBPreservation) {
    for (unsigned int format = 0; format < formats.size(); ++format) {
        for (bool srgb : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "format=" << format << " srgb=" << srgb);
            auto source = std::make_shared<TypedImageSource>(makeSource(format, 4));
            ASSERT_NO_FATAL_FAILURE(checkReloads(source, srgb));
        }
    }
    auto source = std::make_shared<TypedImageSource>(makeSource(0, 4));
    source->info.width = source->info.height = 2;
    source->pixels = {0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 1, 255, 255, 255, 1};
    const auto expected = expectedMipChain(*source, true);
    ASSERT_EQ(expected.back().bytes, (std::vector<unsigned char>{188, 188, 188, 0}));
    ASSERT_NO_FATAL_FAILURE(checkReloads(source, true));
}

TEST_P(MipRoundingUploadTest, FailedSourceLevelNeverPublishesGeneratedMipsAndCanRetry) {
    for (unsigned int failedLevel : {0u, 1u, 2u}) {
        SCOPED_TRACE(failedLevel);
        auto source = std::make_shared<TypedImageSource>(makeAuthoredMipSource(false));
        source->info.numMipLevels = 3;
        source->mipPixels.resize(3);
        const auto chain = expectedMipChain(*source, false);
        const auto desc = descriptor();
        source->failReadLevel = failedLevel;
        const auto handle = loader_->createTexture(source, desc);
        ASSERT_TRUE(handle.valid);
        loader_->launchPrepare();
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const auto context = loader_->getDeviceContext();
        const uint32_t count = 1;
        ASSERT_EQ(hipMemcpy(context.requests, &handle.id, sizeof(handle.id), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(context.requestCount, &count, sizeof(count), hipMemcpyHostToDevice), hipSuccess);
        EXPECT_EQ(loader_->processRequests(nullptr, context), 0u);
        EXPECT_EQ(loader_->getLastError(), LoaderError::ImageLoadFailed);
        EXPECT_EQ(loader_->getResidentTextureCount(), 0u);
        EXPECT_EQ(loader_->getTotalTextureMemory(), 0u);
        loader_->launchPrepare();
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        hipTextureObject_t texture{};
        ASSERT_EQ(hipMemcpy(&texture, context.textures + handle.id, sizeof(texture), hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(texture, hipTextureObject_t{});
        SamplingInput input;
        input.textureId = handle.id;
        input.path = SamplingPath::Lod;
        input.lod = 3;
        std::vector<SamplingResult> results;
        ASSERT_EQ(harness_.sample(context, {input}, results), hipSuccess);
        ASSERT_EQ(results.size(), 1u);
        EXPECT_EQ(results[0].resident, 0u);
        EXPECT_EQ(results[0].value.x, input.defaultColor.x);
        EXPECT_EQ(results[0].value.y, input.defaultColor.y);
        EXPECT_EQ(results[0].value.z, input.defaultColor.z);
        EXPECT_EQ(results[0].value.w, input.defaultColor.w);
        std::vector<unsigned int> reads;
        for (unsigned int level = 0; level <= failedLevel; ++level)
            reads.push_back(level);
        EXPECT_EQ(source->readLevels, reads);
        source->failReadLevel = std::numeric_limits<unsigned int>::max();
        ASSERT_NO_FATAL_FAILURE(loadRequestedTexture(*loader_, handle, texture));
        ASSERT_NO_FATAL_FAILURE(verifyChain(chain, desc, handle, texture));
        reads.insert(reads.end(), {0, 1, 2});
        EXPECT_EQ(source->readLevels, reads);
        loader_->unloadTexture(handle.id);
        EXPECT_EQ(loader_->getTotalTextureMemory(), 0u);
    }
}

INSTANTIATE_TEST_SUITE_P(SpatialAndMip, MipRoundingUploadTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool()));

} // namespace
} // namespace test
} // namespace hip_demand
