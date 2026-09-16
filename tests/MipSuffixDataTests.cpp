// SPDX-License-Identifier: MIT
// Host-only whole-mip decode tests: no HIP device, stream or runtime initialization.
#include <gtest/gtest.h>
#include "DemandLoading/Internal/MipSuffixData.h"
#include "ImageDataTestUtils.h"

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <tuple>

namespace hip_demand {
namespace test {
namespace {

using contract_v1::Outcome;
using internal::ImageData;
using internal::MipSuffixReadStats;
using internal::visitMipSuffix;

class ObservedMipSource : public TypedImageSource {
public:
    explicit ObservedMipSource(TypedImageSource source)
        : TypedImageSource(std::move(source)) {}

    unsigned int opens = 0;
    unsigned int closes = 0;
    mutable unsigned int infoQueries = 0;
    mutable unsigned int openQueries = 0;
    std::function<void()> beforeOpen;
    std::function<void()> beforeIsOpen;
    std::function<void()> beforeInfo;
    std::function<void()> beforeRead;
    std::function<void()> afterRead;

    void open(TextureInfo* result) override {
        ++opens;
        if (beforeOpen) beforeOpen();
        TypedImageSource::open(result);
    }
    void close() override { ++closes; TypedImageSource::close(); }
    bool isOpen() const override {
        ++openQueries;
        if (beforeIsOpen) beforeIsOpen();
        return TypedImageSource::isOpen();
    }
    const TextureInfo& getInfo() const override {
        ++infoQueries;
        if (beforeInfo) beforeInfo();
        return TypedImageSource::getInfo();
    }
    bool readMipLevel(char* dest, unsigned int level, unsigned int width,
                      unsigned int height, hipStream_t stream) override {
        if (beforeRead) beforeRead();
        const bool result = TypedImageSource::readMipLevel(dest, level, width, height, stream);
        if (afterRead) afterRead();
        return result;
    }
};

void expectStats(const MipSuffixReadStats& stats, uint64_t peak, uint64_t native,
                 uint32_t authored, uint32_t generated) {
    EXPECT_EQ(stats.decodedPeakBytes, peak);
    EXPECT_EQ(stats.sourceBytes, native);
    EXPECT_EQ(stats.authoredReads, authored);
    EXPECT_EQ(stats.generatedLevels, generated);
}

Outcome ignoreImage(uint32_t, const ImageData&) { return Outcome::Success; }

TypedImageSource patternedSource(uint32_t width, uint32_t height, uint32_t formatIndex,
                                uint32_t channels = 4) {
    auto source = makeSource(formatIndex, channels);
    source.info.width = width;
    source.info.height = height;
    const auto& format = formats[formatIndex];
    const size_t channelBytes = getBytesPerChannel(format.format);
    source.pixels.resize(size_t{width} * height * channels * channelBytes);
    for (size_t pixel = 0; pixel < size_t{width} * height; ++pixel) {
        for (uint32_t c = 0; c < channels; ++c) {
            std::memcpy(source.pixels.data() + (pixel * channels + c) * channelBytes,
                        format.values.data() + ((pixel + c) % 4) * channelBytes, channelBytes);
        }
    }
    return source;
}

class MipSuffixAuthored : public ::testing::TestWithParam<bool> {};

TEST_P(MipSuffixAuthored, VisitsOnlyRequestedOriginalLevelsAndPreservesAuthoredPixels) {
    const bool floating = GetParam();
    ObservedMipSource source(makeAuthoredMipSource(floating));
    MipSuffixReadStats stats;
    std::vector<uint32_t> visited;
    const uint64_t stride = floating ? 16 : 4;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, false, 2, 4, 4 * stride, stats,
        [&](uint32_t level, const ImageData& image) {
            visited.push_back(level);
            EXPECT_EQ(image.width, 8u >> level);
            EXPECT_EQ(image.height, 8u >> level);
            EXPECT_EQ(image.isFloat(), floating);
            EXPECT_EQ(image.sizeBytes(), source.mipPixels[level].size());
            EXPECT_EQ(std::memcmp(image.data(), source.mipPixels[level].data(),
                                  image.sizeBytes()), 0);
            EXPECT_EQ(source.destination, image.data());
            EXPECT_EQ(source.readLevels, visited);
            return Outcome::Success;
        }), Outcome::Success);
    EXPECT_EQ(visited, (std::vector<uint32_t>{2, 3}));
    EXPECT_EQ(source.readLevels, visited);
    EXPECT_EQ(source.opens, 1u);
    EXPECT_EQ(source.closes, 0u);
    EXPECT_EQ(source.baseColorReads, 0u);
    expectStats(stats, 4 * stride, 5 * stride, 2, 0);
}

TEST_P(MipSuffixAuthored, MissingSuffixStartsAtNearestAuthoredDependency) {
    const bool floating = GetParam();
    auto fixture = makeAuthoredMipSource(floating);
    fixture.info.numMipLevels = 2;
    fixture.mipPixels.resize(2);
    ObservedMipSource source(std::move(fixture));
    MipSuffixReadStats stats;
    const uint64_t stride = floating ? 16 : 4;
    unsigned int visits = 0;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, true, 3, 4, 20 * stride, stats,
        [&](uint32_t level, const ImageData& image) {
            ++visits;
            EXPECT_EQ(level, 3u);
            EXPECT_EQ(image.width, 1u);
            EXPECT_EQ(image.height, 1u);
            const auto expected = floating ? packed(authoredFloatMipColors[1])
                                           : packed(authoredByteMipColors[1]);
            EXPECT_EQ(std::memcmp(image.data(), expected.data(), expected.size()), 0);
            EXPECT_EQ(stats.generatedLevels, 2u);
            return Outcome::Success;
        }), Outcome::Success);
    EXPECT_EQ(visits, 1u);
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{1}));
    expectStats(stats, 20 * stride, 16 * stride, 1, 2);
}

TEST_P(MipSuffixAuthored, PartialChainReadsAuthoredLevelsThenGeneratesRecursively) {
    const bool floating = GetParam();
    auto source = makeAuthoredMipSource(floating);
    source.info.numMipLevels = 2;
    source.mipPixels.resize(2);
    MipSuffixReadStats stats;
    std::vector<uint32_t> visited;
    const uint64_t stride = floating ? 16 : 4;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, true, 0, 4, 64 * stride, stats,
        [&](uint32_t level, const ImageData& image) {
            visited.push_back(level);
            const uint32_t authored = std::min(level, 1u);
            const auto expected = floating ? packed(authoredFloatMipColors[authored])
                                           : packed(authoredByteMipColors[authored]);
            for (size_t offset = 0; offset < image.sizeBytes(); offset += expected.size())
                EXPECT_EQ(std::memcmp(static_cast<const unsigned char*>(image.data()) + offset,
                                      expected.data(), expected.size()), 0);
            return Outcome::Success;
        }), Outcome::Success);
    EXPECT_EQ(visited, (std::vector<uint32_t>{0, 1, 2, 3}));
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{0, 1}));
    expectStats(stats, 64 * stride, 80 * stride, 2, 2);
}

INSTANTIATE_TEST_SUITE_P(ByteAndFloat, MipSuffixAuthored, ::testing::Bool());

TEST(MipSuffixData, PolicyLimitedChainStopsWithoutReadingCoarserAuthoredLevels) {
    auto source = makeFilteringMipSource();
    MipSuffixReadStats stats;
    std::vector<uint32_t> visited;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, false, 1, 3, 1024, stats,
        [&](uint32_t level, const ImageData&) {
            visited.push_back(level);
            return Outcome::Success;
        }), Outcome::Success);
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{1, 2}));
    EXPECT_EQ(visited, source.readLevels);
    expectStats(stats, 1024, 1280, 2, 0);
}

TEST(MipSuffixData, RecursiveByteHalfUpIncludesDiscardedIntermediateLevels) {
    auto source = patternedSource(4, 4, 0);
    std::fill(source.pixels.begin(), source.pixels.end(), 0);
    std::fill_n(source.pixels.begin(), 16, 1);
    MipSuffixReadStats stats;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, true, 2, 3, 80, stats,
        [&](uint32_t level, const ImageData& image) {
            EXPECT_EQ(level, 2u);
            EXPECT_EQ(image.bytes, (std::vector<unsigned char>{1, 1, 1, 1}));
            return Outcome::Success;
        }), Outcome::Success);
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{0}));
    expectStats(stats, 80, 64, 1, 2);
}

TEST(MipSuffixData, ByteSRGBPreservesAuthoredBytesAndFractionalAlphaTruncation) {
    auto source = patternedSource(2, 1, 0);
    source.pixels = {0, 0, 0, 1, 255, 255, 255, 0};
    MipSuffixReadStats stats;
    EXPECT_EQ(visitMipSuffix(source, source.info, true, true, 0, 2, 12, stats,
        [&](uint32_t level, const ImageData& image) {
            EXPECT_FALSE(image.isFloat());
            if (level == 0) EXPECT_EQ(image.bytes, source.pixels);
            else EXPECT_EQ(image.bytes, (std::vector<unsigned char>{188, 188, 188, 0}));
            return Outcome::Success;
        }), Outcome::Success);
    expectStats(stats, 12, 8, 1, 1);
}

TEST(MipSuffixData, FloatSRGBLinearizesDependencyOnceBeforeRecursiveGeneration) {
    auto source = patternedSource(4, 1, 7);
    const auto pixel = packed<float>({.04045f, .5f, 2.f, .25f});
    for (size_t i = 0; i < 4; ++i)
        std::memcpy(source.pixels.data() + i * pixel.size(), pixel.data(), pixel.size());
    MipSuffixReadStats stats;
    EXPECT_EQ(visitMipSuffix(source, source.info, true, true, 2, 3, 96, stats,
        [&](uint32_t level, const ImageData& image) {
            EXPECT_EQ(level, 2u);
            EXPECT_EQ(image.floats.size(), 4u);
            EXPECT_FLOAT_EQ(image.floats[0], .04045f / 12.92f);
            EXPECT_NEAR(image.floats[1], .21404114f, 1e-6f);
            EXPECT_NEAR(image.floats[2], 4.9538474f, 1e-6f);
            EXPECT_FLOAT_EQ(image.floats[3], .25f);
            return Outcome::Success;
        }), Outcome::Success);
    expectStats(stats, 96, 64, 1, 2);
}

TEST(MipSuffixData, FloatSRGBTransformsEveryIndependentAuthoredMip) {
    auto source = makeAuthoredMipSource(true);
    MipSuffixReadStats stats;
    EXPECT_EQ(visitMipSuffix(source, source.info, true, false, 1, 4, 256, stats,
        [&](uint32_t level, const ImageData& image) {
            for (size_t pixel = 0; pixel < image.floats.size(); pixel += 4) {
                for (uint32_t channel = 0; channel < 3; ++channel)
                    EXPECT_FLOAT_EQ(image.floats[pixel + channel],
                        internal::srgbToLinear(authoredFloatMipColors[level][channel]));
                EXPECT_FLOAT_EQ(image.floats[pixel + 3], authoredFloatMipColors[level][3]);
            }
            return Outcome::Success;
        }), Outcome::Success);
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{1, 2, 3}));
    expectStats(stats, 256, 336, 3, 0);
}

class MipSuffixFormat : public ::testing::TestWithParam<std::tuple<unsigned int, unsigned int>> {};

TEST_P(MipSuffixFormat, PreservesNativeChannelsAndGeneratedRepresentationWithinExactBound) {
    const auto [formatIndex, channels] = GetParam();
    auto source = makeSource(formatIndex, channels);
    const bool floating = formatIndex != 0;
    auto expected = internal::decodeImagePixels(source.pixels.data(), source.info);
    const auto generated = internal::downsampleImage(expected);
    const uint64_t decoded = 6 * (floating ? 16 : 4);
    const uint64_t native = 6 * channels * getBytesPerChannel(source.info.format);
    const bool direct = channels == 4 && (formatIndex == 0 || formatIndex == 7);
    const uint64_t peak = std::max(decoded + (direct ? 0 : native), decoded + (floating ? 16 : 4));
    MipSuffixReadStats stats;
    uint32_t visits = 0;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, true, 0, 2, peak, stats,
        [&](uint32_t level, const ImageData& image) {
            ++visits;
            EXPECT_EQ(image.isFloat(), floating);
            if (level == 0) {
                for (size_t pixel = 0; pixel < 6; ++pixel) {
                    for (size_t channel = 0; channel < 4; ++channel) {
                        const float value = channel < channels
                            ? formats[formatIndex].expected[(pixel + channel) % 4]
                            : (channel == 3 ? 1.f : 0.f);
                        EXPECT_FLOAT_EQ(floating ? image.floats[pixel * 4 + channel]
                                                : image.bytes[pixel * 4 + channel] / 255.f, value);
                    }
                }
                if (direct) EXPECT_EQ(source.destination, image.data());
            } else {
                EXPECT_EQ(image.bytes, generated.bytes);
                EXPECT_EQ(image.floats, generated.floats);
            }
            return Outcome::Success;
        }), Outcome::Success);
    EXPECT_EQ(visits, 2u);
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{0}));
    expectStats(stats, peak, native, 1, 1);
}

INSTANTIATE_TEST_SUITE_P(AllFormatsAndChannels, MipSuffixFormat,
    ::testing::Combine(::testing::Range(0u, 8u), ::testing::Range(1u, 5u)));

class MipSuffixDimensions
    : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, bool>> {};

TEST_P(MipSuffixDimensions, OddThinAndSingletonSuffixMatchesRecursiveAreaFilter) {
    const auto [width, height, floating] = GetParam();
    auto source = patternedSource(width, height, floating ? 7 : 0);
    const uint32_t levels = contract_v1::fullMipCount(width, height);
    const uint32_t firstMip = levels > 1 ? 1 : 0;
    ImageData expected = internal::decodeImagePixels(source.pixels.data(), source.info);
    const uint64_t native = expected.sizeBytes();
    const uint64_t next = levels > 1 ? uint64_t{std::max(1u, width / 2)} *
        std::max(1u, height / 2) * (floating ? 16 : 4) : 0;
    MipSuffixReadStats stats;
    std::vector<uint32_t> visited;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, true, firstMip, levels,
                            native + next, stats,
        [&](uint32_t level, const ImageData& image) {
            visited.push_back(level);
            if (level) expected = internal::downsampleImage(expected);
            EXPECT_EQ(image.width, std::max(1u, width >> level));
            EXPECT_EQ(image.height, std::max(1u, height >> level));
            EXPECT_EQ(image.bytes, expected.bytes);
            EXPECT_EQ(image.floats, expected.floats);
            return Outcome::Success;
        }), Outcome::Success);
    EXPECT_EQ(visited.size(), levels - firstMip);
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{0}));
    expectStats(stats, native + next, native, 1, levels - 1);
}

INSTANTIATE_TEST_SUITE_P(RecursiveShapes, MipSuffixDimensions, ::testing::Values(
    std::make_tuple(5u, 3u, false), std::make_tuple(5u, 3u, true),
    std::make_tuple(3u, 5u, false), std::make_tuple(3u, 5u, true),
    std::make_tuple(1u, 7u, false), std::make_tuple(1u, 7u, true),
    std::make_tuple(7u, 1u, false), std::make_tuple(7u, 1u, true),
    std::make_tuple(1u, 2u, false), std::make_tuple(2u, 1u, true),
    std::make_tuple(1u, 1u, false), std::make_tuple(1u, 1u, true)));

TEST(MipSuffixData, OddThinLastPixelContributesToGeneratedFloatMip) {
    for (bool vertical : {false, true}) {
        auto source = patternedSource(vertical ? 1 : 3, vertical ? 3 : 1, 7);
        const std::array<float, 12> values{0, 0, 0, 1, 0, 0, 0, 1, 99, 99, 99, 1};
        std::memcpy(source.pixels.data(), values.data(), sizeof(values));
        MipSuffixReadStats stats;
        EXPECT_EQ(visitMipSuffix(source, source.info, false, true, 1, 2, 64, stats,
            [](uint32_t, const ImageData& image) {
                EXPECT_EQ(image.floats, (std::vector<float>{33, 33, 33, 1}));
                return Outcome::Success;
            }), Outcome::Success);
        expectStats(stats, 64, 48, 1, 1);
    }
}

struct DecodeLimitCase {
    uint32_t width, height, formatIndex, channels, firstMip, levels;
    uint64_t peak, native;
};

class MipSuffixLimit : public ::testing::TestWithParam<DecodeLimitCase> {};

TEST_P(MipSuffixLimit, ExactLimitSucceedsAndOneByteLessRejectsBeforeAnySourceAccess) {
    const auto p = GetParam();
    ObservedMipSource source(patternedSource(p.width, p.height, p.formatIndex, p.channels));
    MipSuffixReadStats stats{123, 456, 7, 8};
    uint32_t visits = 0;
    const auto visitor = [&](uint32_t, const ImageData&) { ++visits; return Outcome::Success; };
    EXPECT_EQ(visitMipSuffix(source, source.info, false, true, p.firstMip, p.levels,
                            p.peak - 1, stats, visitor), Outcome::DemandTooLarge);
    EXPECT_EQ(source.opens, 0u);
    EXPECT_EQ(source.openQueries, 0u);
    EXPECT_EQ(source.infoQueries, 0u);
    EXPECT_EQ(source.reads, 0u);
    EXPECT_EQ(visits, 0u);
    expectStats(stats, 0, 0, 0, 0);
    EXPECT_EQ(visitMipSuffix(source, source.info, false, true, p.firstMip, p.levels,
                            p.peak, stats, visitor), Outcome::Success);
    EXPECT_EQ(visits, p.levels - p.firstMip);
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{0}));
    expectStats(stats, p.peak, p.native, 1, p.levels - 1);
}

INSTANTIATE_TEST_SUITE_P(NativeAndConversionPeaks, MipSuffixLimit, ::testing::Values(
    DecodeLimitCase{4, 4, 0, 4, 2, 3, 80, 64},
    DecodeLimitCase{4, 4, 7, 4, 2, 3, 320, 256},
    DecodeLimitCase{4, 4, 6, 4, 2, 3, 384, 128},
    DecodeLimitCase{4, 4, 0, 3, 2, 3, 112, 48},
    DecodeLimitCase{4, 4, 7, 3, 2, 3, 448, 192},
    DecodeLimitCase{4, 4, 1, 4, 2, 3, 320, 64},
    DecodeLimitCase{5, 1, 0, 1, 2, 3, 28, 5},
    DecodeLimitCase{5, 3, 0, 4, 2, 3, 68, 60},
    DecodeLimitCase{1, 1, 0, 4, 0, 1, 4, 4},
    DecodeLimitCase{1, 1, 6, 1, 0, 1, 18, 2}));

TEST(MipSuffixData, IndependentAuthoredReadsReleasePreviousPayloadBeforeNextAllocation) {
    ObservedMipSource source(makeAuthoredMipSource(true));
    MipSuffixReadStats stats;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, false, 1, 4, 255, stats, ignoreImage),
              Outcome::DemandTooLarge);
    EXPECT_EQ(source.openQueries, 0u);
    expectStats(stats, 0, 0, 0, 0);
    EXPECT_EQ(visitMipSuffix(source, source.info, false, false, 1, 4, 256, stats, ignoreImage),
              Outcome::Success);
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{1, 2, 3}));
    expectStats(stats, 256, 336, 3, 0);
}

TEST(MipSuffixData, ZeroBudgetDoesNotOpenOrReadSource) {
    ObservedMipSource source(patternedSource(1, 1, 0));
    MipSuffixReadStats stats;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, false, 0, 1, 0, stats, ignoreImage),
              Outcome::DemandTooLarge);
    EXPECT_EQ(source.opens, 0u);
    EXPECT_EQ(source.openQueries, 0u);
    EXPECT_EQ(source.reads, 0u);
    expectStats(stats, 0, 0, 0, 0);
}

TEST(MipSuffixData, MissingAuthoredSuffixWithoutGenerationIsUnsupportedBeforeSourceAccess) {
    for (uint32_t authored : {1u, 2u}) {
        for (uint32_t first : {0u, 1u, 3u}) {
            for (uint64_t budget : {uint64_t{0}, UINT64_MAX}) {
                ObservedMipSource source(makeAuthoredMipSource(false));
                source.info.numMipLevels = authored;
                MipSuffixReadStats stats{123, 456, 7, 8};
                EXPECT_EQ(visitMipSuffix(source, source.info, false, false, first, 4, budget, stats,
                    [](uint32_t, const ImageData&) {
                        ADD_FAILURE() << "Unsupported generation policy must not visit images";
                        return Outcome::Success;
                    }), Outcome::Unsupported);
                EXPECT_EQ(source.opens, 0u);
                EXPECT_EQ(source.openQueries, 0u);
                EXPECT_EQ(source.infoQueries, 0u);
                EXPECT_EQ(source.reads, 0u);
                expectStats(stats, 0, 0, 0, 0);
            }
        }
    }
}

enum class InvalidCase {
    InvalidFlag, ZeroWidth, ZeroHeight, WidthBeyondInt, HeightBeyondInt, ZeroChannels,
    ExcessChannels, UnknownFormat, ZeroAuthoredLevels, ExcessAuthoredLevels,
    ZeroOriginalLevels, ExcessOriginalLevels, FirstAtEnd, FirstBeyondEnd,
    FloatSizeOverflow, ByteReductionOverflow, HostContainerOverflow
};

class MipSuffixInvalid : public ::testing::TestWithParam<InvalidCase> {};

TEST_P(MipSuffixInvalid, RejectsBeforeSourceAccessOrCallback) {
    ObservedMipSource source(patternedSource(4, 4, 0));
    uint32_t first = 0, levels = 3;
    switch (GetParam()) {
    case InvalidCase::InvalidFlag: source.info.isValid = false; break;
    case InvalidCase::ZeroWidth: source.info.width = 0; break;
    case InvalidCase::ZeroHeight: source.info.height = 0; break;
    case InvalidCase::WidthBeyondInt: source.info.width = uint32_t{INT32_MAX} + 1; break;
    case InvalidCase::HeightBeyondInt: source.info.height = uint32_t{INT32_MAX} + 1; break;
    case InvalidCase::ZeroChannels: source.info.numChannels = 0; break;
    case InvalidCase::ExcessChannels: source.info.numChannels = 5; break;
    case InvalidCase::UnknownFormat: source.info.format = static_cast<hipArray_Format>(0); break;
    case InvalidCase::ZeroAuthoredLevels: source.info.numMipLevels = 0; break;
    case InvalidCase::ExcessAuthoredLevels: source.info.numMipLevels = 4; break;
    case InvalidCase::ZeroOriginalLevels: levels = 0; break;
    case InvalidCase::ExcessOriginalLevels: levels = 4; break;
    case InvalidCase::FirstAtEnd: first = levels; break;
    case InvalidCase::FirstBeyondEnd: first = UINT32_MAX; break;
    case InvalidCase::FloatSizeOverflow:
        source.info.width = source.info.height = INT32_MAX;
        source.info.format = HIP_AD_FORMAT_FLOAT;
        break;
    case InvalidCase::ByteReductionOverflow:
        source.info.width = source.info.height = 300000000;
        first = 1;
        levels = 2;
        break;
    case InvalidCase::HostContainerOverflow:
        source.info.width = source.info.height = INT32_MAX;
        source.info.numChannels = 1;
        levels = 1;
        break;
    }
    for (bool generate : {false, true}) {
        MipSuffixReadStats stats{123, 456, 7, 8};
        EXPECT_EQ(visitMipSuffix(source, source.info, false, generate, first, levels, UINT64_MAX, stats,
            [](uint32_t, const ImageData&) {
                ADD_FAILURE() << "Invalid request must not visit images";
                return Outcome::Success;
            }), Outcome::InvalidInput);
        expectStats(stats, 0, 0, 0, 0);
    }
    EXPECT_EQ(source.opens, 0u);
    EXPECT_EQ(source.openQueries, 0u);
    EXPECT_EQ(source.infoQueries, 0u);
    EXPECT_EQ(source.reads, 0u);
}

INSTANTIATE_TEST_SUITE_P(InvalidMetadataAndRanges, MipSuffixInvalid, ::testing::Values(
    InvalidCase::InvalidFlag, InvalidCase::ZeroWidth, InvalidCase::ZeroHeight,
    InvalidCase::WidthBeyondInt, InvalidCase::HeightBeyondInt, InvalidCase::ZeroChannels,
    InvalidCase::ExcessChannels, InvalidCase::UnknownFormat, InvalidCase::ZeroAuthoredLevels,
    InvalidCase::ExcessAuthoredLevels, InvalidCase::ZeroOriginalLevels,
    InvalidCase::ExcessOriginalLevels, InvalidCase::FirstAtEnd, InvalidCase::FirstBeyondEnd,
    InvalidCase::FloatSizeOverflow,
    InvalidCase::ByteReductionOverflow, InvalidCase::HostContainerOverflow));

TEST(MipSuffixData, FailedAuthoredReadIsNeverRegenerated) {
    auto source = makeAuthoredMipSource(true);
    source.failReadLevel = 2;
    MipSuffixReadStats stats;
    std::vector<uint32_t> visited;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, true, 1, 4, 256, stats,
        [&](uint32_t level, const ImageData&) {
            visited.push_back(level);
            return Outcome::Success;
        }), Outcome::SourceFailure);
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{1, 2}));
    EXPECT_EQ(visited, (std::vector<uint32_t>{1}));
    expectStats(stats, 256, 256, 1, 0);
}

TEST(MipSuffixData, FailedNearestDependencyDoesNotReadFinerAuthoredLevels) {
    auto source = makeAuthoredMipSource(true);
    source.info.numMipLevels = 2;
    source.failReadLevel = 1;
    MipSuffixReadStats stats;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, true, 3, 4, 320, stats,
        [](uint32_t, const ImageData&) {
            ADD_FAILURE() << "Failed dependency must not visit images";
            return Outcome::Success;
        }), Outcome::SourceFailure);
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{1}));
    expectStats(stats, 256, 0, 0, 0);
}

class MipSuffixCallback : public ::testing::TestWithParam<Outcome> {};

TEST_P(MipSuffixCallback, NonSuccessStopsWithoutFurtherReadsAndPropagatesExactly) {
    auto source = makeAuthoredMipSource(true);
    MipSuffixReadStats stats;
    uint32_t calls = 0;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, false, 1, 4, 256, stats,
        [&](uint32_t level, const ImageData&) {
            EXPECT_EQ(level, 1u);
            ++calls;
            return GetParam();
        }), GetParam());
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{1}));
    expectStats(stats, 256, 256, 1, 0);
}

INSTANTIATE_TEST_SUITE_P(VisitorOutcomes, MipSuffixCallback, ::testing::Values(
    Outcome::Pending, Outcome::Deferred, Outcome::InvalidInput, Outcome::Cancelled,
    Outcome::RuntimeFailure, Outcome::HostOutOfMemory, Outcome::SourceFailure, Outcome::DemandTooLarge));

TEST(MipSuffixData, CallbackFailureStopsBeforeFurtherGeneration) {
    auto source = patternedSource(8, 8, 7);
    MipSuffixReadStats stats;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, true, 1, 4, 1280, stats,
        [](uint32_t level, const ImageData&) {
            EXPECT_EQ(level, 1u);
            return Outcome::Cancelled;
        }), Outcome::Cancelled);
    expectStats(stats, 1280, 1024, 1, 1);
}

TEST(MipSuffixData, VisitorExceptionsIncludingBadAllocAreNotCaught) {
    auto source = makeAuthoredMipSource(false);
    MipSuffixReadStats stats;
    EXPECT_THROW(visitMipSuffix(source, source.info, false, false, 3, 4, 4, stats,
        [](uint32_t, const ImageData&) -> Outcome { throw std::runtime_error("visitor"); }),
        std::runtime_error);
    expectStats(stats, 4, 4, 1, 0);
    EXPECT_THROW(visitMipSuffix(source, source.info, false, false, 3, 4, 4, stats,
        [](uint32_t, const ImageData&) -> Outcome { throw std::bad_alloc(); }), std::bad_alloc);
    expectStats(stats, 4, 4, 1, 0);
    EXPECT_THROW(visitMipSuffix(source, source.info, false, false, 3, 4, 4, stats,
        [](uint32_t, const ImageData&) -> Outcome { throw 17; }), int);
    expectStats(stats, 4, 4, 1, 0);
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{3, 3, 3}));
}

enum class SourceStage { Open, IsOpen, GetInfo, Read };
enum class SourceException { Standard, NonStandard, BadAlloc };

class MipSuffixSourceException
    : public ::testing::TestWithParam<std::tuple<SourceStage, SourceException>> {};

TEST_P(MipSuffixSourceException, ClassifiesStandardFailuresAndPropagatesUnknownExceptions) {
    const auto [stage, exception] = GetParam();
    ObservedMipSource source(makeAuthoredMipSource(false));
    const auto fail = [exception] {
        if (exception == SourceException::BadAlloc) throw std::bad_alloc();
        if (exception == SourceException::NonStandard) throw 17;
        throw std::runtime_error("source");
    };
    switch (stage) {
    case SourceStage::Open: source.beforeOpen = fail; break;
    case SourceStage::IsOpen: source.beforeIsOpen = fail; break;
    case SourceStage::GetInfo: source.beforeInfo = fail; break;
    case SourceStage::Read: source.beforeRead = fail; break;
    }
    MipSuffixReadStats stats;
    const auto visit = [&] {
        return visitMipSuffix(source, source.info, false, false, 3, 4, 4, stats,
        [](uint32_t, const ImageData&) {
            ADD_FAILURE() << "Source exception must not visit images";
            return Outcome::Success;
        });
    };
    if (exception == SourceException::NonStandard) {
        try {
            visit();
            ADD_FAILURE() << "Non-standard source exception must propagate";
        } catch (int value) {
            EXPECT_EQ(value, 17);
        }
    } else {
        const Outcome expected = exception == SourceException::BadAlloc ? Outcome::HostOutOfMemory
                                                                        : Outcome::SourceFailure;
        EXPECT_EQ(visit(), expected);
    }
    expectStats(stats, stage == SourceStage::Read ? 4 : 0, 0, 0, 0);
}

INSTANTIATE_TEST_SUITE_P(SourceMethodsAndExceptionKinds, MipSuffixSourceException,
    ::testing::Combine(::testing::Values(SourceStage::Open, SourceStage::IsOpen,
                                        SourceStage::GetInfo, SourceStage::Read),
                       ::testing::Values(SourceException::Standard, SourceException::NonStandard,
                                         SourceException::BadAlloc)));

class MipSuffixMetadata : public ::testing::TestWithParam<uint32_t> {};

TEST_P(MipSuffixMetadata, ChangedRegisteredHeaderFailsExplicitlyWithoutPixelReads) {
    ObservedMipSource source(makeAuthoredMipSource(false));
    const TextureInfo registered = source.info;
    source.opened = true;
    switch (GetParam()) {
    case 0: source.info.width /= 2; break;
    case 1: source.info.height /= 2; break;
    case 2: source.info.format = HIP_AD_FORMAT_FLOAT; break;
    case 3: source.info.numChannels = 3; break;
    case 4: source.info.numMipLevels = 3; break;
    case 5: source.info.isValid = false; break;
    case 6: source.info.isTiled = true; break;
    }
    MipSuffixReadStats stats;
    EXPECT_EQ(visitMipSuffix(source, registered, false, false, 3, 4, 4, stats, ignoreImage),
              Outcome::SourceFailure);
    EXPECT_EQ(source.reads, 0u);
    expectStats(stats, 0, 0, 0, 0);
}

INSTANTIATE_TEST_SUITE_P(AllTextureInfoFields, MipSuffixMetadata, ::testing::Range(0u, 7u));

TEST(MipSuffixData, MetadataChangedByOpenDoesNotExpandAdmittedPayload) {
    ObservedMipSource source(makeAuthoredMipSource(false));
    source.beforeOpen = [&] { source.info.width = 1024; };
    MipSuffixReadStats stats;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, false, 3, 4, 4, stats, ignoreImage),
              Outcome::SourceFailure);
    EXPECT_EQ(source.opens, 1u);
    EXPECT_EQ(source.reads, 0u);
    expectStats(stats, 0, 0, 0, 0);
}

TEST(MipSuffixData, MetadataChangedDuringReadCountsNativeInputButNeverVisitsImage) {
    ObservedMipSource source(makeAuthoredMipSource(false));
    source.afterRead = [&] { source.info.width = 1024; };
    MipSuffixReadStats stats;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, false, 2, 4, 16, stats,
        [](uint32_t, const ImageData&) {
            ADD_FAILURE() << "Changed source must not visit image";
            return Outcome::Success;
        }), Outcome::SourceFailure);
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{2}));
    expectStats(stats, 16, 16, 1, 0);
}

TEST(MipSuffixData, HeaderChangeBetweenReadsCannotAlterAllocationMetadata) {
    ObservedMipSource source(makeAuthoredMipSource(false));
    MipSuffixReadStats stats;
    uint32_t visits = 0;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, false, 2, 4, 16, stats,
        [&](uint32_t, const ImageData&) {
            ++visits;
            source.info.width = 1024;
            return Outcome::Success;
        }), Outcome::SourceFailure);
    EXPECT_EQ(visits, 1u);
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{2}));
    expectStats(stats, 16, 16, 1, 0);
}

TEST(MipSuffixData, ClosedOrChangedSourceStopsFurtherGenerationWithoutReopening) {
    for (bool close : {false, true}) {
        ObservedMipSource source(patternedSource(8, 8, 0));
        MipSuffixReadStats stats;
        uint32_t visits = 0;
        EXPECT_EQ(visitMipSuffix(source, source.info, false, true, 1, 4, 320, stats,
            [&](uint32_t level, const ImageData&) {
                EXPECT_EQ(level, 1u);
                ++visits;
                if (close) source.opened = false;
                else source.info.width = 1024;
                return Outcome::Success;
            }), Outcome::SourceFailure);
        EXPECT_EQ(visits, 1u);
        EXPECT_EQ(source.opens, 1u);
        EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{0}));
        expectStats(stats, 320, 256, 1, 1);
    }
}

TEST(MipSuffixData, RepeatedVisitsResetStatsAndDoNotCacheDecodedLevelsOrCloseSource) {
    ObservedMipSource source(makeAuthoredMipSource(false));
    MipSuffixReadStats stats;
    for (int pass = 0; pass < 3; ++pass) {
        EXPECT_EQ(visitMipSuffix(source, source.info, false, false, 2, 4, 16, stats, ignoreImage),
                  Outcome::Success);
        expectStats(stats, 16, 20, 2, 0);
    }
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{2, 3, 2, 3, 2, 3}));
    EXPECT_EQ(source.opens, 1u);
    EXPECT_EQ(source.closes, 0u);
}

TEST(MipSuffixData, CallbackIsSynchronousAndCallerOwnedSourceSurvivesBlockedConsumption) {
    auto owner = std::make_shared<ObservedMipSource>(makeAuthoredMipSource(false));
    std::weak_ptr<ObservedMipSource> weak = owner;
    std::promise<void> entered;
    auto entry = entered.get_future();
    std::promise<void> release;
    auto released = release.get_future();
    {
        auto work = std::async(std::launch::async, [source = owner, &entered, &released] {
            MipSuffixReadStats stats;
            return visitMipSuffix(*source, source->info, false, false, 2, 4, 16, stats,
                [&](uint32_t level, const ImageData& image) {
                    if (level == 2) {
                        EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{2}));
                        entered.set_value();
                        released.wait();
                        EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{2}));
                        EXPECT_EQ(std::memcmp(image.data(), source->mipPixels[2].data(),
                                              image.sizeBytes()), 0);
                        EXPECT_EQ(source->closes, 0u);
                    }
                    return Outcome::Success;
                });
        });
        owner.reset();
        EXPECT_EQ(entry.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        EXPECT_FALSE(weak.expired());
        release.set_value();
        EXPECT_EQ(work.get(), Outcome::Success);
    }
    EXPECT_TRUE(weak.expired());
}

TEST(MipSuffixData, MoveOnlyVisitorIsNotCopied) {
    auto source = patternedSource(1, 1, 0);
    MipSuffixReadStats stats;
    EXPECT_EQ(visitMipSuffix(source, source.info, false, false, 0, 1, 4, stats,
        [state = std::make_unique<int>(17)](uint32_t level, const ImageData& image) {
            EXPECT_EQ(*state, 17);
            EXPECT_EQ(level, 0u);
            EXPECT_EQ(image.sizeBytes(), 4u);
            return Outcome::Success;
        }), Outcome::Success);
    expectStats(stats, 4, 4, 1, 0);
}

} // namespace
} // namespace test
} // namespace hip_demand
