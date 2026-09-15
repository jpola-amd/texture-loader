// SPDX-License-Identifier: MIT
// Host-only fixture checks: no HIP device initialization, allocation, or launch.
#include "ImageDataTestUtils.h"
#include "SamplingTestSupport.h"
#include <chrono>
#include <fstream>

namespace hip_demand { namespace test {
namespace {

TEST(SamplingTestUtils, KnownMipIssueIsLimitedToObservedPlatform) {
    EXPECT_TRUE(hasKnownMipBlendIssue(true, "gfx1201"));
    EXPECT_TRUE(hasKnownMipBlendIssue(true, "gfx1201:xnack-"));
    EXPECT_FALSE(hasKnownMipBlendIssue(false, "gfx1201"));
    for (std::string_view architecture : {"", "gfx1100", "gfx1151", "gfx1200", "gfx12010"})
        EXPECT_FALSE(hasKnownMipBlendIssue(true, architecture));
}

TEST(SamplingTestUtils, CorrectMipBlendsPassEvenOnKnownIssuePlatform) {
    const std::array<float, 4> first{0, 0, 0, 1}, second{1, 1, 1, 1};
    const std::array<float, 4> tolerance{1e-6f, 1e-6f, 1e-6f, 1e-6f};
    for (bool allowKnownIssue : {false, true}) {
        for (float fraction : {0.f, .25f, .5f, .75f, 1.f}) {
            const std::array<float, 4> actual{fraction, fraction, fraction, 1};
            EXPECT_EQ(compareMipBlend(actual, first, second, fraction, tolerance, allowKnownIssue),
                      MipBlendComparison::Correct);
        }
    }
}

TEST(SamplingTestUtils, KnownIncorrectMipCurveRequiresExplicitEligibility) {
    const std::array<float, 4> first{-2, .25f, 3, .5f}, second{4, -1, .125f, 1};
    const std::array<float, 4> tolerance{1e-6f, 1e-6f, 1e-6f, 1e-6f};
    for (const auto& sample : {std::array<float, 5>{.25f, -.875f, .015625f, 2.4609375f, .59375f},
                               std::array<float, 5>{.75f, 2.875f, -.765625f, .6640625f, .90625f}}) {
        const std::array<float, 4> actual{sample[1], sample[2], sample[3], sample[4]};
        EXPECT_EQ(compareMipBlend(actual, first, second, sample[0], tolerance, true),
                  MipBlendComparison::KnownIncorrect);
        EXPECT_EQ(compareMipBlend(actual, first, second, sample[0], tolerance, false),
                  MipBlendComparison::Unexpected);
    }
}

TEST(SamplingTestUtils, KnownMipIssueDoesNotHideDifferentOrNonfiniteErrors) {
    const std::array<float, 4> first{0, 0, 0, 1}, second{1, 1, 1, 1};
    const std::array<float, 4> tolerance{1e-6f, 1e-6f, 1e-6f, 1e-6f};
    for (size_t channel = 0; channel < 4; ++channel) {
        for (float value : {.2f, .25f, std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN()}) {
            std::array<float, 4> actual{.1875f, .1875f, .1875f, 1};
            actual[channel] = value;
            EXPECT_EQ(compareMipBlend(actual, first, second, .25f, tolerance, true),
                      MipBlendComparison::Unexpected);
        }
    }
}

TEST(SamplingTestUtils, KnownMipComparisonKeepsTheOriginalTolerance) {
    const std::array<float, 4> first{0, 0, 0, 1}, second{1, 1, 1, 1};
    const std::array<float, 4> tolerance{1e-6f, 1e-6f, 1e-6f, 1e-6f};
    std::array<float, 4> actual{.1875f + .5e-6f, .1875f, .1875f, 1};
    EXPECT_EQ(compareMipBlend(actual, first, second, .25f, tolerance, true),
              MipBlendComparison::KnownIncorrect);
    actual[0] = .1875f + 2e-6f;
    EXPECT_EQ(compareMipBlend(actual, first, second, .25f, tolerance, true),
              MipBlendComparison::Unexpected);
}

TEST(SamplingTestUtils, AuthoredMipPixelsAndReadIndices) {
    const std::array<std::array<float, 4>, 4> floatColors{{
        {-2, .25f, 3, .5f}, {4, -1, .125f, 1}, {.5f, 2, -3, .25f}, {8, .75f, 1.5f, 0}
    }};
    const std::array<std::array<unsigned int, 4>, 4> byteColors{{
        {17, 53, 101, 239}, {211, 29, 149, 127}, {67, 223, 13, 191}, {251, 109, 181, 43}
    }};
    for (bool floating : {false, true}) {
        auto source = makeAuthoredMipSource(floating);
        EXPECT_EQ(source.info.numMipLevels, 4u);
        EXPECT_EQ(source.pixels, source.mipPixels[0]);
        // Read out of order to catch a fixture that always returns level zero.
        for (unsigned int level : {3u, 0u, 2u, 1u}) {
            SCOPED_TRACE(::testing::Message() << "floating=" << floating << " level=" << level);
            internal::ImageData image;
            ASSERT_TRUE(internal::readImageSource(source, image, level));
            EXPECT_EQ(image.width, 8u >> level);
            EXPECT_EQ(image.height, 8u >> level);
            EXPECT_EQ(image.sizeBytes(), size_t(8u >> level) * (8u >> level) * (floating ? 16 : 4));
            for (size_t pixel = 0; pixel < image.width * image.height; ++pixel) {
                for (size_t channel = 0; channel < 4; ++channel) {
                    if (floating)
                        EXPECT_FLOAT_EQ(image.floats[pixel * 4 + channel], floatColors[level][channel]);
                    else
                        EXPECT_EQ(image.bytes[pixel * 4 + channel], byteColors[level][channel]);
                }
            }
        }
        EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{3, 0, 2, 1}));
        EXPECT_EQ(source.reads, 4u);
        EXPECT_EQ(source.baseColorReads, 0u);
        internal::ImageData image;
        EXPECT_FALSE(internal::readImageSource(source, image, 4));
        EXPECT_EQ(source.reads, 4u);
        source.close();
        EXPECT_FALSE(source.isOpen());
    }
}

TEST(SamplingTestUtils, BoundaryPatternIsNonconstantAndPreservesNativeChannels) {
    for (bool floating : {false, true}) {
        auto source = makeBoundaryPatternSource(floating);
        internal::ImageData image;
        ASSERT_TRUE(internal::readImageSource(source, image));
        EXPECT_EQ(image.width, 4u);
        EXPECT_EQ(image.height, 4u);
        EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{0}));
        EXPECT_EQ(source.baseColorReads, 0u);
        for (unsigned int y = 0; y < 4; ++y) {
            for (unsigned int x = 0; x < 4; ++x) {
                const size_t index = (y * 4 + x) * 4;
                if (floating) {
                    EXPECT_FLOAT_EQ(image.floats[index], (6.f*x + y - 16) / 8);
                    EXPECT_FLOAT_EQ(image.floats[index + 1], (1.f - 2*x + 5*y) / 4);
                    EXPECT_FLOAT_EQ(image.floats[index + 2], (12.f - x - 2*y) / 4);
                    EXPECT_FLOAT_EQ(image.floats[index + 3], (2.f + 2*x + y) / 16);
                } else {
                    EXPECT_EQ(image.bytes[index], 17 + 41*x + 7*y);
                    EXPECT_EQ(image.bytes[index + 1], 13 + 11*x + 47*y);
                    EXPECT_EQ(image.bytes[index + 2], 29 + 23*x + 19*y);
                    EXPECT_EQ(image.bytes[index + 3], 251 - 13*x - 17*y);
                }
            }
        }
        EXPECT_NE(std::vector<unsigned char>(source.pixels.begin(), source.pixels.begin() + (floating ? 16 : 4)),
                  std::vector<unsigned char>(source.pixels.end() - (floating ? 16 : 4), source.pixels.end()));
    }
}

TEST(SamplingTestUtils, ReadFailureAndFreshFixtureAreIsolated) {
    auto failed = makeAuthoredMipSource(true);
    failed.failRead = true;
    internal::ImageData image;
    EXPECT_FALSE(internal::readImageSource(failed, image, 2));
    EXPECT_EQ(failed.readLevels, (std::vector<unsigned int>{2}));
    failed.close();
    auto fresh = makeAuthoredMipSource(true);
    EXPECT_TRUE(internal::readImageSource(fresh, image, 1));
    EXPECT_EQ(fresh.readLevels, (std::vector<unsigned int>{1}));
    EXPECT_EQ(failed.readLevels, (std::vector<unsigned int>{2}));
}

TEST(SamplingTestUtils, DeviceSelectionRejectsMalformedOrOverflowingOrdinals) {
    EXPECT_EQ(parseTestDevice(nullptr), 0);
    EXPECT_EQ(parseTestDevice("0"), 0);
    EXPECT_EQ(parseTestDevice("2"), 2);
    EXPECT_EQ(parseTestDevice("2147483647"), 2147483647);
    for (const char* invalid : {"", "-1", "+1", " 1", "1 ", "1x", "0.5", "2147483648",
                                "99999999999999999999999999999999999"})
        EXPECT_THROW(parseTestDevice(invalid), std::invalid_argument) << invalid;
}

class ModuleDirectory {
public:
    ModuleDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned int attempt = 0; attempt < 100; ++attempt) {
            path = std::filesystem::current_path() /
                ("sampling-module-fixture-" + std::to_string(stamp) + "-" + std::to_string(attempt));
            if (std::filesystem::create_directory(path))
                return;
        }
        throw std::runtime_error("Could not create isolated module-discovery fixture");
    }
    ~ModuleDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
        EXPECT_FALSE(error) << error.message();
    }
    void file(const std::filesystem::path& relative) {
        const auto filename = path / relative;
        std::filesystem::create_directories(filename.parent_path());
        std::ofstream output(filename, std::ios::binary);
        output << "fixture";
        ASSERT_TRUE(output.good());
    }
    std::filesystem::path path;
};

TEST(SamplingTestUtils, ModuleDiscoveryPrefersRelocatedInstallAndHonorsExplicitOverride) {
    ModuleDirectory fixture;
    fixture.file(std::filesystem::path("build") / "tests" / "texture_sampling_kernel.co");
    fixture.file(std::filesystem::path("installed tree") / "bin" / "texture_sampling_kernel.co");
    const auto build = fixture.path / "build" / "tests" / "texture_sampling_kernel.co";
    const auto installed = fixture.path / "installed tree" / "bin" / "texture_loader_tests";
    const auto installedModule = installed.parent_path() / build.filename();
    EXPECT_EQ(findSamplingModule(samplingModuleCandidates(installed, build)), installedModule);
    EXPECT_EQ(findSamplingModule(samplingModuleCandidates(fixture.path / "unrelated" / "runner", build)), build);
    const auto multiConfig = fixture.path / "build" / "Release" / "texture_loader_tests";
    EXPECT_EQ(findSamplingModule(samplingModuleCandidates(multiConfig, build)), build);
    const auto missing = (fixture.path / "missing override").string();
    const auto candidates = samplingModuleCandidates(installed, build, missing.c_str());
    ASSERT_EQ(candidates.size(), 1u);
    EXPECT_THROW(findSamplingModule(candidates), std::runtime_error);
    std::filesystem::remove(installedModule);
    std::filesystem::remove(build);
    EXPECT_THROW(findSamplingModule(samplingModuleCandidates(installed, build)), std::runtime_error);
}

} // namespace
} }
