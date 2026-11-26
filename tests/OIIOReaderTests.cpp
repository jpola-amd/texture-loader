#include <hip/hip_runtime.h>
#include <gtest/gtest.h>
#include <ImageSource/OIIOReader.h>
#include <OpenImageIO/imageio.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

struct FormatCase {
    OIIO::TypeDesc type;
    hipArray_Format format;
    const char* name;
    const char* extension;
};
const FormatCase formats[] = {
    {OIIO::TypeDesc::UINT8, HIP_AD_FORMAT_UNSIGNED_INT8, "UInt8", ".tif"},
    {OIIO::TypeDesc::INT8, HIP_AD_FORMAT_SIGNED_INT8, "Int8", ".tif"},
    {OIIO::TypeDesc::UINT16, HIP_AD_FORMAT_UNSIGNED_INT16, "UInt16", ".tif"},
    {OIIO::TypeDesc::INT16, HIP_AD_FORMAT_SIGNED_INT16, "Int16", ".tif"},
    {OIIO::TypeDesc::UINT32, HIP_AD_FORMAT_UNSIGNED_INT32, "UInt32", ".tif"},
    {OIIO::TypeDesc::INT32, HIP_AD_FORMAT_SIGNED_INT32, "Int32", ".tif"},
    {OIIO::TypeDesc::HALF, HIP_AD_FORMAT_HALF, "Half", ".exr"},
    {OIIO::TypeDesc::FLOAT, HIP_AD_FORMAT_FLOAT, "Float", ".exr"},
};

class OIIOReaderFiles : public ::testing::Test {
protected:
    std::filesystem::path directory;
    void SetUp() override {
        static std::atomic<unsigned int> sequence{0};
        directory = std::filesystem::temp_directory_path() /
            ("hip_demand_oiio_" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
             std::to_string(sequence.fetch_add(1)));
        ASSERT_TRUE(std::filesystem::create_directory(directory));
    }
    void TearDown() override {
        if (!directory.empty()) std::filesystem::remove_all(directory);
    }
};

class OIIOReaderNativeFile : public OIIOReaderFiles,
                           public ::testing::WithParamInterface<std::tuple<FormatCase, unsigned int>> {};

TEST_P(OIIOReaderNativeFile, MetadataPayloadAndBaseColorAgree) {
    const auto [format, channels] = GetParam();
    const std::array<float, 4> values{{2.5f, -0.5f, 0.25f, 0.75f}};
    std::vector<unsigned char> native(channels * format.type.size());
    ASSERT_TRUE(OIIO::convert_types(OIIO::TypeDesc::FLOAT, values.data(),
                                    format.type, native.data(), channels));
    const auto filename = (directory / (std::string("typed") + format.extension)).string();
    auto output = OIIO::ImageOutput::create(filename);
    ASSERT_NE(output, nullptr);
    OIIO::ImageSpec spec(1, 1, static_cast<int>(channels), format.type);
    ASSERT_TRUE(output->open(filename, spec)) << output->geterror();
    ASSERT_TRUE(output->write_image(format.type, native.data())) << output->geterror();
    ASSERT_TRUE(output->close()) << output->geterror();

    hip_demand::OIIOReader reader(filename);
    hip_demand::TextureInfo info;
    ASSERT_NO_THROW(reader.open(&info));
    ASSERT_TRUE(info.isValid);
    ASSERT_EQ(info.format, format.format);
    ASSERT_EQ(info.numChannels, channels);
    ASSERT_EQ(info.numMipLevels, 1u);
    ASSERT_EQ(hip_demand::getBytesPerChannel(info.format) * info.numChannels, native.size());
    constexpr size_t guard = 16;
    std::vector<unsigned char> copied(native.size() + 2 * guard, 0xa5);
    ASSERT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(copied.data() + guard), 0, 1, 1));
    EXPECT_EQ(std::vector<unsigned char>(copied.begin() + guard, copied.end() - guard), native);
    EXPECT_EQ(std::vector<unsigned char>(copied.begin(), copied.begin() + guard),
              std::vector<unsigned char>(guard, 0xa5));
    EXPECT_EQ(std::vector<unsigned char>(copied.end() - guard, copied.end()),
              std::vector<unsigned char>(guard, 0xa5));
    EXPECT_EQ(reader.getNumBytesRead(), native.size());
    EXPECT_FALSE(reader.readMipLevel(nullptr, 0, 1, 1));
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(copied.data()), 0, 2, 1));
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(copied.data()), 1, 1, 1));

    std::array<float, 4> expected{{0.f, 0.f, 0.f, 1.f}};
    ASSERT_TRUE(OIIO::convert_types(format.type, native.data(), OIIO::TypeDesc::FLOAT,
                                    expected.data(), channels));
    float4 color{};
    ASSERT_TRUE(reader.readBaseColor(color));
    EXPECT_FLOAT_EQ(color.x, expected[0]);
    EXPECT_FLOAT_EQ(color.y, expected[1]);
    EXPECT_FLOAT_EQ(color.z, expected[2]);
    EXPECT_FLOAT_EQ(color.w, expected[3]);
    reader.close();
    EXPECT_FALSE(reader.readBaseColor(color));
}

INSTANTIATE_TEST_SUITE_P(FormatsAndChannels, OIIOReaderNativeFile,
    ::testing::Combine(::testing::ValuesIn(formats), ::testing::Range(1u, 5u)),
    [](const auto& parameter) {
        return std::string(std::get<0>(parameter.param).name) + "C" +
               std::to_string(std::get<1>(parameter.param));
    });

TEST_F(OIIOReaderFiles, ReportsOnlyAuthoredLevelsSoLoaderOwnsFallbackFiltering) {
    const auto filename = (directory / "base.exr").string();
    auto output = OIIO::ImageOutput::create(filename);
    ASSERT_NE(output, nullptr);
    OIIO::ImageSpec spec(3, 1, 1, OIIO::TypeDesc::FLOAT);
    const std::array<float, 3> pixels{{-2.f, 0.5f, 90.f}};
    ASSERT_TRUE(output->open(filename, spec));
    ASSERT_TRUE(output->write_image(OIIO::TypeDesc::FLOAT, pixels.data()));
    ASSERT_TRUE(output->close());
    hip_demand::OIIOReader reader(filename);
    hip_demand::TextureInfo info;
    reader.open(&info);
    ASSERT_EQ(info.numMipLevels, 1u);
    std::array<float, 3> actual{};
    ASSERT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(actual.data()), 0, 3, 1));
    EXPECT_EQ(actual, pixels);
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(actual.data()), 1, 1, 1));
    float4 color{};
    EXPECT_FALSE(reader.readBaseColor(color));
}

TEST_F(OIIOReaderFiles, ReadsAuthoredHdrMipValuesInsteadOfRegeneratingThem) {
    const auto filename = (directory / "authored.exr").string();
    auto output = OIIO::ImageOutput::create(filename);
    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(output->supports("mipmap"));
    const std::array<float, 3> values{{2.f, 10.f, -3.5f}};
    for (int level = 0; level < 3; ++level) {
        const int size = 4 >> level;
        OIIO::ImageSpec spec(size, size, 4, OIIO::TypeDesc::FLOAT);
        spec.tile_width = spec.tile_height = 16;
        spec.attribute("textureformat", "Plain Texture");
        spec.attribute("openexr:levelmode", 1);
        spec.attribute("openexr:roundingmode", 0);
        ASSERT_TRUE(output->open(filename, spec,
            level == 0 ? OIIO::ImageOutput::Create : OIIO::ImageOutput::AppendMIPLevel))
            << output->geterror();
        const std::vector<float> pixels(static_cast<size_t>(size) * size * 4, values[level]);
        ASSERT_TRUE(output->write_image(OIIO::TypeDesc::FLOAT, pixels.data())) << output->geterror();
    }
    ASSERT_TRUE(output->close()) << output->geterror();
    output.reset();
    auto input = OIIO::ImageInput::open(filename);
    ASSERT_NE(input, nullptr) << OIIO::geterror();
    input->close();
    hip_demand::OIIOReader reader(filename);
    hip_demand::TextureInfo info;
    reader.open(&info);
    ASSERT_EQ(info.numMipLevels, 3u);
    ASSERT_TRUE(info.isTiled);
    for (unsigned int level = 0; level < 3; ++level) {
        const unsigned int size = 4 >> level;
        std::vector<float> pixels(static_cast<size_t>(size) * size * 4);
        ASSERT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(pixels.data()), level, size, size));
        for (const float value : pixels) EXPECT_FLOAT_EQ(value, values[level]);
    }
    EXPECT_EQ(reader.getNumBytesRead(), (16u + 4u + 1u) * 4u * sizeof(float));
    float4 color{};
    ASSERT_TRUE(reader.readBaseColor(color));
    EXPECT_FLOAT_EQ(color.x, -3.5f);
    EXPECT_FLOAT_EQ(color.w, -3.5f);
}

TEST_F(OIIOReaderFiles, KeepsValidBaseImageWhenAuthoredMipDimensionsRoundUp) {
    const auto filename = (directory / "rounded-up.exr").string();
    auto output = OIIO::ImageOutput::create(filename);
    ASSERT_NE(output, nullptr);
    const std::array<int, 3> widths{{3, 2, 1}};
    for (int level = 0; level < 3; ++level) {
        OIIO::ImageSpec spec(widths[level], 1, 4, OIIO::TypeDesc::FLOAT);
        spec.tile_width = spec.tile_height = 16;
        spec.attribute("textureformat", "Plain Texture");
        spec.attribute("openexr:levelmode", 1);
        spec.attribute("openexr:roundingmode", 1);
        ASSERT_TRUE(output->open(filename, spec,
            level == 0 ? OIIO::ImageOutput::Create : OIIO::ImageOutput::AppendMIPLevel))
            << output->geterror();
        const std::vector<float> pixels(static_cast<size_t>(widths[level]) * 4, 6.f + level);
        ASSERT_TRUE(output->write_image(OIIO::TypeDesc::FLOAT, pixels.data())) << output->geterror();
    }
    ASSERT_TRUE(output->close()) << output->geterror();
    output.reset();
    hip_demand::OIIOReader reader(filename);
    hip_demand::TextureInfo info;
    ASSERT_NO_THROW(reader.open(&info));
    ASSERT_EQ(info.width, 3u);
    ASSERT_EQ(info.numMipLevels, 1u);
    std::array<float, 12> pixels{};
    ASSERT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(pixels.data()), 0, 3, 1));
    for (const float value : pixels) EXPECT_FLOAT_EQ(value, 6.f);
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(pixels.data()), 1, 1, 1));
}

// --- Reading tiles straight from the file -------------------------------------------------------------------------

// A pixel value that depends on everything a read can get wrong: position, level and channel.
int patternValue(unsigned int x, unsigned int y, unsigned int level, unsigned int channel) {
    return static_cast<int>((x * 7 + y * 13 + level * 31 + channel * 50) % 256);
}

struct LevelSpec {
    unsigned int width;
    unsigned int height;
};

// Writes a file whose levels hold patternValue(); a zero tile size writes scanlines.
void writePatternImage(const std::string& filename, const std::vector<LevelSpec>& levels, unsigned int channels,
                       OIIO::TypeDesc type, unsigned int tileSize) {
    auto output = OIIO::ImageOutput::create(filename);
    ASSERT_NE(output, nullptr) << OIIO::geterror();
    if (levels.size() > 1) ASSERT_TRUE(output->supports("mipmap"));
    for (unsigned int level = 0; level < levels.size(); ++level) {
        const LevelSpec& dims = levels[level];
        OIIO::ImageSpec spec(static_cast<int>(dims.width), static_cast<int>(dims.height), static_cast<int>(channels), type);
        if (tileSize) spec.tile_width = spec.tile_height = static_cast<int>(tileSize);
        if (levels.size() > 1) {
            spec.attribute("textureformat", "Plain Texture");
            spec.attribute("openexr:levelmode", 1);
            spec.attribute("openexr:roundingmode", 0);
        }
        ASSERT_TRUE(output->open(filename, spec, level == 0 ? OIIO::ImageOutput::Create : OIIO::ImageOutput::AppendMIPLevel))
            << output->geterror();
        std::vector<float> pixels(static_cast<size_t>(dims.width) * dims.height * channels);
        for (unsigned int y = 0; y < dims.height; ++y)
            for (unsigned int x = 0; x < dims.width; ++x)
                for (unsigned int c = 0; c < channels; ++c)
                    pixels[(static_cast<size_t>(y) * dims.width + x) * channels + c] = patternValue(x, y, level, c) / 255.f;
        ASSERT_TRUE(output->write_image(OIIO::TypeDesc::FLOAT, pixels.data())) << output->geterror();
    }
    ASSERT_TRUE(output->close()) << output->geterror();
}

// Reads a tile and compares it with the pattern; pixels outside the level must be zero.
template <class T>
void expectTile(hip_demand::OIIOReader& reader, unsigned int level, const LevelSpec& dims, unsigned int channels,
                const hip_demand::Tile& tile, float scale) {
    constexpr unsigned char fill = 0xa5;
    std::vector<T> pixels(static_cast<size_t>(tile.width) * tile.height * channels);
    std::memset(pixels.data(), fill, pixels.size() * sizeof(T));
    ASSERT_TRUE(reader.readTile(reinterpret_cast<char*>(pixels.data()), level, tile, nullptr))
        << "level " << level << " tile " << tile.x << ',' << tile.y;
    for (unsigned int ty = 0; ty < tile.height; ++ty) {
        for (unsigned int tx = 0; tx < tile.width; ++tx) {
            const unsigned int x = tile.x * tile.width + tx;
            const unsigned int y = tile.y * tile.height + ty;
            for (unsigned int c = 0; c < channels; ++c) {
                const float expected = x < dims.width && y < dims.height ? patternValue(x, y, level, c) * scale : 0.f;
                const float actual = static_cast<float>(pixels[(static_cast<size_t>(ty) * tile.width + tx) * channels + c]);
                ASSERT_FLOAT_EQ(actual, expected) << "level " << level << " pixel " << x << ',' << y << " channel " << c;
            }
        }
    }
}

template <class T>
void expectAllTiles(hip_demand::OIIOReader& reader, const std::vector<LevelSpec>& levels, unsigned int channels,
                    unsigned int tileSize, float scale) {
    for (unsigned int level = 0; level < levels.size(); ++level) {
        const LevelSpec& dims = levels[level];
        const unsigned int tilesX = (dims.width + tileSize - 1) / tileSize;
        const unsigned int tilesY = (dims.height + tileSize - 1) / tileSize;
        for (unsigned int ty = 0; ty < tilesY; ++ty)
            for (unsigned int tx = 0; tx < tilesX; ++tx)
                expectTile<T>(reader, level, dims, channels, hip_demand::Tile{tx, ty, tileSize, tileSize}, scale);
    }
}

// An EXR in mipmap mode declares the whole chain down to 1x1, so every level has to be written.
const std::vector<LevelSpec> fullChain{{40, 24}, {20, 12}, {10, 6}, {5, 3}, {2, 1}, {1, 1}};

TEST_F(OIIOReaderFiles, TiledMipmappedExrServesTilesOfEveryLevelWithZeroPadding) {
    const auto filename = (directory / "tiled.exr").string();
    writePatternImage(filename, fullChain, 3, OIIO::TypeDesc::FLOAT, 16);

    hip_demand::OIIOReader reader(filename);
    EXPECT_FALSE(reader.isOpen());
    hip_demand::TextureInfo info;
    ASSERT_NO_THROW(reader.open(&info));
    EXPECT_TRUE(reader.isOpen());
    EXPECT_TRUE(info.isValid);
    EXPECT_TRUE(info.isTiled);
    EXPECT_EQ(info.width, 40u);
    EXPECT_EQ(info.height, 24u);
    EXPECT_EQ(info.numChannels, 3u);
    EXPECT_EQ(info.format, HIP_AD_FORMAT_FLOAT);
    EXPECT_EQ(info.numMipLevels, 6u);
    EXPECT_EQ(reader.getInfo(), info);

    // 32x32 tiles span two 16x16 file tiles per axis and cross both edges of every level.
    expectAllTiles<float>(reader, fullChain, 3, 32, 1.f / 255.f);
    EXPECT_EQ(reader.getNumTilesRead(), 2u + 1u + 1u + 1u + 1u + 1u);
    EXPECT_GT(reader.getNumBytesRead(), 0u);
    EXPECT_GT(reader.getTotalReadTime(), 0.0);

    std::vector<float> pixels(40 * 24 * 3);
    EXPECT_FALSE(reader.readTile(reinterpret_cast<char*>(pixels.data()), 0, hip_demand::Tile{2, 0, 32, 32}, nullptr));
    EXPECT_FALSE(reader.readTile(reinterpret_cast<char*>(pixels.data()), 6, hip_demand::Tile{0, 0, 32, 32}, nullptr));
    EXPECT_FALSE(reader.readTile(nullptr, 0, hip_demand::Tile{0, 0, 32, 32}, nullptr));

    float4 color{};
    ASSERT_TRUE(reader.readBaseColor(color));
    EXPECT_FLOAT_EQ(color.x, patternValue(0, 0, 5, 0) / 255.f);
    EXPECT_FLOAT_EQ(color.y, patternValue(0, 0, 5, 1) / 255.f);
    EXPECT_FLOAT_EQ(color.z, patternValue(0, 0, 5, 2) / 255.f);
    EXPECT_FLOAT_EQ(color.w, 1.f);
}

TEST_F(OIIOReaderFiles, ReadMipLevelReturnsTheWholeLevelOnlyForItsOwnDimensions) {
    const auto filename = (directory / "levels.exr").string();
    writePatternImage(filename, fullChain, 2, OIIO::TypeDesc::FLOAT, 16);

    hip_demand::OIIOReader reader(filename);
    reader.open(nullptr);
    for (unsigned int level = 0; level < fullChain.size(); ++level) {
        const LevelSpec& dims = fullChain[level];
        std::vector<float> pixels(static_cast<size_t>(dims.width) * dims.height * 2);
        ASSERT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(pixels.data()), level, dims.width, dims.height));
        for (unsigned int y = 0; y < dims.height; ++y)
            for (unsigned int x = 0; x < dims.width; ++x)
                for (unsigned int c = 0; c < 2; ++c)
                    ASSERT_FLOAT_EQ(pixels[(static_cast<size_t>(y) * dims.width + x) * 2 + c], patternValue(x, y, level, c) / 255.f);
        EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(pixels.data()), level, dims.width + 1, dims.height));
    }
    std::vector<float> scratch(40 * 24 * 2);
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(scratch.data()), 6, 1, 1));
    EXPECT_FALSE(reader.readMipLevel(nullptr, 0, 40, 24));
    EXPECT_EQ(reader.getNumTilesRead(), 6u);
    EXPECT_EQ(reader.getNumBytesRead(), (40u * 24u + 20u * 12u + 10u * 6u + 5u * 3u + 2u * 1u + 1u) * 2u * sizeof(float));
}

TEST_F(OIIOReaderFiles, BaseColorIsTheOneByOneLevelUnlessNotRequested) {
    const auto filename = (directory / "base.exr").string();
    writePatternImage(filename, {{4, 4}, {2, 2}, {1, 1}}, 4, OIIO::TypeDesc::HALF, 16);

    hip_demand::OIIOReader reader(filename);
    hip_demand::TextureInfo info;
    reader.open(&info);
    EXPECT_EQ(info.format, HIP_AD_FORMAT_HALF);
    EXPECT_EQ(info.numMipLevels, 3u);
    float4 color{};
    ASSERT_TRUE(reader.readBaseColor(color));
    // Half precision rounds the pattern values.
    constexpr float halfTolerance = 1e-3f;
    EXPECT_NEAR(color.x, patternValue(0, 0, 2, 0) / 255.f, halfTolerance);
    EXPECT_NEAR(color.y, patternValue(0, 0, 2, 1) / 255.f, halfTolerance);
    EXPECT_NEAR(color.z, patternValue(0, 0, 2, 2) / 255.f, halfTolerance);
    EXPECT_NEAR(color.w, patternValue(0, 0, 2, 3) / 255.f, halfTolerance);
    reader.close();
    EXPECT_FALSE(reader.isOpen());
    EXPECT_FALSE(reader.readBaseColor(color));
    std::vector<uint16_t> pixels(16 * 16 * 4);
    EXPECT_FALSE(reader.readTile(reinterpret_cast<char*>(pixels.data()), 0, hip_demand::Tile{0, 0, 16, 16}, nullptr));

    hip_demand::OIIOReader noBaseColor(filename, false);
    noBaseColor.open(nullptr);
    EXPECT_FALSE(noBaseColor.readBaseColor(color));
}

TEST_F(OIIOReaderFiles, TiledTiffServesSingleChannelBytes) {
    const auto filename = (directory / "tiled.tif").string();
    writePatternImage(filename, {{40, 24}}, 1, OIIO::TypeDesc::UINT8, 16);

    hip_demand::OIIOReader reader(filename);
    hip_demand::TextureInfo info;
    reader.open(&info);
    EXPECT_TRUE(info.isTiled);
    EXPECT_EQ(info.format, HIP_AD_FORMAT_UNSIGNED_INT8);
    EXPECT_EQ(info.numChannels, 1u);
    EXPECT_EQ(info.numMipLevels, 1u);
    expectAllTiles<uint8_t>(reader, {{40, 24}}, 1, 16, 1.f);
    expectAllTiles<uint8_t>(reader, {{40, 24}}, 1, 32, 1.f);
    EXPECT_EQ(reader.getNumTilesRead(), 6u + 2u);
    // A single level is never 1x1 here, so there is no base color.
    float4 color{};
    EXPECT_FALSE(reader.readBaseColor(color));
}

TEST_F(OIIOReaderFiles, ScanlinePngServesTilesByRows) {
    const auto filename = (directory / "scanline.png").string();
    writePatternImage(filename, {{40, 24}}, 3, OIIO::TypeDesc::UINT8, 0);

    hip_demand::OIIOReader reader(filename);
    hip_demand::TextureInfo info;
    reader.open(&info);
    EXPECT_FALSE(info.isTiled);
    EXPECT_EQ(info.numChannels, 3u);
    expectAllTiles<uint8_t>(reader, {{40, 24}}, 3, 16, 1.f);
    EXPECT_EQ(reader.getNumTilesRead(), 6u);
    // Each tile reads its rows across the whole level width: 16 rows for the top row of tiles, 8 for the bottom.
    EXPECT_EQ(reader.getNumBytesRead(), 3u * 16u * 40u * 3u + 3u * 8u * 40u * 3u);
}

TEST_F(OIIOReaderFiles, RejectsTilesThatAreNotMultiplesOfTheFileTile) {
    const auto filename = (directory / "coarse.tif").string();
    writePatternImage(filename, {{64, 64}}, 1, OIIO::TypeDesc::UINT8, 32);

    hip_demand::OIIOReader reader(filename);
    reader.open(nullptr);
    std::vector<uint8_t> pixels(64 * 64);
    EXPECT_THROW(reader.readTile(reinterpret_cast<char*>(pixels.data()), 0, hip_demand::Tile{0, 0, 16, 16}, nullptr), std::runtime_error);
    EXPECT_THROW(reader.readTile(reinterpret_cast<char*>(pixels.data()), 0, hip_demand::Tile{0, 0, 48, 48}, nullptr), std::runtime_error);
    EXPECT_THROW(reader.readTile(reinterpret_cast<char*>(pixels.data()), 0, hip_demand::Tile{0, 0, 64, 48}, nullptr), std::runtime_error);
    // Non-square multiples are fine.
    EXPECT_TRUE(reader.readTile(reinterpret_cast<char*>(pixels.data()), 0, hip_demand::Tile{0, 0, 64, 32}, nullptr));
    expectAllTiles<uint8_t>(reader, {{64, 64}}, 1, 64, 1.f);
}

TEST_F(OIIOReaderFiles, ConcurrentReadsSeeTheSamePixels) {
    const auto filename = (directory / "shared.exr").string();
    writePatternImage(filename, fullChain, 3, OIIO::TypeDesc::FLOAT, 16);

    hip_demand::OIIOReader reader(filename);
    reader.open(nullptr);
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i)
        threads.emplace_back([&] { expectAllTiles<float>(reader, fullChain, 3, 16, 1.f / 255.f); });
    for (std::thread& thread : threads) thread.join();
    EXPECT_EQ(reader.getNumTilesRead(), 4u * (6u + 2u + 1u + 1u + 1u + 1u));
}

TEST_F(OIIOReaderFiles, OpenThrowsForMissingFiles) {
    hip_demand::OIIOReader reader((directory / "missing.exr").string());
    EXPECT_THROW(reader.open(nullptr), std::runtime_error);
    EXPECT_FALSE(reader.isOpen());
}

} // namespace
