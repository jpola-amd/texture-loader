#include <hip/hip_runtime.h>
#include <gtest/gtest.h>
#include <ImageSource/MipMapImageSource.h>
#include <ImageSource/TiledImageSource.h>
#include <DemandLoading/Internal/ImageData.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace {

int patternValue(unsigned int x, unsigned int y, unsigned int level, unsigned int channel) {
    return static_cast<int>((x * 7 + y * 13 + level * 31 + channel * 50) % 256);
}

// An in-memory image that can only read whole levels, like a scanline file. Its levels hold patternValue().
class LevelOnlyImage : public hip_demand::ImageSource {
public:
    LevelOnlyImage(unsigned int width, unsigned int height, unsigned int channels, hipArray_Format format,
                   unsigned int numMipLevels, bool tiled) {
        info_.width = width;
        info_.height = height;
        info_.numChannels = channels;
        info_.format = format;
        info_.numMipLevels = numMipLevels;
        info_.isTiled = tiled;
        info_.isValid = true;
    }

    void open(hip_demand::TextureInfo* info) override {
        open_ = true;
        ++openCalls;
        if (info) *info = info_;
    }
    void close() override { open_ = false; }
    bool isOpen() const override { return open_; }
    const hip_demand::TextureInfo& getInfo() const override { return info_; }

    bool readMipLevel(char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight,
                      hipStream_t) override {
        const unsigned int width = std::max(1u, info_.width >> mipLevel);
        const unsigned int height = std::max(1u, info_.height >> mipLevel);
        if (!open_ || !dest || mipLevel >= info_.numMipLevels || width != expectedWidth || height != expectedHeight)
            return false;
        ++levelReads;
        const size_t channelBytes = hip_demand::getBytesPerChannel(info_.format);
        for (unsigned int y = 0; y < height; ++y)
            for (unsigned int x = 0; x < width; ++x)
                for (unsigned int c = 0; c < info_.numChannels; ++c) {
                    const float value = patternValue(x, y, mipLevel, c) / 255.f;
                    unsigned char* texel = reinterpret_cast<unsigned char*>(dest) +
                        ((static_cast<size_t>(y) * width + x) * info_.numChannels + c) * channelBytes;
                    if (info_.format == HIP_AD_FORMAT_FLOAT)
                        std::memcpy(texel, &value, sizeof(value));
                    else
                        *texel = static_cast<unsigned char>(patternValue(x, y, mipLevel, c));
                }
        return true;
    }

    bool readTile(char*, unsigned int, const hip_demand::Tile&, hipStream_t) override {
        ++tileReads;
        return false;
    }
    bool readBaseColor(float4&) override { return false; }
    unsigned long long getNumTilesRead() const override { return tileReads; }
    unsigned long long getNumBytesRead() const override { return 0; }
    double getTotalReadTime() const override { return 0.0; }
    unsigned long long getHash(hipStream_t) const override { return 42; }

    std::atomic<unsigned int> openCalls{0};
    std::atomic<unsigned int> levelReads{0};
    std::atomic<unsigned int> tileReads{0};

private:
    hip_demand::TextureInfo info_{};
    bool open_ = false;
};

template <class T>
void expectTile(hip_demand::ImageSource& image, unsigned int level, unsigned int channels, const hip_demand::Tile& tile,
                float scale, bool mipsFromBase) {
    const hip_demand::TextureInfo& info = image.getInfo();
    const unsigned int width = std::max(1u, info.width >> level);
    const unsigned int height = std::max(1u, info.height >> level);
    std::vector<T> pixels(static_cast<size_t>(tile.width) * tile.height * channels);
    std::memset(pixels.data(), 0xa5, pixels.size() * sizeof(T));
    ASSERT_TRUE(image.readTile(reinterpret_cast<char*>(pixels.data()), level, tile, nullptr));
    for (unsigned int ty = 0; ty < tile.height; ++ty)
        for (unsigned int tx = 0; tx < tile.width; ++tx) {
            const unsigned int x = tile.x * tile.width + tx;
            const unsigned int y = tile.y * tile.height + ty;
            for (unsigned int c = 0; c < channels; ++c) {
                const float expected = x < width && y < height && mipsFromBase ? patternValue(x, y, level, c) * scale : 0.f;
                const float actual = static_cast<float>(pixels[(static_cast<size_t>(ty) * tile.width + tx) * channels + c]);
                if (x < width && y < height && !mipsFromBase) continue;  // Generated levels are checked elsewhere.
                ASSERT_FLOAT_EQ(actual, expected) << "level " << level << " pixel " << x << ',' << y << " channel " << c;
            }
        }
}

}  // namespace

TEST(TiledImageSource, ServesTilesOfALevelOnlyImageFromOneLevelRead) {
    auto base = std::make_shared<LevelOnlyImage>(40, 24, 3, HIP_AD_FORMAT_UNSIGNED_INT8, 2, false);
    std::shared_ptr<hip_demand::ImageSource> image = hip_demand::createTiledImageSource(base);
    ASSERT_NE(image, base);
    EXPECT_EQ(base->openCalls, 1u);
    hip_demand::TextureInfo info;
    image->open(&info);
    EXPECT_EQ(base->openCalls, 1u);  // Already open.
    EXPECT_TRUE(info.isTiled);
    EXPECT_EQ(info.numMipLevels, 2u);
    EXPECT_EQ(image->getInfo().isTiled, true);
    EXPECT_FALSE(base->getInfo().isTiled);
    EXPECT_EQ(image->getHash(), 42u);

    for (unsigned int level = 0; level < 2; ++level) {
        const unsigned int tilesX = (std::max(1u, 40u >> level) + 15) / 16;
        const unsigned int tilesY = (std::max(1u, 24u >> level) + 15) / 16;
        for (unsigned int ty = 0; ty < tilesY; ++ty)
            for (unsigned int tx = 0; tx < tilesX; ++tx)
                expectTile<uint8_t>(*image, level, 3, hip_demand::Tile{tx, ty, 16, 16}, 1.f, true);
    }
    EXPECT_EQ(base->levelReads, 2u);
    EXPECT_EQ(base->tileReads, 0u);
    auto* tiled = dynamic_cast<hip_demand::TiledImageSource*>(image.get());
    ASSERT_NE(tiled, nullptr);
    EXPECT_EQ(tiled->getNumTilesRead(), 6u + 2u);  // 40x24 and 20x12 in 16x16 tiles.

    std::vector<uint8_t> pixels(16 * 16 * 3);
    EXPECT_FALSE(image->readTile(reinterpret_cast<char*>(pixels.data()), 0, hip_demand::Tile{3, 0, 16, 16}, nullptr));
    EXPECT_FALSE(image->readTile(reinterpret_cast<char*>(pixels.data()), 2, hip_demand::Tile{0, 0, 16, 16}, nullptr));
    EXPECT_FALSE(image->readTile(nullptr, 0, hip_demand::Tile{0, 0, 16, 16}, nullptr));

    // Whole levels go to the base image.
    std::vector<uint8_t> level(40 * 24 * 3);
    ASSERT_TRUE(image->readMipLevel(reinterpret_cast<char*>(level.data()), 0, 40, 24));
    EXPECT_EQ(base->levelReads, 3u);
    EXPECT_EQ(level[(5 * 40 + 7) * 3 + 2], static_cast<uint8_t>(patternValue(7, 5, 0, 2)));
    EXPECT_FALSE(image->readMipLevel(reinterpret_cast<char*>(level.data()), 0, 41, 24));

    image->close();
    EXPECT_FALSE(base->isOpen());
    image->open(nullptr);
    ASSERT_TRUE(image->readTile(reinterpret_cast<char*>(pixels.data()), 0, hip_demand::Tile{0, 0, 16, 16}, nullptr));
    EXPECT_EQ(base->levelReads, 4u);
}

TEST(TiledImageSource, PassesATiledImageThrough) {
    auto base = std::make_shared<LevelOnlyImage>(8, 8, 1, HIP_AD_FORMAT_UNSIGNED_INT8, 1, true);
    std::shared_ptr<hip_demand::ImageSource> image = hip_demand::createTiledImageSource(base);
    EXPECT_EQ(image, base);
    hip_demand::TiledImageSource wrapped(base);
    std::vector<uint8_t> pixels(16 * 16);
    EXPECT_FALSE(wrapped.readTile(reinterpret_cast<char*>(pixels.data()), 0, hip_demand::Tile{0, 0, 16, 16}, nullptr));
    EXPECT_EQ(base->tileReads, 1u);
    EXPECT_EQ(base->levelReads, 0u);
    EXPECT_EQ(wrapped.getNumTilesRead(), 1u);
}

TEST(MipMapImageSource, BuildsABoxFilteredChainFromASingleLevel) {
    auto base = std::make_shared<LevelOnlyImage>(5, 3, 2, HIP_AD_FORMAT_FLOAT, 1, false);
    std::shared_ptr<hip_demand::ImageSource> image = hip_demand::createMipMapImageSource(base);
    ASSERT_NE(image, base);
    hip_demand::TextureInfo info;
    image->open(&info);
    EXPECT_EQ(info.numMipLevels, 3u);  // 5x3 -> 2x1 -> 1x1
    EXPECT_TRUE(info.isTiled);
    EXPECT_EQ(info.width, 5u);
    EXPECT_EQ(info.numChannels, 2u);

    // Level 0 is the base image.
    expectTile<float>(*image, 0, 2, hip_demand::Tile{0, 0, 4, 4}, 1.f / 255.f, true);
    expectTile<float>(*image, 0, 2, hip_demand::Tile{1, 0, 4, 4}, 1.f / 255.f, true);
    EXPECT_EQ(base->levelReads, 1u);

    // Level 1 averages 2x2 texels of level 0; the fifth column is dropped.
    std::vector<float> level1(2 * 1 * 2);
    ASSERT_TRUE(image->readMipLevel(reinterpret_cast<char*>(level1.data()), 1, 2, 1));
    for (unsigned int x = 0; x < 2; ++x)
        for (unsigned int c = 0; c < 2; ++c) {
            const float expected = (patternValue(2 * x, 0, 0, c) + patternValue(2 * x + 1, 0, 0, c) +
                                    patternValue(2 * x, 1, 0, c) + patternValue(2 * x + 1, 1, 0, c)) / 4.f / 255.f;
            EXPECT_FLOAT_EQ(level1[x * 2 + c], expected);
        }

    // Level 2 averages the two texels of level 1.
    std::vector<float> level2(2);
    ASSERT_TRUE(image->readMipLevel(reinterpret_cast<char*>(level2.data()), 2, 1, 1));
    EXPECT_FLOAT_EQ(level2[0], (level1[0] + level1[2]) / 2.f);
    EXPECT_FLOAT_EQ(level2[1], (level1[1] + level1[3]) / 2.f);
    // The base color goes to the base image, which has none.
    float4 color{};
    EXPECT_FALSE(image->readBaseColor(color));

    // Tiles of generated levels are zero outside the level.
    std::vector<float> tile(4 * 4 * 2, -1.f);
    ASSERT_TRUE(image->readTile(reinterpret_cast<char*>(tile.data()), 2, hip_demand::Tile{0, 0, 4, 4}, nullptr));
    EXPECT_FLOAT_EQ(tile[0], level2[0]);
    EXPECT_FLOAT_EQ(tile[1], level2[1]);
    for (size_t i = 2; i < tile.size(); ++i) EXPECT_FLOAT_EQ(tile[i], 0.f);
    EXPECT_FALSE(image->readTile(reinterpret_cast<char*>(tile.data()), 3, hip_demand::Tile{0, 0, 4, 4}, nullptr));
    EXPECT_FALSE(image->readMipLevel(reinterpret_cast<char*>(tile.data()), 1, 3, 1));
    EXPECT_EQ(base->levelReads, 1u);
    EXPECT_EQ(dynamic_cast<hip_demand::MipMapImageSource*>(image.get())->getNumTilesRead(), 3u);

    // Already tiled, so the tiled factory passes it through.
    EXPECT_EQ(hip_demand::createTiledImageSource(image), image);
}

TEST(MipMapImageSource, PassesAMipmappedImageThrough) {
    auto base = std::make_shared<LevelOnlyImage>(8, 8, 1, HIP_AD_FORMAT_UNSIGNED_INT8, 4, false);
    EXPECT_EQ(hip_demand::createMipMapImageSource(base), base);
    hip_demand::MipMapImageSource wrapped(base);
    EXPECT_EQ(wrapped.getInfo().numMipLevels, 4u);
    EXPECT_FALSE(wrapped.getInfo().isTiled);
    std::vector<uint8_t> level(4 * 4);
    ASSERT_TRUE(wrapped.readMipLevel(reinterpret_cast<char*>(level.data()), 1, 4, 4));
    EXPECT_EQ(base->levelReads, 1u);
    float4 color{};
    EXPECT_FALSE(wrapped.readBaseColor(color));
    std::vector<uint8_t> tile(16 * 16);
    EXPECT_FALSE(wrapped.readTile(reinterpret_cast<char*>(tile.data()), 0, hip_demand::Tile{0, 0, 16, 16}, nullptr));
    EXPECT_EQ(wrapped.getNumTilesRead(), 1u);
}

TEST(MipMapImageSource, ConcurrentReadersBuildTheChainOnce) {
    auto base = std::make_shared<LevelOnlyImage>(64, 32, 4, HIP_AD_FORMAT_UNSIGNED_INT8, 1, false);
    std::shared_ptr<hip_demand::ImageSource> image = hip_demand::createMipMapImageSource(base);
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i)
        threads.emplace_back([&] {
            for (unsigned int level = 0; level < 7; ++level)
                expectTile<uint8_t>(*image, level, 4, hip_demand::Tile{0, 0, 16, 16}, 1.f, level == 0);
        });
    for (std::thread& thread : threads) thread.join();
    EXPECT_EQ(base->levelReads, 1u);
}

// Reads levels first..numMipLevels-1 of image through readMipTail() and checks them against the base pattern.
void expectMipTail(hip_demand::ImageSource& image, unsigned int first, unsigned int channels) {
    const hip_demand::TextureInfo& info = image.getInfo();
    std::vector<uint2> dims(info.numMipLevels);
    size_t texels = 0;
    for (unsigned int level = 0; level < info.numMipLevels; ++level) {
        dims[level] = make_uint2(std::max(1u, info.width >> level), std::max(1u, info.height >> level));
        if (level >= first) texels += static_cast<size_t>(dims[level].x) * dims[level].y;
    }
    std::vector<uint8_t> tail(texels * channels, 0xa5);
    ASSERT_TRUE(image.readMipTail(reinterpret_cast<char*>(tail.data()), first, info.numMipLevels, dims.data(), nullptr));
    size_t offset = 0;
    for (unsigned int level = first; level < info.numMipLevels; ++level) {
        for (unsigned int y = 0; y < dims[level].y; ++y)
            for (unsigned int x = 0; x < dims[level].x; ++x)
                for (unsigned int c = 0; c < channels; ++c)
                    ASSERT_EQ(tail[offset + (static_cast<size_t>(y) * dims[level].x + x) * channels + c], patternValue(x, y, level, c))
                        << "level " << level << " pixel " << x << ',' << y;
        offset += static_cast<size_t>(dims[level].x) * dims[level].y * channels;
    }
    EXPECT_EQ(offset, tail.size());
}

TEST(ImageSource, DefaultReadMipTailPacksLevelReads) {
    LevelOnlyImage image(20, 12, 3, HIP_AD_FORMAT_UNSIGNED_INT8, 4, false);
    image.open(nullptr);
    expectMipTail(image, 1, 3);
    EXPECT_EQ(image.levelReads, 3u);

    // A level whose size differs from the image's fails the read.
    uint2 dims[4] = {make_uint2(20, 12), make_uint2(10, 6), make_uint2(5, 3), make_uint2(1, 1)};
    std::vector<uint8_t> tail(20 * 12 * 3);
    EXPECT_FALSE(image.readMipTail(reinterpret_cast<char*>(tail.data()), 2, 4, dims, nullptr));
}

TEST(TiledImageSource, ReadsTheMipTailFromItsLevels) {
    auto base = std::make_shared<LevelOnlyImage>(40, 24, 2, HIP_AD_FORMAT_UNSIGNED_INT8, 3, false);
    std::shared_ptr<hip_demand::ImageSource> image = hip_demand::createTiledImageSource(base);
    ASSERT_NE(image, base);
    expectMipTail(*image, 1, 2);
    EXPECT_EQ(base->levelReads, 2u);
    // Levels loaded for the tail serve tiles without another base read.
    expectTile<uint8_t>(*image, 1, 2, hip_demand::Tile{0, 0, 16, 16}, 1.f, true);
    EXPECT_EQ(base->levelReads, 2u);
    uint2 dims[3] = {make_uint2(40, 24), make_uint2(20, 12), make_uint2(10, 6)};
    std::vector<uint8_t> tail(40 * 24 * 2);
    EXPECT_FALSE(image->readMipTail(reinterpret_cast<char*>(tail.data()), 3, 3, dims, nullptr));
    EXPECT_FALSE(image->readMipTail(reinterpret_cast<char*>(tail.data()), 1, 4, dims, nullptr));
    EXPECT_FALSE(image->readMipTail(nullptr, 1, 3, dims, nullptr));

    // A tiled base image answers the tail itself.
    auto tiledBase = std::make_shared<LevelOnlyImage>(8, 8, 1, HIP_AD_FORMAT_UNSIGNED_INT8, 2, true);
    hip_demand::TiledImageSource wrapped(tiledBase);
    wrapped.open(nullptr);
    expectMipTail(wrapped, 0, 1);
    EXPECT_EQ(tiledBase->levelReads, 2u);
}

TEST(MipMapImageSource, ReadsTheMipTailFromTheGeneratedChain) {
    auto base = std::make_shared<LevelOnlyImage>(16, 8, 1, HIP_AD_FORMAT_UNSIGNED_INT8, 1, false);
    std::shared_ptr<hip_demand::ImageSource> image = hip_demand::createMipMapImageSource(base);
    ASSERT_NE(image, base);
    const hip_demand::TextureInfo& info = image->getInfo();
    ASSERT_EQ(info.numMipLevels, 5u);
    std::vector<uint2> dims(5);
    for (unsigned int level = 0; level < 5; ++level) dims[level] = make_uint2(std::max(1u, 16u >> level), std::max(1u, 8u >> level));

    std::vector<uint8_t> tail(4 * 2 + 2 * 1 + 1 * 1, 0xa5);
    ASSERT_TRUE(image->readMipTail(reinterpret_cast<char*>(tail.data()), 2, 5, dims.data(), nullptr));
    EXPECT_EQ(base->levelReads, 1u);
    // Each tail level matches the same level read on its own.
    size_t offset = 0;
    for (unsigned int level = 2; level < 5; ++level) {
        std::vector<uint8_t> single(static_cast<size_t>(dims[level].x) * dims[level].y);
        ASSERT_TRUE(image->readMipLevel(reinterpret_cast<char*>(single.data()), level, dims[level].x, dims[level].y));
        EXPECT_TRUE(std::equal(single.begin(), single.end(), tail.begin() + offset)) << "level " << level;
        offset += single.size();
    }
    EXPECT_EQ(offset, tail.size());
    EXPECT_FALSE(image->readMipTail(reinterpret_cast<char*>(tail.data()), 2, 6, dims.data(), nullptr));
    dims[4] = make_uint2(2, 2);
    EXPECT_FALSE(image->readMipTail(reinterpret_cast<char*>(tail.data()), 4, 5, dims.data(), nullptr));
}

TEST(ImageData, HalfConversionRoundTrips) {
    const float values[] = {0.f, -0.f, 1.f, -1.f, 0.5f, 0.25f, 65504.f, 1.f / 1024.f, 6.103515625e-05f, 5.9604644775390625e-08f, 3.14159f};
    for (const float value : values) {
        const uint16_t half = hip_demand::internal::floatToHalf(value);
        const float back = hip_demand::internal::halfToFloat(half);
        EXPECT_NEAR(back, value, std::abs(value) * 1e-3f + 1e-9f) << value;
    }
    EXPECT_EQ(hip_demand::internal::floatToHalf(1.f), 0x3c00u);
    EXPECT_EQ(hip_demand::internal::floatToHalf(-2.f), 0xc000u);
    EXPECT_EQ(hip_demand::internal::floatToHalf(1e10f), 0x7c00u);  // Overflows to infinity.
    EXPECT_EQ(hip_demand::internal::floatToHalf(1e-10f), 0x0000u);  // Underflows to zero.
    // Round to nearest even: 1 + 2^-11 is halfway between 1 and the next half; it rounds to 1.
    EXPECT_EQ(hip_demand::internal::floatToHalf(1.f + 1.f / 2048.f), 0x3c00u);
    EXPECT_EQ(hip_demand::internal::floatToHalf(1.f + 3.f / 2048.f), 0x3c02u);
}
