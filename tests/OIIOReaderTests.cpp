#include <hip/hip_runtime.h>
#include <gtest/gtest.h>
#include <ImageSource/OIIOReader.h>
#include <OpenImageIO/imageio.h>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
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
        directory = std::filesystem::current_path() /
            ("hip_demand_oiio_" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
             std::to_string(sequence.fetch_add(1)));
        ASSERT_TRUE(std::filesystem::create_directory(directory));
    }
    void TearDown() override {
        if (!directory.empty()) std::filesystem::remove_all(directory);
    }

    void writePyramid(const std::string& filename, unsigned int width, unsigned int height,
                      OIIO::TypeDesc type, unsigned int channels, unsigned int levels,
                      std::vector<std::vector<unsigned char>>& pixels) {
        auto output = OIIO::ImageOutput::create(filename);
        ASSERT_NE(output, nullptr);
        ASSERT_TRUE(output->supports("mipmap"));
        pixels.clear();
        for (unsigned int level = 0; level < levels; ++level) {
            OIIO::ImageSpec spec(static_cast<int>(width), static_cast<int>(height),
                                 static_cast<int>(channels), type);
            spec.tile_width = spec.tile_height = 16;
            spec.attribute("textureformat", "Plain Texture");
            spec.attribute("openexr:levelmode", 1);
            spec.attribute("openexr:roundingmode", 0);
            spec.attribute("oiio:ColorSpace", "sRGB");
            ASSERT_TRUE(output->open(filename, spec, level == 0 ?
                OIIO::ImageOutput::Create : OIIO::ImageOutput::AppendMIPLevel))
                << output->geterror();
            std::vector<float> values(static_cast<size_t>(width) * height * channels);
            for (size_t component = 0; component < values.size(); ++component)
                values[component] = static_cast<float>(level * 8) - 3.5f +
                                    static_cast<float>(component % 19) * 0.125f;
            pixels.emplace_back(values.size() * type.size());
            ASSERT_TRUE(OIIO::convert_types(OIIO::TypeDesc::FLOAT, values.data(), type,
                                            pixels.back().data(), values.size()));
            ASSERT_TRUE(output->write_image(type, pixels.back().data())) << output->geterror();
            width = std::max(1u, width / 2);
            height = std::max(1u, height / 2);
        }
        ASSERT_TRUE(output->close()) << output->geterror();
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

class OIIOReaderLazyFile : public OIIOReaderFiles,
    public ::testing::WithParamInterface<std::tuple<bool, unsigned int,
                                                    std::pair<unsigned int, unsigned int>>> {};

TEST_P(OIIOReaderLazyFile, CoarseOutOfOrderAndRepeatedReadsPreserveNativeAuthoredPixels) {
    const auto [half, channels, shape] = GetParam();
    const auto [width, height] = shape;
    const OIIO::TypeDesc type = half ? OIIO::TypeDesc::HALF : OIIO::TypeDesc::FLOAT;
    unsigned int levels = 1;
    for (unsigned int size = std::max(width, height); size > 1; size /= 2) ++levels;
    std::vector<std::vector<unsigned char>> expected;
    const auto filename = (directory / "lazy.exr").string();
    ASSERT_NO_FATAL_FAILURE(writePyramid(filename, width, height, type, channels, levels, expected));
    hip_demand::OIIOReader reader(filename);
    hip_demand::TextureInfo info;
    ASSERT_NO_THROW(reader.open(&info));
    ASSERT_EQ(info.numMipLevels, levels);
    EXPECT_EQ(info.format, half ? HIP_AD_FORMAT_HALF : HIP_AD_FORMAT_FLOAT);
    EXPECT_EQ(info.numChannels, channels);
    EXPECT_EQ(reader.getNumBytesRead(), 0u);

    std::vector<unsigned int> order{levels - 1, levels - 1, 0, levels / 2};
    for (unsigned int level = 0; level < levels; ++level) order.push_back(level);
    unsigned long long bytes = 0;
    for (const unsigned int level : order) {
        SCOPED_TRACE(level);
        constexpr size_t guard = 16;
        std::vector<unsigned char> actual(expected[level].size() + 2 * guard, 0xa5);
        ASSERT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(actual.data() + guard), level,
            std::max(1u, width >> level), std::max(1u, height >> level)));
        bytes += expected[level].size();
        EXPECT_EQ(reader.getNumBytesRead(), bytes);
        EXPECT_EQ(std::vector<unsigned char>(actual.begin() + guard, actual.end() - guard),
                  expected[level]);
        EXPECT_EQ(std::vector<unsigned char>(actual.begin(), actual.begin() + guard),
                  std::vector<unsigned char>(guard, 0xa5));
        EXPECT_EQ(std::vector<unsigned char>(actual.end() - guard, actual.end()),
                  std::vector<unsigned char>(guard, 0xa5));
    }
    reader.close();
    ASSERT_NO_THROW(reader.open(nullptr));
    std::vector<unsigned char> actual(expected.back().size());
    ASSERT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(actual.data()), levels - 1, 1, 1));
    EXPECT_EQ(actual, expected.back());
    EXPECT_EQ(reader.getNumBytesRead(), bytes + expected.back().size());
}

INSTANTIATE_TEST_SUITE_P(AuthoredShapesAndFormats, OIIOReaderLazyFile,
    ::testing::Combine(::testing::Bool(), ::testing::Values(1u, 3u, 4u),
        ::testing::Values(std::make_pair(16u, 16u), std::make_pair(19u, 11u),
                          std::make_pair(19u, 1u), std::make_pair(1u, 19u),
                          std::make_pair(1u, 1u))),
    [](const auto& parameter) {
        const auto half = std::get<0>(parameter.param);
        const auto channels = std::get<1>(parameter.param);
        const auto shape = std::get<2>(parameter.param);
        return std::string(half ? "Half" : "Float") + "C" + std::to_string(channels) +
               "_" + std::to_string(shape.first) + "x" + std::to_string(shape.second);
    });

TEST_F(OIIOReaderFiles, BaseColorReadsOnlyAuthoredCoarsestPixelEveryTime) {
    const auto filename = (directory / "base-color.exr").string();
    std::vector<std::vector<unsigned char>> pixels;
    ASSERT_NO_FATAL_FAILURE(writePyramid(filename, 16, 8, OIIO::TypeDesc::FLOAT, 4, 5, pixels));
    hip_demand::OIIOReader reader(filename);
    reader.open(nullptr);
    for (unsigned int repeat = 1; repeat <= 4; ++repeat) {
        float4 color{};
        ASSERT_TRUE(reader.readBaseColor(color));
        std::array<float, 4> expected;
        std::memcpy(expected.data(), pixels.back().data(), sizeof(expected));
        EXPECT_FLOAT_EQ(color.x, expected[0]);
        EXPECT_FLOAT_EQ(color.y, expected[1]);
        EXPECT_FLOAT_EQ(color.z, expected[2]);
        EXPECT_FLOAT_EQ(color.w, expected[3]);
        EXPECT_EQ(reader.getNumBytesRead(), repeat * sizeof(float4));
    }
}

TEST_F(OIIOReaderFiles, MissingFileAfterSuccessfulReadDoesNotReturnCachedPixels) {
    const auto filename = (directory / "removed.exr").string();
    std::vector<std::vector<unsigned char>> pixels;
    ASSERT_NO_FATAL_FAILURE(writePyramid(filename, 4, 4, OIIO::TypeDesc::FLOAT, 4, 3, pixels));
    hip_demand::OIIOReader reader(filename);
    reader.open(nullptr);
    float4 color{};
    ASSERT_TRUE(reader.readBaseColor(color));
    ASSERT_TRUE(std::filesystem::remove(filename));
    std::array<unsigned char, sizeof(float4)> actual;
    actual.fill(0xa5);
    const auto untouched = actual;
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(actual.data()), 2, 1, 1));
    EXPECT_EQ(actual, untouched);
    color = make_float4(1.f, 2.f, 3.f, 4.f);
    EXPECT_FALSE(reader.readBaseColor(color));
    EXPECT_FLOAT_EQ(color.x, 1.f);
    EXPECT_FLOAT_EQ(color.w, 4.f);
    EXPECT_EQ(reader.getNumBytesRead(), sizeof(float4));
    ASSERT_NO_FATAL_FAILURE(writePyramid(filename, 4, 4, OIIO::TypeDesc::FLOAT, 4, 3, pixels));
    EXPECT_TRUE(reader.readBaseColor(color));
    EXPECT_EQ(reader.getNumBytesRead(), 2 * sizeof(float4));
}

// A registered OIIO plugin exercises the production reader's error and metadata
// paths without corrupt-file heuristics or a test hook in the loader ABI.
struct ScriptedImage {
    std::vector<OIIO::ImageSpec> specs;
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<unsigned int> reads;
    void* destination = nullptr;
    int failedLevel = -1;
    int failedSeek = -1;
    bool throwRead = false;
    bool failClose = false;
    bool block = false;
    bool entered = false;
    unsigned int activeInputs = 0;
};

std::mutex scriptedImagesMutex;
std::map<std::string, std::shared_ptr<ScriptedImage>> scriptedImages;

class ScriptedInput : public OIIO::ImageInput {
public:
    const char* format_name() const override { return "hip_demand_lazy_test"; }
    bool open(const std::string& filename, OIIO::ImageSpec& spec) override {
        std::lock_guard<std::mutex> lock(scriptedImagesMutex);
        const auto found = scriptedImages.find(filename);
        if (found == scriptedImages.end()) return false;
        image = found->second;
        std::lock_guard<std::mutex> imageLock(image->mutex);
        ++image->activeInputs;
        m_spec = spec = image->specs.front();
        return true;
    }
    ~ScriptedInput() override { close(); }
    bool close() override {
        if (!image) return true;
        auto closing = std::move(image);
        std::lock_guard<std::mutex> lock(closing->mutex);
        --closing->activeInputs;
        closing->changed.notify_all();
        return !closing->failClose;
    }
    bool seek_subimage(int subimage, int mip) override {
        std::lock_guard<std::mutex> lock(image->mutex);
        if (image->failedSeek == mip) {
            errorfmt("scripted damaged mip metadata");
            return false;
        }
        if (subimage != 0 || mip < 0 || static_cast<size_t>(mip) >= image->specs.size())
            return false;
        m_spec = image->specs[mip];
        currentMip = mip;
        return true;
    }
    int current_miplevel() const override { return currentMip; }
    bool read_native_scanline(int, int, int, int, void*) override { return false; }
    bool read_image(int subimage, int mip, int chbegin, int chend, OIIO::TypeDesc type,
                    void* destination, OIIO::stride_t, OIIO::stride_t, OIIO::stride_t,
                    OIIO::ProgressCallback, void*) override {
        std::unique_lock<std::mutex> lock(image->mutex);
        image->reads.push_back(static_cast<unsigned int>(mip));
        image->destination = destination;
        image->entered = true;
        image->changed.notify_all();
        image->changed.wait(lock, [&] { return !image->block; });
        if (image->throwRead) throw std::runtime_error("scripted OIIO decode exception");
        if (image->failedLevel == mip) return false;
        if (subimage != 0 || chbegin != 0 || chend != m_spec.nchannels || type != m_spec.format)
            return false;
        const size_t components = static_cast<size_t>(m_spec.width) * m_spec.height * chend;
        std::vector<float> values(components, 20.f + mip);
        return OIIO::convert_types(OIIO::TypeDesc::FLOAT, values.data(), type,
                                    destination, components);
    }
private:
    std::shared_ptr<ScriptedImage> image;
    int currentMip = 0;
};

class OIIOReaderScripted : public OIIOReaderFiles {
protected:
    std::shared_ptr<ScriptedImage> image;
    std::string filename;
    void SetUp() override {
        OIIOReaderFiles::SetUp();
        static std::once_flag registration;
        std::call_once(registration, [] {
            static const char* extensions[] = {"hip_demand_lazy_test", nullptr};
            OIIO::declare_imageio_format("hip_demand_lazy_test",
                []() -> OIIO::ImageInput* { return new ScriptedInput; },
                extensions, nullptr, nullptr, "test");
        });
        filename = (directory / "scripted.hip_demand_lazy_test").string();
        std::ofstream file(filename);
        file << "test";
        file.close();
        image = std::make_shared<ScriptedImage>();
        for (int size : {8, 4, 2, 1})
            image->specs.emplace_back(size, size, 4, OIIO::TypeDesc::FLOAT);
        std::lock_guard<std::mutex> lock(scriptedImagesMutex);
        scriptedImages.emplace(filename, image);
    }
    void TearDown() override {
        {
            std::lock_guard<std::mutex> lock(scriptedImagesMutex);
            scriptedImages.erase(filename);
        }
        EXPECT_EQ(image->activeInputs, 0u);
        OIIOReaderFiles::TearDown();
    }
};

TEST_F(OIIOReaderScripted, CoarseReadUsesCallerStorageAndNeverReadsFineLevels) {
    hip_demand::OIIOReader reader(filename);
    reader.open(nullptr);
    EXPECT_TRUE(image->reads.empty());
    image->failedLevel = 0;
    std::array<float, 4> pixel{};
    for (unsigned int repeat = 1; repeat <= 10; ++repeat) {
        ASSERT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3, 1, 1));
        EXPECT_EQ(image->destination, pixel.data());
        EXPECT_EQ(pixel, (std::array<float, 4>{{23.f, 23.f, 23.f, 23.f}}));
        EXPECT_EQ(reader.getNumBytesRead(), repeat * sizeof(pixel));
    }
    EXPECT_EQ(image->reads, std::vector<unsigned int>(10, 3));
}

TEST_F(OIIOReaderScripted, PartialChainExposesOnlyAuthoredPrefixWithoutGeneratingPixels) {
    image->specs.resize(2);
    hip_demand::OIIOReader reader(filename);
    hip_demand::TextureInfo info;
    reader.open(&info);
    ASSERT_EQ(info.numMipLevels, 2u);
    std::array<float, 64> pixels{};
    ASSERT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(pixels.data()), 1, 4, 4));
    for (float value : pixels) EXPECT_FLOAT_EQ(value, 21.f);
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(pixels.data()), 2, 2, 2));
    float4 color = make_float4(1.f, 2.f, 3.f, 4.f);
    EXPECT_FALSE(reader.readBaseColor(color));
    EXPECT_FLOAT_EQ(color.x, 1.f);
    EXPECT_FLOAT_EQ(color.w, 4.f);
    EXPECT_EQ(image->reads, std::vector<unsigned int>({1}));
    EXPECT_EQ(reader.getNumBytesRead(), sizeof(pixels));
}

TEST_F(OIIOReaderScripted, InvalidRequestsDoNotReadOrWriteAndClosedReaderCanReopen) {
    hip_demand::OIIOReader reader(filename);
    std::array<float, 4> pixel{{-1.f, -2.f, -3.f, -4.f}};
    const auto untouched = pixel;
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3, 1, 1));
    reader.open(nullptr);
    EXPECT_FALSE(reader.readMipLevel(nullptr, 3, 1, 1));
    for (unsigned int level : {4u, 32u, std::numeric_limits<unsigned int>::max()})
        EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), level, 1, 1));
    for (const auto shape : {std::make_pair(0u, 1u), std::make_pair(1u, 0u),
                            std::make_pair(2u, 1u), std::make_pair(1u, 2u)})
        EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3,
                                         shape.first, shape.second));
    EXPECT_EQ(pixel, untouched);
    EXPECT_TRUE(image->reads.empty());
    EXPECT_EQ(reader.getNumBytesRead(), 0u);
    reader.close();
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3, 1, 1));
    reader.open(nullptr);
    EXPECT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3, 1, 1));
}

TEST_F(OIIOReaderScripted, FailedAndThrowingReadsAreNotCachedAndCanRetry) {
    hip_demand::OIIOReader reader(filename);
    reader.open(nullptr);
    std::array<float, 4> pixel{};
    image->failedLevel = 3;
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3, 1, 1));
    EXPECT_EQ(reader.getNumBytesRead(), 0u);
    EXPECT_EQ(image->activeInputs, 0u);
    image->throwRead = true;
    EXPECT_THROW(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3, 1, 1),
                 std::runtime_error);
    EXPECT_EQ(reader.getNumBytesRead(), 0u);
    EXPECT_EQ(image->activeInputs, 0u);
    image->failedLevel = -1;
    image->throwRead = false;
    ASSERT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3, 1, 1));
    EXPECT_EQ(reader.getNumBytesRead(), sizeof(pixel));
    EXPECT_EQ(image->reads, std::vector<unsigned int>({3, 3, 3}));
}

TEST_F(OIIOReaderScripted, MismatchedAuthoredFormatsFailMetadataOpenTransactionally) {
    image->specs[1].format = OIIO::TypeDesc::HALF;
    hip_demand::OIIOReader reader(filename);
    hip_demand::TextureInfo info;
    EXPECT_THROW(reader.open(&info), std::runtime_error);
    EXPECT_FALSE(reader.isOpen());
    EXPECT_FALSE(info.isValid);
    EXPECT_TRUE(image->reads.empty());
    EXPECT_EQ(image->activeInputs, 0u);
    image->specs[1].format = OIIO::TypeDesc::FLOAT;
    EXPECT_NO_THROW(reader.open(&info));
    EXPECT_TRUE(info.isValid);
}

TEST_F(OIIOReaderScripted, DamagedMetadataIsNotReportedAsAnAuthoredPartialChain) {
    image->failedSeek = 2;
    hip_demand::OIIOReader reader(filename);
    EXPECT_THROW(reader.open(nullptr), std::runtime_error);
    EXPECT_FALSE(reader.isOpen());
    EXPECT_TRUE(image->reads.empty());
    EXPECT_EQ(image->activeInputs, 0u);
    image->failedSeek = -1;
    reader.open(nullptr);
    image->failedSeek = 3;
    std::array<float, 4> pixel{};
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3, 1, 1));
    EXPECT_TRUE(image->reads.empty());
    EXPECT_EQ(reader.getNumBytesRead(), 0u);
    image->failedSeek = -1;
    EXPECT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3, 1, 1));
}

TEST_F(OIIOReaderScripted, ChangedBaseOrRequestedMetadataRejectsBeforeWriting) {
    hip_demand::OIIOReader reader(filename);
    reader.open(nullptr);
    const auto original = image->specs;
    std::array<float, 4> pixel{{-1.f, -2.f, -3.f, -4.f}};
    const auto untouched = pixel;
    for (unsigned int level : {0u, 3u}) {
        for (unsigned int field = 0; field < 5; ++field) {
            SCOPED_TRACE(::testing::Message() << "level=" << level << " field=" << field);
            image->specs = original;
            OIIO::ImageSpec& spec = image->specs[level];
            switch (field) {
            case 0: ++spec.width; break;
            case 1: ++spec.height; break;
            case 2: ++spec.depth; break;
            case 3: --spec.nchannels; break;
            case 4: spec.format = OIIO::TypeDesc::HALF; break;
            }
            EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3, 1, 1));
            EXPECT_EQ(pixel, untouched);
        }
    }
    image->specs = original;
    image->specs.resize(3);
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3, 1, 1));
    EXPECT_TRUE(image->reads.empty());
    EXPECT_EQ(reader.getNumBytesRead(), 0u);
    image->specs = original;
    EXPECT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3, 1, 1));
}

TEST_F(OIIOReaderScripted, CloseFailureIsNotSuccessfulReadOrSuccessfulMetadataOpen) {
    hip_demand::OIIOReader reader(filename);
    image->failClose = true;
    EXPECT_THROW(reader.open(nullptr), std::runtime_error);
    EXPECT_FALSE(reader.isOpen());
    image->failClose = false;
    reader.open(nullptr);
    image->failClose = true;
    std::array<float, 4> pixel{};
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3, 1, 1));
    EXPECT_EQ(reader.getNumBytesRead(), sizeof(pixel));
    EXPECT_EQ(image->activeInputs, 0u);
    image->failClose = false;
    EXPECT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3, 1, 1));
    EXPECT_EQ(reader.getNumBytesRead(), 2 * sizeof(pixel));
}

TEST_F(OIIOReaderScripted, ConcurrentReadsKeepSeparateDestinationsAndExactStatistics) {
    hip_demand::OIIOReader reader(filename);
    reader.open(nullptr);
    std::array<std::future<bool>, 12> reads;
    std::array<std::array<float, 16>, 12> pixels{};
    for (size_t index = 0; index < reads.size(); ++index) {
        reads[index] = std::async(std::launch::async, [&, index] {
            const unsigned int level = 2 + static_cast<unsigned int>(index % 2);
            const unsigned int size = 8 >> level;
            return reader.readMipLevel(reinterpret_cast<char*>(pixels[index].data()),
                                        level, size, size);
        });
    }
    for (size_t index = 0; index < reads.size(); ++index) {
        EXPECT_TRUE(reads[index].get());
        const size_t components = index % 2 == 0 ? 16 : 4;
        for (size_t component = 0; component < components; ++component)
            EXPECT_FLOAT_EQ(pixels[index][component], 22.f + static_cast<float>(index % 2));
    }
    EXPECT_EQ(image->activeInputs, 0u);
    EXPECT_EQ(image->reads.size(), reads.size());
    EXPECT_EQ(reader.getNumBytesRead(), 6 * (16 + 4) * sizeof(float));
}

TEST_F(OIIOReaderScripted, ClosingWaitsForBlockedDecodeAndKeepsInputAlive) {
    hip_demand::OIIOReader reader(filename);
    reader.open(nullptr);
    image->block = true;
    std::array<float, 4> pixel{};
    auto read = std::async(std::launch::async, [&] {
        return reader.readMipLevel(reinterpret_cast<char*>(pixel.data()), 3, 1, 1);
    });
    bool entered;
    {
        std::unique_lock<std::mutex> lock(image->mutex);
        entered = image->changed.wait_for(lock, std::chrono::seconds(5),
                                          [&] { return image->entered; });
    }
    EXPECT_TRUE(entered);
    std::promise<void> closing;
    auto close = std::async(std::launch::async, [&] {
        closing.set_value();
        reader.close();
    });
    closing.get_future().wait();
    EXPECT_EQ(close.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    {
        std::lock_guard<std::mutex> lock(image->mutex);
        EXPECT_EQ(image->activeInputs, 1u);
        image->block = false;
        image->changed.notify_all();
    }
    EXPECT_TRUE(read.get());
    close.get();
    EXPECT_FALSE(reader.isOpen());
    EXPECT_EQ(image->activeInputs, 0u);
    EXPECT_EQ(reader.getNumBytesRead(), sizeof(pixel));
}

} // namespace
