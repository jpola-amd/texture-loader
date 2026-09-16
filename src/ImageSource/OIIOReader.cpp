#include <hip/hip_runtime.h>
#include "ImageSource/OIIOReader.h"
#include "../DemandLoading/Internal/ImageData.h"
#include <OpenImageIO/imageio.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <stdexcept>

namespace hip_demand {
namespace {

hipArray_Format imageFormat(OIIO::TypeDesc type) {
    switch (type.basetype) {
    case OIIO::TypeDesc::UINT8: return HIP_AD_FORMAT_UNSIGNED_INT8;
    case OIIO::TypeDesc::INT8: return HIP_AD_FORMAT_SIGNED_INT8;
    case OIIO::TypeDesc::UINT16: return HIP_AD_FORMAT_UNSIGNED_INT16;
    case OIIO::TypeDesc::INT16: return HIP_AD_FORMAT_SIGNED_INT16;
    case OIIO::TypeDesc::UINT32: return HIP_AD_FORMAT_UNSIGNED_INT32;
    case OIIO::TypeDesc::INT32: return HIP_AD_FORMAT_SIGNED_INT32;
    case OIIO::TypeDesc::HALF: return HIP_AD_FORMAT_HALF;
    default: return HIP_AD_FORMAT_FLOAT;
    }
}

OIIO::TypeDesc pixelType(hipArray_Format format) {
    switch (format) {
    case HIP_AD_FORMAT_UNSIGNED_INT8: return OIIO::TypeDesc::UINT8;
    case HIP_AD_FORMAT_SIGNED_INT8: return OIIO::TypeDesc::INT8;
    case HIP_AD_FORMAT_UNSIGNED_INT16: return OIIO::TypeDesc::UINT16;
    case HIP_AD_FORMAT_SIGNED_INT16: return OIIO::TypeDesc::INT16;
    case HIP_AD_FORMAT_UNSIGNED_INT32: return OIIO::TypeDesc::UINT32;
    case HIP_AD_FORMAT_SIGNED_INT32: return OIIO::TypeDesc::INT32;
    case HIP_AD_FORMAT_HALF: return OIIO::TypeDesc::HALF;
    case HIP_AD_FORMAT_FLOAT: return OIIO::TypeDesc::FLOAT;
    default: throw std::invalid_argument("Unsupported OIIO image channel format");
    }
}

size_t levelBytes(const TextureInfo& info, unsigned int width, unsigned int height) {
    return internal::imageByteSize(width, height, info.numChannels,
                                   getBytesPerChannel(info.format));
}

bool matchesLevel(const OIIO::ImageSpec& spec, const TextureInfo& info,
                  unsigned int level) {
    return spec.width == static_cast<int>(std::max(1u, info.width >> level)) &&
           spec.height == static_cast<int>(std::max(1u, info.height >> level)) &&
           spec.depth == 1 && spec.nchannels == static_cast<int>(info.numChannels) &&
           imageFormat(spec.format) == info.format;
}

} // namespace

OIIOReader::OIIOReader(const std::string& filename) : filename_(filename) {}
OIIOReader::~OIIOReader() { close(); }

void OIIOReader::open(TextureInfo* info) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (isOpen_) {
        if (info) *info = info_;
        return;
    }
    const auto start = std::chrono::high_resolution_clock::now();
    auto input = OIIO::ImageInput::open(filename_);
    if (!input) throw std::runtime_error("Failed to open image: " + filename_);
    const OIIO::ImageSpec spec = input->spec();
    if (spec.width <= 0 || spec.height <= 0 || spec.depth != 1 ||
        spec.nchannels < 1 || spec.nchannels > 4)
        throw std::runtime_error("Unsupported OIIO texture dimensions or channels: " + filename_);

    TextureInfo opened{};
    opened.width = static_cast<unsigned int>(spec.width);
    opened.height = static_cast<unsigned int>(spec.height);
    opened.numChannels = static_cast<unsigned int>(spec.nchannels);
    opened.format = imageFormat(spec.format);
    opened.isTiled = spec.tile_width > 0;
    opened.numMipLevels = 1;
    levelBytes(opened, opened.width, opened.height);
    unsigned int fullLevels = 1;
    for (unsigned int size = std::max(opened.width, opened.height); size > 1; size /= 2)
        ++fullLevels;
    for (unsigned int level = 1; level < fullLevels; ++level) {
        if (!input->seek_subimage(0, static_cast<int>(level))) {
            const std::string error = input->geterror();
            if (!error.empty())
                throw std::runtime_error("Failed to read mip metadata: " + filename_ + ": " + error);
            break;
        }
        const OIIO::ImageSpec mipSpec = input->spec();
        if (mipSpec.format != spec.format || mipSpec.channelformats != spec.channelformats)
            throw std::runtime_error("Inconsistent OIIO mip pixel formats: " + filename_);
        if (mipSpec.width != static_cast<int>(std::max(1u, opened.width >> level)) ||
            mipSpec.height != static_cast<int>(std::max(1u, opened.height >> level)) ||
            mipSpec.depth != 1 || mipSpec.nchannels != static_cast<int>(opened.numChannels))
            // HIP arrays halve dimensions with rounding down. A valid EXR can round up;
            // expose its compatible prefix and let the loader generate the remaining levels.
            break;
        ++opened.numMipLevels;
    }
    opened.isValid = true;
    if (!input->close())
        throw std::runtime_error("Failed to close image metadata: " + filename_ +
                                 ": " + input->geterror());
    info_ = opened;
    isOpen_ = true;
    if (info) *info = info_;
    totalReadTime_ += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start).count();
}

void OIIOReader::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    mipLevels_.clear();
    isOpen_ = false;
}

bool OIIOReader::isOpen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return isOpen_;
}

const TextureInfo& OIIOReader::getInfo() const { return info_; }

bool OIIOReader::readMipLevelLocked(char* dest, unsigned int level) {
    const auto start = std::chrono::high_resolution_clock::now();
    auto input = OIIO::ImageInput::open(filename_);
    if (!input) return false;
    const OIIO::ImageSpec baseSpec = input->spec();
    if (!matchesLevel(baseSpec, info_, 0) ||
        (baseSpec.tile_width > 0) != info_.isTiled)
        return false;
    if (!input->seek_subimage(0, static_cast<int>(level))) return false;
    const OIIO::ImageSpec spec = input->spec();
    if (!matchesLevel(spec, info_, level) || spec.format != baseSpec.format ||
        spec.channelformats != baseSpec.channelformats)
        return false;

    const unsigned int width = std::max(1u, info_.width >> level);
    const unsigned int height = std::max(1u, info_.height >> level);
    const size_t size = levelBytes(info_, width, height);
    const OIIO::TypeDesc type = pixelType(info_.format);
    const bool read = input->read_image(0, static_cast<int>(level), 0, spec.nchannels,
                                        type, dest);
    if (read) bytesRead_ += size;
    const bool closed = input->close();
    totalReadTime_ += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start).count();
    return read && closed;
}

bool OIIOReader::readMipLevel(char* dest, unsigned int level, unsigned int expectedWidth,
                              unsigned int expectedHeight, hipStream_t /*stream*/) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!dest || !isOpen_ || level >= info_.numMipLevels) return false;
    const unsigned int width = std::max(1u, info_.width >> level);
    const unsigned int height = std::max(1u, info_.height >> level);
    if (width != expectedWidth || height != expectedHeight) return false;
    return readMipLevelLocked(dest, level);
}

bool OIIOReader::readBaseColor(float4& dest) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!isOpen_ || std::max(1u, info_.width >> (info_.numMipLevels - 1)) != 1 ||
        std::max(1u, info_.height >> (info_.numMipLevels - 1)) != 1)
        return false;
    alignas(float) std::array<unsigned char, 4 * sizeof(float)> pixel{};
    if (!readMipLevelLocked(reinterpret_cast<char*>(pixel.data()), info_.numMipLevels - 1))
        return false;
    std::array<float, 4> color{{0.f, 0.f, 0.f, 1.f}};
    for (unsigned int channel = 0; channel < info_.numChannels; ++channel)
        color[channel] = internal::floatChannel(
            pixel.data() + channel * getBytesPerChannel(info_.format), info_.format);
    dest = make_float4(color[0], color[1], color[2], color[3]);
    return true;
}

unsigned long long OIIOReader::getNumBytesRead() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return bytesRead_;
}

double OIIOReader::getTotalReadTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return totalReadTime_;
}

unsigned long long OIIOReader::getHash(hipStream_t /*stream*/) const {
    return static_cast<unsigned long long>(std::hash<std::string>{}(filename_));
}

std::unique_ptr<ImageSource> createImageSource(const std::string& filename) {
    return std::make_unique<OIIOReader>(filename);
}

} // namespace hip_demand
