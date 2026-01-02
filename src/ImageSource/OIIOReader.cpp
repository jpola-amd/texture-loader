#include <hip/hip_runtime.h>
#include "ImageSource/OIIOReader.h"
#include <OpenImageIO/imageio.h>
#include <algorithm>
#include <chrono>
#include <cstring>

namespace hip_demand {

OIIOReader::OIIOReader(const std::string& filename)
    : filename_(filename)
{
}

OIIOReader::~OIIOReader()
{
    close();
}

void OIIOReader::open(TextureInfo* info)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (isOpen_)
    {
        if (info) *info = info_;
        return;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Open image with OIIO
    auto inp = OIIO::ImageInput::open(filename_);
    if (!inp)
    {
        throw std::runtime_error("Failed to open image: " + filename_);
    }
    
    const OIIO::ImageSpec& spec = inp->spec();
    
    // Fill texture info
    info_.width = spec.width;
    info_.height = spec.height;
    info_.numChannels = spec.nchannels;
    info_.numMipLevels = calculateNumMipLevels(spec.width, spec.height);
    info_.isValid = true;
    info_.isTiled = false;
    
    // Determine format
    switch (spec.format.basetype)
    {
        case OIIO::TypeDesc::UINT8:
            info_.format = HIP_AD_FORMAT_UNSIGNED_INT8;
            break;
        case OIIO::TypeDesc::UINT16:
            info_.format = HIP_AD_FORMAT_UNSIGNED_INT16;
            break;
        case OIIO::TypeDesc::HALF:
            info_.format = HIP_AD_FORMAT_HALF;
            break;
        case OIIO::TypeDesc::FLOAT:
            info_.format = HIP_AD_FORMAT_FLOAT;
            break;
        default:
            // Default to UINT8 and let OIIO convert
            info_.format = HIP_AD_FORMAT_UNSIGNED_INT8;
            break;
    }
    
    inp->close();
    
    isOpen_ = true;
    
    if (info) *info = info_;
    
    auto end = std::chrono::high_resolution_clock::now();
    totalReadTime_ += std::chrono::duration<double>(end - start).count();
}

void OIIOReader::close()
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!isOpen_) return;
    
    mipLevels_.clear();
    isOpen_ = false;
}

bool OIIOReader::isOpen() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return isOpen_;
}

const TextureInfo& OIIOReader::getInfo() const
{
    return info_;
}

bool OIIOReader::loadImage()
{
    if (!isOpen_) return false;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Open image
    auto inp = OIIO::ImageInput::open(filename_);
    if (!inp) return false;
    
    const OIIO::ImageSpec& spec = inp->spec();
    
    // Read base level (level 0)
    size_t baseSize = spec.width * spec.height * info_.numChannels;
    std::vector<unsigned char> baseLevel(baseSize);
    
    // Read and convert to UINT8 RGBA
    bool success = inp->read_image(0, 0, 0, spec.nchannels, OIIO::TypeDesc::UINT8, baseLevel.data());
    inp->close();
    
    if (!success) return false;
    
    bytesRead_ += baseSize;
    
    // Store base level
    mipLevels_.resize(info_.numMipLevels);
    mipLevels_[0] = std::move(baseLevel);
    
    // Generate remaining mip levels
    int width = spec.width;
    int height = spec.height;
    
    for (unsigned int level = 1; level < info_.numMipLevels; ++level)
    {
        int prevWidth = width;
        int prevHeight = height;
        width = std::max(1, width / 2);
        height = std::max(1, height / 2);
        
        size_t levelSize = width * height * info_.numChannels;
        mipLevels_[level].resize(levelSize);
        
        generateMipLevel(mipLevels_[level - 1].data(), prevWidth, prevHeight,
                        mipLevels_[level].data(), width, height,
                        info_.numChannels);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    totalReadTime_ += std::chrono::duration<double>(end - start).count();
    
    return true;
}

void OIIOReader::generateMipLevel(const unsigned char* srcData, int srcWidth, int srcHeight,
                                  unsigned char* dstData, int dstWidth, int dstHeight,
                                  int channels)
{
    // Simple box filter downsampling
    for (int y = 0; y < dstHeight; ++y)
    {
        for (int x = 0; x < dstWidth; ++x)
        {
            int sx = x * 2;
            int sy = y * 2;
            
            for (int c = 0; c < channels; ++c)
            {
                int sum = 0;
                int count = 0;
                
                for (int dy = 0; dy < 2 && (sy + dy) < srcHeight; ++dy)
                {
                    for (int dx = 0; dx < 2 && (sx + dx) < srcWidth; ++dx)
                    {
                        sum += srcData[((sy + dy) * srcWidth + (sx + dx)) * channels + c];
                        count++;
                    }
                }
                
                dstData[(y * dstWidth + x) * channels + c] = sum / count;
            }
        }
    }
}

bool OIIOReader::readMipLevel(char* dest,
                              unsigned int mipLevel,
                              unsigned int expectedWidth,
                              unsigned int expectedHeight,
                              hipStream_t stream)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!isOpen_ || mipLevel >= info_.numMipLevels)
        return false;
    
    // Load image if not already loaded
    if (mipLevels_.empty())
    {
        if (!loadImage())
            return false;
    }
    
    // Verify dimensions
    unsigned int w = info_.width >> mipLevel;
    unsigned int h = info_.height >> mipLevel;
    w = std::max(1u, w);
    h = std::max(1u, h);
    
    if (w != expectedWidth || h != expectedHeight)
        return false;
    
    // Copy data
    size_t size = w * h * info_.numChannels;
    std::memcpy(dest, mipLevels_[mipLevel].data(), size);
    
    return true;
}

bool OIIOReader::readBaseColor(float4& dest)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!isOpen_) return false;
    
    // Load image if not already loaded
    if (mipLevels_.empty())
    {
        if (!loadImage())
            return false;
    }
    
    // Get 1x1 mip level (last level)
    unsigned int lastLevel = info_.numMipLevels - 1;
    const unsigned char* data = mipLevels_[lastLevel].data();
    
    // Convert to float4
    dest.x = data[0] / 255.0f;
    dest.y = (info_.numChannels > 1) ? data[1] / 255.0f : dest.x;
    dest.z = (info_.numChannels > 2) ? data[2] / 255.0f : dest.x;
    dest.w = (info_.numChannels > 3) ? data[3] / 255.0f : 1.0f;
    
    return true;
}

unsigned long long OIIOReader::getNumBytesRead() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return bytesRead_;
}

double OIIOReader::getTotalReadTime() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return totalReadTime_;
}

unsigned long long OIIOReader::getHash(hipStream_t /*stream*/) const
{
    // Hash the filename for content-based deduplication
    // Two ImageSource objects with the same filename should return the same hash
    return static_cast<unsigned long long>(std::hash<std::string>{}(filename_));
}

// Factory function implementation
std::unique_ptr<ImageSource> createImageSource(const std::string& filename)
{
    return std::make_unique<OIIOReader>(filename);
}

}  // namespace hip_demand
