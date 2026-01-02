#pragma once

/// @file ImageSource.h
/// @brief Interface for mipmapped image loading.
/// @note This header requires HIP types. Include <hip/hip_runtime.h> before this header.

#include <memory>
#include <string>

namespace hip_demand {

struct TextureInfo;

/// Interface for a mipmapped image source.
/// All methods must be thread-safe.
class ImageSource
{
  public:
    virtual ~ImageSource() = default;

    /// Open the image and read header info. Throws on error.
    virtual void open(TextureInfo* info) = 0;

    /// Close the image.
    virtual void close() = 0;

    /// Check if image is currently open.
    virtual bool isOpen() const = 0;

    /// Get the image info. Valid only after calling open().
    virtual const TextureInfo& getInfo() const = 0;

    /// Read the specified mip level into dest buffer.
    /// dest must be large enough to hold the mip level data.
    /// Returns true if successful.
    virtual bool readMipLevel(char* dest, 
                             unsigned int mipLevel,
                             unsigned int expectedWidth,
                             unsigned int expectedHeight,
                             hipStream_t stream = 0) = 0;

    /// Read the base color (1x1 mip level) as float4. Returns true on success.
    virtual bool readBaseColor(float4& dest) = 0;

    /// Returns the number of bytes read from disk.
    virtual unsigned long long getNumBytesRead() const = 0;

    /// Returns the time in seconds spent reading image data.
    virtual double getTotalReadTime() const = 0;

    /// Returns a hash that uniquely identifies the image source content.
    /// Used for deduplication - two ImageSource objects with the same hash
    /// are assumed to produce identical image data.
    /// Default implementation returns 0 (no deduplication by content).
    virtual unsigned long long getHash(hipStream_t stream = 0) const { (void)stream; return 0; }
};

/// Calculate number of mip levels for given dimensions
inline unsigned int calculateNumMipLevels(unsigned int width, unsigned int height)
{
    unsigned int dim = (width > height) ? width : height;
    return 1 + static_cast<unsigned int>(std::log2f(static_cast<float>(dim)));
}

/// Factory function to create image source from file
std::unique_ptr<ImageSource> createImageSource(const std::string& filename);

}  // namespace hip_demand
