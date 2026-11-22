#pragma once

/// @file ImageSource.h
/// @brief Interface for mipmapped image loading.
/// @note This header requires HIP types. Include <hip/hip_runtime.h> before this header.

#include "TextureInfo.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>

namespace hip_demand {

struct Tile
{
    unsigned int x;
    unsigned int y;
    unsigned int width;
    unsigned int height;
};

/// Interface for a mipmapped image source.
/// All methods must be thread-safe.
class ImageSource
{
  public:
    virtual ~ImageSource() = default;

    /// Open the image and read header info. Throws on error.
    virtual void open( TextureInfo* info ) = 0;

    /// Close the image.
    virtual void close() = 0;

    /// Check if image is currently open.
    virtual bool isOpen() const = 0;

    /// Get the image info. Valid only after calling open().
    virtual const TextureInfo& getInfo() const = 0;

    /// Read the specified mip level into dest buffer.
    /// dest must be large enough to hold the mip level data.
    /// Returns true if successful.
    virtual bool readMipLevel( char*        dest,
                               unsigned int mipLevel,
                               unsigned int expectedWidth,
                               unsigned int expectedHeight,
                               hipStream_t  stream = 0 ) = 0;

    /// Reads one tile of a mip level into dest, tile.width * tile.height texels in row-major order.
    virtual bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, hipStream_t stream ) = 0;

    /// Read the mip levels from mipTailFirstLevel to numMipLevels - 1 into dest, packed one after another with no
    /// padding; mipLevelDims[level] gives each level's size. The default reads them one at a time with readMipLevel()
    /// and stops at the first failure. Returns true if successful.
    virtual bool readMipTail( char*        dest,
                              unsigned int mipTailFirstLevel,
                              unsigned int numMipLevels,
                              const uint2* mipLevelDims,
                              hipStream_t  stream );

    /// Read the base color (1x1 mip level) as float4. Returns true on success.
    virtual bool readBaseColor( float4& dest ) = 0;

    /// Returns the number of tiles read, counting a whole level as one. Zero when not tracked.
    virtual unsigned long long getNumTilesRead() const { return 0; }

    /// Returns the number of bytes read from disk.
    virtual unsigned long long getNumBytesRead() const = 0;

    /// Returns the time in seconds spent reading image data.
    virtual double getTotalReadTime() const = 0;

    /// Returns a hash that uniquely identifies the image source content.
    /// Used for deduplication - two ImageSource objects with the same hash
    /// are assumed to produce identical image data.
    /// Default implementation returns 0 (no deduplication by content).
    virtual unsigned long long getHash( hipStream_t stream = 0 ) const
    {
        (void)stream;
        return 0;
    }
};

inline bool ImageSource::readMipTail( char* dest, unsigned int mipTailFirstLevel, unsigned int numMipLevels, const uint2* mipLevelDims, hipStream_t stream )
{
    const TextureInfo& info          = getInfo();
    const size_t       bytesPerPixel = static_cast<size_t>( getBytesPerChannel( info.format ) ) * info.numChannels;
    size_t             offset        = 0;
    for( unsigned int level = mipTailFirstLevel; level < numMipLevels; ++level )
    {
        const uint2 dims = mipLevelDims[level];
        if( !readMipLevel( dest + offset, level, dims.x, dims.y, stream ) )
            return false;
        offset += static_cast<size_t>( dims.x ) * dims.y * bytesPerPixel;
    }
    return true;
}

/// Calculate number of mip levels for given dimensions
inline unsigned int calculateNumMipLevels( unsigned int width, unsigned int height )
{
    unsigned int dim = ( width > height ) ? width : height;
    return 1 + static_cast<unsigned int>( std::log2f( static_cast<float>( dim ) ) );
}

/// Factory function to create an image source from a file.
std::unique_ptr<ImageSource> createImageSource( const std::string& filename );

}  // namespace hip_demand
