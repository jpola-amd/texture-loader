#include <hip/hip_runtime.h>
#include "ImageSource/MipMapImageSource.h"
#include "../DemandLoading/Internal/ImageData.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace hip_demand {

namespace {

using internal::floatToHalf;
using internal::halfToFloat;
using internal::readChannel;

// Loads and stores one channel as a double; half goes through the float conversion.
template <typename T>
struct Sample
{
    static double load( const unsigned char* data ) { return static_cast<double>( readChannel<T>( data ) ); }
    static void   store( unsigned char* data, double value )
    {
        const T sample = static_cast<T>( value );
        std::memcpy( data, &sample, sizeof( sample ) );
    }
    static constexpr size_t bytes = sizeof( T );
};

struct HalfSample
{
    static double load( const unsigned char* data ) { return halfToFloat( readChannel<uint16_t>( data ) ); }
    static void   store( unsigned char* data, double value )
    {
        const uint16_t sample = floatToHalf( static_cast<float>( value ) );
        std::memcpy( data, &sample, sizeof( sample ) );
    }
    static constexpr size_t bytes = sizeof( uint16_t );
};

template <class Sample>
void generateMipLevelTyped( const unsigned char* srcData, uint32_t srcWidth, uint32_t srcHeight, unsigned char* dstData,
                            uint32_t dstWidth, uint32_t dstHeight, uint32_t channels )
{
    for( uint32_t y = 0; y < dstHeight; ++y )
    {
        for( uint32_t x = 0; x < dstWidth; ++x )
        {
            const uint32_t srcX = x * 2;
            const uint32_t srcY = y * 2;
            for( uint32_t channel = 0; channel < channels; ++channel )
            {
                double       sum   = 0.0;
                uint32_t count = 0;
                for( uint32_t dy = 0; dy < 2 && srcY + dy < srcHeight; ++dy )
                {
                    for( uint32_t dx = 0; dx < 2 && srcX + dx < srcWidth; ++dx )
                    {
                        const size_t srcSample = ( ( static_cast<size_t>( srcY + dy ) * srcWidth + srcX + dx ) * channels + channel ) * Sample::bytes;
                        sum += Sample::load( srcData + srcSample );
                        ++count;
                    }
                }
                const size_t dstSample = ( ( static_cast<size_t>( y ) * dstWidth + x ) * channels + channel ) * Sample::bytes;
                Sample::store( dstData + dstSample, sum / count );
            }
        }
    }
}

inline void generateMipLevel( hipArray_Format      format,
                              uint32_t         channels,
                              const unsigned char* srcData,
                              uint32_t         srcWidth,
                              uint32_t         srcHeight,
                              unsigned char*       dstData,
                              uint32_t         dstWidth,
                              uint32_t         dstHeight )
{
    switch( format )
    {
        case HIP_AD_FORMAT_UNSIGNED_INT8:
            return generateMipLevelTyped<Sample<uint8_t>>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
        case HIP_AD_FORMAT_SIGNED_INT8:
            return generateMipLevelTyped<Sample<int8_t>>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
        case HIP_AD_FORMAT_UNSIGNED_INT16:
            return generateMipLevelTyped<Sample<uint16_t>>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
        case HIP_AD_FORMAT_SIGNED_INT16:
            return generateMipLevelTyped<Sample<int16_t>>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
        case HIP_AD_FORMAT_UNSIGNED_INT32:
            return generateMipLevelTyped<Sample<uint32_t>>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
        case HIP_AD_FORMAT_SIGNED_INT32:
            return generateMipLevelTyped<Sample<int32_t>>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
        case HIP_AD_FORMAT_HALF:
            return generateMipLevelTyped<HalfSample>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
        case HIP_AD_FORMAT_FLOAT:
            return generateMipLevelTyped<Sample<float>>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
        default:
            throw std::runtime_error( "Cannot generate mip level for unsupported HIP array format: " + std::to_string( static_cast<int>( format ) ) );
    }
}

}  // namespace

MipMapImageSource::MipMapImageSource( std::shared_ptr<ImageSource> baseImage )
    : WrappedImageSource( std::move( baseImage ) )
{
    if( WrappedImageSource::isOpen() )
        cacheBaseInfo();
}

void MipMapImageSource::cacheBaseInfo()
{
    info_          = WrappedImageSource::getInfo();
    mipMappedBase_ = info_.numMipLevels > 1;
    bytesPerPixel_ = static_cast<size_t>( getBytesPerChannel( info_.format ) ) * info_.numChannels;
    if( mipMappedBase_ )
        return;
    info_.numMipLevels = calculateNumMipLevels( info_.width, info_.height );
    info_.isTiled      = true;
}

void MipMapImageSource::open( TextureInfo* info )
{
    std::unique_lock lock( mutex_ );
    if( !WrappedImageSource::isOpen() )
        WrappedImageSource::open( nullptr );
    cacheBaseInfo();
    if( info )
        *info = info_;
}

void MipMapImageSource::close()
{
    std::unique_lock lock( mutex_ );
    WrappedImageSource::close();
    levels_.clear();
    info_ = TextureInfo{};
}

const TextureInfo& MipMapImageSource::getInfo() const
{
    std::shared_lock lock( mutex_ );
    return mipMappedBase_ ? WrappedImageSource::getInfo() : info_;
}

bool MipMapImageSource::buildLevels( hipStream_t stream )
{
    if( !levels_.empty() )
        return true;
    if( !info_.isValid || bytesPerPixel_ == 0 )
        return false;

    std::vector<std::vector<char>> levels( info_.numMipLevels );
    for( uint32_t level = 0; level < info_.numMipLevels; ++level )
    {
        const uint32_t width  = std::max( 1u, info_.width >> level );
        const uint32_t height = std::max( 1u, info_.height >> level );
        levels[level].resize( static_cast<size_t>( width ) * height * bytesPerPixel_ );
    }

    if( !WrappedImageSource::readMipLevel( levels[0].data(), 0, info_.width, info_.height, stream ) )
        return false;

    for( uint32_t level = 1; level < info_.numMipLevels; ++level )
    {
        const uint32_t srcWidth  = std::max( 1u, info_.width >> ( level - 1 ) );
        const uint32_t srcHeight = std::max( 1u, info_.height >> ( level - 1 ) );
        generateMipLevel( info_.format, info_.numChannels, reinterpret_cast<const unsigned char*>( levels[level - 1].data() ),
                                    srcWidth, srcHeight, reinterpret_cast<unsigned char*>( levels[level].data() ),
                                    std::max( 1u, info_.width >> level ), std::max( 1u, info_.height >> level ) );
    }
    levels_ = std::move( levels );
    return true;
}

bool MipMapImageSource::readTile( char* dest, uint32_t mipLevel, const Tile& tile, hipStream_t stream )
{
    {
        std::shared_lock lock( mutex_ );
        if( mipMappedBase_ )
            return WrappedImageSource::readTile( dest, mipLevel, tile, stream );
    }
    if( !dest || tile.width == 0 || tile.height == 0 )
        return false;

    const auto copy = [&]( std::vector<std::vector<char>>& levels ) {
        if( mipLevel >= levels.size() )
            return false;
        const uint32_t width  = std::max( 1u, info_.width >> mipLevel );
        const uint32_t height = std::max( 1u, info_.height >> mipLevel );
        if( !internal::copyTileFromLevel( dest, tile, levels[mipLevel].data(), width, height, bytesPerPixel_ ) )
            return false;
        numTilesRead_.fetch_add( 1, std::memory_order_relaxed );
        return true;
    };

    {
        std::shared_lock lock( mutex_ );
        if( !levels_.empty() )
            return copy( levels_ );
    }
    {
        std::unique_lock lock( mutex_ );
        if( !buildLevels( stream ) )
            return false;
    }
    std::shared_lock lock( mutex_ );
    return !levels_.empty() && copy( levels_ );  // The chain may have been closed in between.
}

bool MipMapImageSource::readMipLevel( char* dest, uint32_t mipLevel, uint32_t expectedWidth, uint32_t expectedHeight, hipStream_t stream )
{
    {
        std::shared_lock lock( mutex_ );
        if( mipMappedBase_ )
            return WrappedImageSource::readMipLevel( dest, mipLevel, expectedWidth, expectedHeight, stream );
    }
    if( !dest )
        return false;

    const auto copy = [&]( std::vector<std::vector<char>>& levels ) {
        if( mipLevel >= levels.size() || expectedWidth != std::max( 1u, info_.width >> mipLevel )
            || expectedHeight != std::max( 1u, info_.height >> mipLevel ) )
            return false;
        std::copy( levels[mipLevel].begin(), levels[mipLevel].end(), dest );
        return true;
    };

    {
        std::shared_lock lock( mutex_ );
        if( !levels_.empty() )
            return copy( levels_ );
    }
    {
        std::unique_lock lock( mutex_ );
        if( !buildLevels( stream ) )
            return false;
    }
    std::shared_lock lock( mutex_ );
    return !levels_.empty() && copy( levels_ );
}

bool MipMapImageSource::readMipTail( char* dest, uint32_t mipTailFirstLevel, uint32_t numMipLevels, const uint2* mipLevelDims, hipStream_t stream )
{
    {
        std::shared_lock lock( mutex_ );
        if( mipMappedBase_ )
            return WrappedImageSource::readMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims, stream );
    }
    if( !dest || !mipLevelDims || mipTailFirstLevel >= numMipLevels )
        return false;

    {
        std::shared_lock lock( mutex_ );
        if( !levels_.empty() )
            return copyMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims );
    }
    {
        std::unique_lock lock( mutex_ );
        if( !buildLevels( stream ) )
            return false;
    }
    std::shared_lock lock( mutex_ );
    return !levels_.empty() && copyMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims );  // The chain may have been closed in between.
}

bool MipMapImageSource::copyMipTail( char* dest, uint32_t mipTailFirstLevel, uint32_t numMipLevels, const uint2* mipLevelDims ) const
{
    if( numMipLevels > levels_.size() )
        return false;
    size_t offset = 0;
    for( uint32_t level = mipTailFirstLevel; level < numMipLevels; ++level )
    {
        const std::vector<char>& data  = levels_[level];
        const size_t             bytes = static_cast<size_t>( mipLevelDims[level].x ) * mipLevelDims[level].y * bytesPerPixel_;
        if( data.size() != bytes )  // The caller's level size differs from the image's.
            return false;
        std::copy( data.begin(), data.end(), dest + offset );
        offset += bytes;
    }
    return true;
}

unsigned long long MipMapImageSource::getNumTilesRead() const
{
    std::shared_lock lock( mutex_ );
    return mipMappedBase_ ? WrappedImageSource::getNumTilesRead() : numTilesRead_.load( std::memory_order_relaxed );
}

}  // namespace hip_demand
