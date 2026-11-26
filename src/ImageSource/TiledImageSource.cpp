#include <hip/hip_runtime.h>
#include "ImageSource/TiledImageSource.h"
#include "../DemandLoading/Internal/ImageData.h"

#include <algorithm>
#include <mutex>

namespace hip_demand {

TiledImageSource::TiledImageSource( std::shared_ptr<ImageSource> baseImage )
    : WrappedImageSource( std::move( baseImage ) )
{
    if( WrappedImageSource::isOpen() )
        cacheBaseInfo();
}

void TiledImageSource::cacheBaseInfo()
{
    info_         = WrappedImageSource::getInfo();
    baseIsTiled_  = info_.isTiled;
    info_.isTiled = true;
    levels_.assign( info_.numMipLevels, {} );
}

void TiledImageSource::open( TextureInfo* info )
{
    std::unique_lock lock( mutex_ );
    if( !WrappedImageSource::isOpen() )
        WrappedImageSource::open( nullptr );
    cacheBaseInfo();
    if( info )
        *info = info_;
}

void TiledImageSource::close()
{
    std::unique_lock lock( mutex_ );
    WrappedImageSource::close();
    levels_.clear();
    info_ = TextureInfo{};
}

const TextureInfo& TiledImageSource::getInfo() const
{
    std::shared_lock lock( mutex_ );
    return baseIsTiled_ ? WrappedImageSource::getInfo() : info_;
}

bool TiledImageSource::loadLevel( uint32_t mipLevel, hipStream_t stream )
{
    if( mipLevel >= levels_.size() )
        return false;
    std::vector<char>& level = levels_[mipLevel];
    if( !level.empty() )
        return true;

    const uint32_t width  = std::max( 1u, info_.width >> mipLevel );
    const uint32_t height = std::max( 1u, info_.height >> mipLevel );
    const size_t       bytes  = static_cast<size_t>( width ) * height * getBytesPerChannel( info_.format ) * info_.numChannels;
    level.resize( bytes );
    if( WrappedImageSource::readMipLevel( level.data(), mipLevel, width, height, stream ) )
        return true;
    level.clear();
    return false;
}

bool TiledImageSource::readTile( char* dest, uint32_t mipLevel, const Tile& tile, hipStream_t stream )
{
    {
        std::shared_lock lock( mutex_ );
        if( baseIsTiled_ )
            return WrappedImageSource::readTile( dest, mipLevel, tile, stream );
    }
    if( !dest || tile.width == 0 || tile.height == 0 )
        return false;

    {
        std::shared_lock lock( mutex_ );
        if( mipLevel < levels_.size() && !levels_[mipLevel].empty() )
            return copyTile( dest, mipLevel, tile );
    }
    {
        std::unique_lock lock( mutex_ );
        if( !loadLevel( mipLevel, stream ) )
            return false;
    }
    std::shared_lock lock( mutex_ );
    // The level may have been closed in between.
    return mipLevel < levels_.size() && !levels_[mipLevel].empty() && copyTile( dest, mipLevel, tile );
}

bool TiledImageSource::copyTile( char* dest, uint32_t mipLevel, const Tile& tile )
{
    const uint32_t width  = std::max( 1u, info_.width >> mipLevel );
    const uint32_t height = std::max( 1u, info_.height >> mipLevel );
    const size_t       bytesPerPixel = static_cast<size_t>( getBytesPerChannel( info_.format ) ) * info_.numChannels;
    if( !internal::copyTileFromLevel( dest, tile, levels_[mipLevel].data(), width, height, bytesPerPixel ) )
        return false;
    numTilesRead_.fetch_add( 1, std::memory_order_relaxed );
    return true;
}

bool TiledImageSource::readMipTail( char* dest, uint32_t mipTailFirstLevel, uint32_t numMipLevels, const uint2* mipLevelDims, hipStream_t stream )
{
    {
        std::shared_lock lock( mutex_ );
        if( baseIsTiled_ )
            return WrappedImageSource::readMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims, stream );
    }
    if( !dest || !mipLevelDims || mipTailFirstLevel >= numMipLevels )
        return false;

    {
        std::unique_lock lock( mutex_ );
        for( uint32_t level = mipTailFirstLevel; level < numMipLevels; ++level )
        {
            if( !loadLevel( level, stream ) )
                return false;
        }
    }
    std::shared_lock lock( mutex_ );
    return copyMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims );
}

bool TiledImageSource::copyMipTail( char* dest, uint32_t mipTailFirstLevel, uint32_t numMipLevels, const uint2* mipLevelDims ) const
{
    if( numMipLevels > levels_.size() )
        return false;
    const size_t bytesPerPixel = static_cast<size_t>( getBytesPerChannel( info_.format ) ) * info_.numChannels;
    size_t       offset        = 0;
    for( uint32_t level = mipTailFirstLevel; level < numMipLevels; ++level )
    {
        const std::vector<char>& data = levels_[level];
        const size_t             bytes = static_cast<size_t>( mipLevelDims[level].x ) * mipLevelDims[level].y * bytesPerPixel;
        if( data.size() != bytes )  // Closed in between, or the caller's level size differs from the image's.
            return false;
        std::copy( data.begin(), data.end(), dest + offset );
        offset += bytes;
    }
    return true;
}

unsigned long long TiledImageSource::getNumTilesRead() const
{
    std::shared_lock lock( mutex_ );
    return baseIsTiled_ ? WrappedImageSource::getNumTilesRead() : numTilesRead_.load( std::memory_order_relaxed );
}

}  // namespace hip_demand
