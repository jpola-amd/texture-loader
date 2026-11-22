#pragma once

/// \file MipMapImageSource.h
/// Gives an image source without mip levels a full mip chain built in memory.

#include "TextureInfo.h"
#include "WrappedImageSource.h"

#include <memory>
#include <cstdint>
#include <atomic>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace hip_demand {

/// Adapts an image source with a single level to a full mip chain: the first read loads the base level into memory
/// through readMipLevel() and box-filters the coarser levels from it; tiles and levels are then served from memory,
/// so the adapted image reports itself as tiled. A base image that already has mip levels is passed through
/// untouched.
class MipMapImageSource : public WrappedImageSource
{
  public:
    explicit MipMapImageSource( std::shared_ptr<ImageSource> baseImage );
    ~MipMapImageSource() override = default;

    void               open( TextureInfo* info ) override;
    void               close() override;
    const TextureInfo& getInfo() const override;

    bool readTile( char* dest, uint32_t mipLevel, const Tile& tile, hipStream_t stream ) override;
    bool readMipLevel( char* dest, uint32_t mipLevel, uint32_t expectedWidth, uint32_t expectedHeight, hipStream_t stream = 0 ) override;
    bool readMipTail( char* dest, uint32_t mipTailFirstLevel, uint32_t numMipLevels, const uint2* mipLevelDims, hipStream_t stream ) override;
    unsigned long long getNumTilesRead() const override;

  private:
    void cacheBaseInfo();
    bool buildLevels( hipStream_t stream );
    bool copyMipTail( char* dest, uint32_t mipTailFirstLevel, uint32_t numMipLevels, const uint2* mipLevelDims ) const;

    mutable std::shared_mutex      mutex_;
    bool                           mipMappedBase_ = false;
    TextureInfo                    info_{};
    size_t                         bytesPerPixel_ = 0;
    std::vector<std::vector<char>> levels_;
    std::atomic<unsigned long long> numTilesRead_{ 0 };
};

inline std::shared_ptr<ImageSource> createMipMapImageSource( std::shared_ptr<ImageSource> baseImage )
{
    if( !baseImage )
        return {};
    if( !baseImage->isOpen() )
        baseImage->open( nullptr );
    if( baseImage->getInfo().numMipLevels > 1 )
        return baseImage;
    return std::make_shared<MipMapImageSource>( std::move( baseImage ) );
}

}  // namespace hip_demand
