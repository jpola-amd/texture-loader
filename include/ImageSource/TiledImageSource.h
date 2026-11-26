#pragma once

/// \file TiledImageSource.h
/// Serves tiles of an image source that cannot read them from its file.

#include "TextureInfo.h"
#include "WrappedImageSource.h"

#include <memory>
#include <cstdint>
#include <atomic>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace hip_demand {

/// Adapts a non-tiled image source to tile reads: the first tile read of a mip level reads the whole level into
/// memory through readMipLevel(), and tiles are copied out of it from then on. A tiled base image is passed through
/// untouched.
class TiledImageSource : public WrappedImageSource
{
  public:
    explicit TiledImageSource( std::shared_ptr<ImageSource> baseImage );
    ~TiledImageSource() override = default;

    void               open( TextureInfo* info ) override;
    void               close() override;
    const TextureInfo& getInfo() const override;

    bool readTile( char* dest, uint32_t mipLevel, const Tile& tile, hipStream_t stream ) override;
    bool readMipTail( char* dest, uint32_t mipTailFirstLevel, uint32_t numMipLevels, const uint2* mipLevelDims, hipStream_t stream ) override;

    /// Number of tiles served, by the base image when it is tiled or by this adapter otherwise.
    unsigned long long getNumTilesRead() const override;

  private:
    void cacheBaseInfo();
    bool loadLevel( uint32_t mipLevel, hipStream_t stream );
    bool copyTile( char* dest, uint32_t mipLevel, const Tile& tile );
    bool copyMipTail( char* dest, uint32_t mipTailFirstLevel, uint32_t numMipLevels, const uint2* mipLevelDims ) const;

    mutable std::shared_mutex      mutex_;
    bool                           baseIsTiled_ = false;
    TextureInfo                    info_{};
    std::vector<std::vector<char>> levels_;
    std::atomic<unsigned long long> numTilesRead_{ 0 };
};

inline std::shared_ptr<ImageSource> createTiledImageSource( std::shared_ptr<ImageSource> baseImage )
{
    if( !baseImage )
        return {};
    if( !baseImage->isOpen() )
        baseImage->open( nullptr );
    if( baseImage->getInfo().isTiled )
        return baseImage;
    return std::make_shared<TiledImageSource>( std::move( baseImage ) );
}

}  // namespace hip_demand
