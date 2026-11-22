#pragma once

/// \file WrappedImageSource.h
/// An ImageSource that delegates every call to another one; the base for adapters that change part of its behaviour.

#include "ImageSource.h"

#include <cstdint>
#include <memory>
#include <utility>

namespace hip_demand {

class WrappedImageSource : public ImageSource
{
  public:
    explicit WrappedImageSource( std::shared_ptr<ImageSource> imageSource )
        : imageSource_( std::move( imageSource ) )
    {
    }

    ~WrappedImageSource() override = default;

    void               open( TextureInfo* info ) override { imageSource_->open( info ); }
    void               close() override { imageSource_->close(); }
    bool               isOpen() const override { return imageSource_->isOpen(); }
    const TextureInfo& getInfo() const override { return imageSource_->getInfo(); }

    bool readMipLevel( char* dest, uint32_t mipLevel, uint32_t expectedWidth, uint32_t expectedHeight, hipStream_t stream = 0 ) override
    {
        return imageSource_->readMipLevel( dest, mipLevel, expectedWidth, expectedHeight, stream );
    }

    bool readTile( char* dest, uint32_t mipLevel, const Tile& tile, hipStream_t stream ) override
    {
        return imageSource_->readTile( dest, mipLevel, tile, stream );
    }

    bool readMipTail( char* dest, uint32_t mipTailFirstLevel, uint32_t numMipLevels, const uint2* mipLevelDims, hipStream_t stream ) override
    {
        return imageSource_->readMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims, stream );
    }

    bool               readBaseColor( float4& dest ) override { return imageSource_->readBaseColor( dest ); }
    unsigned long long getNumTilesRead() const override { return imageSource_->getNumTilesRead(); }
    unsigned long long getNumBytesRead() const override { return imageSource_->getNumBytesRead(); }
    double             getTotalReadTime() const override { return imageSource_->getTotalReadTime(); }
    unsigned long long getHash( hipStream_t stream = 0 ) const override { return imageSource_->getHash( stream ); }
    const std::shared_ptr<ImageSource>& getBaseImage() const { return imageSource_; }

  private:
    std::shared_ptr<ImageSource> imageSource_;
};

}  // namespace hip_demand
