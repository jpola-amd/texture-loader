#pragma once

/// \file OIIOReader.h
/// OpenImageIO-based image reader implementation.

#include "ImageSource.h"
#include "TextureInfo.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace hip_demand {

class OIIOReader : public ImageSource
{
  public:
    explicit OIIOReader( const std::string& filename, bool readBaseColor = true );
    ~OIIOReader() override;

    void               open( TextureInfo* info ) override;
    void               close() override;
    bool               isOpen() const override;
    const TextureInfo& getInfo() const override;

    bool readTile( char* dest, uint32_t mipLevel, const Tile& tile, hipStream_t stream ) override;
    bool readMipLevel( char* dest, uint32_t mipLevel, uint32_t expectedWidth, uint32_t expectedHeight, hipStream_t stream = 0 ) override;
    bool readBaseColor( float4& dest ) override;

    unsigned long long getNumBytesRead() const override;
    double             getTotalReadTime() const override;
    unsigned long long getHash( hipStream_t stream = 0 ) const override;
    
    unsigned long long getNumTilesRead() const override;

  private:
    struct File;
    std::shared_ptr<File> acquireFile() const;
    void                  addStats( unsigned long long tiles, unsigned long long bytes, double seconds );

    std::string filename_;
    bool        readBaseColor_ = false;

    // shared_ptr so that a read can hold the file alive: reads copy it under mutex_ and then decode without the
    // lock, so a racing close() cannot free the input mid-decode.
    std::shared_ptr<File> file_;
    mutable std::mutex    mutex_;  // guards file_ only; reads are lock-free.

    TextureInfo  info_{};
    uint32_t bytesPerPixel_ = 0; 

    std::vector<uint2> levelSizes_;  // Width and height of each level.
    std::vector<uint2> tileSizes_;   // Tile size of each level; zero for scanline files.

    float4 baseColor_{};
    bool   baseColorWasRead_ = false;

    mutable std::mutex statsMutex_;
    unsigned long long numTilesRead_  = 0;
    unsigned long long numBytesRead_  = 0;
    double             totalReadTime_ = 0.0;
};

}  // namespace hip_demand
