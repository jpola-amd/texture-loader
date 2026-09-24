#pragma once

/// \file OIIOReader.h
/// OpenImageIO-based image reader implementation

#include "ImageSource.h"
#include "TextureInfo.h"
#include <optional>
#include <shared_mutex>
#include <vector>

namespace hip_demand {

/// Image reader using OpenImageIO
/// Supports many formats: PNG, JPEG, TIFF, EXR, HDR, TGA, BMP, etc.
class OIIOReader : public ImageSource
{
  public:
    /// Constructor
    explicit OIIOReader( const std::string& filename, std::optional<hipArray_Format> outputFormat = std::nullopt );
    
    /// Destructor
    ~OIIOReader() override;

    // ImageSource interface
    void open(TextureInfo* info) override;
    void close() override;
    bool isOpen() const override;
    const TextureInfo& getInfo() const override;
    
    bool readMipLevel(char* dest,
                     unsigned int mipLevel,
                     unsigned int expectedWidth,
                     unsigned int expectedHeight,
                     hipStream_t stream = 0) override;

    bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, hipStream_t stream ) override;
    
    bool readBaseColor(float4& dest) override;
    
    unsigned long long getNumBytesRead() const override;
    double getTotalReadTime() const override;
    
    /// Returns a hash based on the filename for deduplication
    unsigned long long getHash(hipStream_t stream = 0) const override;

  private:
    bool readMipLevelNoLock( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight );
    bool readTileNoLock( char* dest, unsigned int mipLevel, const Tile& tile );

    std::string filename_;
    std::optional<hipArray_Format> outputFormat_;
    TextureInfo info_;
    bool isOpen_ = false;
    
    mutable std::shared_mutex mutex_;
    unsigned long long bytesRead_ = 0;
    double totalReadTime_ = 0.0;
    
    // Cached image data (all mip levels)
    std::vector<std::vector<unsigned char>> mipLevels_;
    
    // Load entire image with all mip levels
    bool loadImage();
    
    // Generate mip level from previous level
    void generateMipLevel( const unsigned char* srcData, int srcWidth, int srcHeight, unsigned char* dstData, int dstWidth,
                           int dstHeight, int channels );
};

}  // namespace hip_demand
