#pragma once

/// \file OIIOReader.h
/// OpenImageIO-based image reader implementation

#include "ImageSource.h"
#include "TextureInfo.h"
#include <mutex>
#include <vector>

namespace hip_demand {

/// Image reader using OpenImageIO
/// Supports many formats: PNG, JPEG, TIFF, EXR, HDR, TGA, BMP, etc.
class OIIOReader : public ImageSource
{
  public:
    /// Constructor
    explicit OIIOReader(const std::string& filename);
    
    /// Destructor
    ~OIIOReader() override;

    // ImageSource interface
    void open(TextureInfo* info) override;
    void close() override;
    bool isOpen() const override;
    const TextureInfo& getInfo() const override;
    
    /// Decode only this authored level directly into caller-owned storage.
    /// No decoded pixels are retained. On an I/O failure dest may be partially written.
    bool readMipLevel(char* dest,
                     unsigned int mipLevel,
                     unsigned int expectedWidth,
                     unsigned int expectedHeight,
                     hipStream_t stream = 0) override;
    
    bool readBaseColor(float4& dest) override;
    
    /// Successfully decoded native payload bytes, including repeated reads.
    /// Excludes metadata, compression and plugin-internal I/O/scratch allocations.
    unsigned long long getNumBytesRead() const override;
    double getTotalReadTime() const override;
    
    /// Returns a hash based on the filename for deduplication
    unsigned long long getHash(hipStream_t stream = 0) const override;

  private:
    std::string filename_;
    TextureInfo info_;
    bool isOpen_ = false;
    
    mutable std::mutex mutex_;
    unsigned long long bytesRead_ = 0;
    double totalReadTime_ = 0.0;
    
    // Reserved, unused: retain the existing public class layout without a pixel cache.
    std::vector<std::vector<unsigned char>> mipLevels_;
    
    // Caller holds mutex_ and has validated the destination and original level.
    bool readMipLevelLocked(char* dest, unsigned int mipLevel);
    
};

}  // namespace hip_demand
