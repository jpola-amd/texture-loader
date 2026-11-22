// SPDX-License-Identifier: MIT
// Shared fixture for tests that drive the VMM loader with in-memory images.

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <DemandLoading/VmmDemandTextureLoader.h>
#include <DemandLoading/vmm/DemandTextureLoaderImpl.h>
#include <ImageSource/ImageSource.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace hip_demand::vmm::test {

// An in-memory 8-bit image. Its texels are one value in every channel (Uniform), or each tile is a byte ramp that
// starts with the marker tileMarker() of the tile (Tiles), or every texel is a hash of its level and position
// (Noise), which is reproduced by readMipLevel() so that a native texture can be built from the same content.
// The part of a tile past the edge of its level is always a byte ramp, as an image source may leave anything there.
class TestImage : public ImageSource
{
  public:
    enum class Pattern
    {
        Uniform,
        Tiles,
        Noise
    };

    // A call of readMipTail().
    struct MipTailRead
    {
        unsigned int       firstLevel = 0;
        unsigned int       numLevels  = 0;
        std::vector<uint2> levelDims;  // Indexed by level, as the loader passes it.
    };

    TestImage( uint32_t width, uint32_t height, uint32_t numMipLevels, bool uniform, uint8_t value, uint32_t numChannels = 1 )
        : TestImage( width, height, numMipLevels, uniform ? Pattern::Uniform : Pattern::Tiles, value, numChannels )
    {
    }

    TestImage( uint32_t width, uint32_t height, uint32_t numMipLevels, Pattern pattern, uint8_t value = 0, uint32_t numChannels = 1 )
        : pattern_( pattern )
        , value_( value )
    {
        info_.width        = width;
        info_.height       = height;
        info_.numChannels  = numChannels;
        info_.format       = HIP_AD_FORMAT_UNSIGNED_INT8;
        info_.numMipLevels = numMipLevels;
        info_.isTiled      = true;
        info_.isValid      = true;
    }

    // The info open() reports, for tests of the loader's validation.
    TextureInfo& mutableInfo() { return info_; }

    // The first bytes of a Tiles tile: its level and position, and a byte no ramp starts with.
    static std::vector<uint8_t> tileMarker( uint32_t mipLevel, uint32_t tileX, uint32_t tileY )
    {
        return { static_cast<uint8_t>( mipLevel ), static_cast<uint8_t>( tileX ), static_cast<uint8_t>( tileY ), 0xA5 };
    }

    // The value of a Noise texel's channel.
    static uint8_t noiseTexel( uint32_t mipLevel, uint32_t x, uint32_t y, uint32_t channel )
    {
        uint32_t hash = mipLevel * 0x9E3779B9u ^ x * 0x85EBCA6Bu ^ y * 0xC2B2AE35u ^ channel * 0x27D4EB2Fu;
        hash ^= hash >> 15;
        hash *= 0x2C1B3C6Du;
        hash ^= hash >> 12;
        return static_cast<uint8_t>( hash );
    }

    void open( TextureInfo* info ) override
    {
        open_ = true;
        if( info )
            *info = info_;
    }
    void               close() override { open_ = false; }
    bool               isOpen() const override { return open_; }
    const TextureInfo& getInfo() const override { return info_; }

    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, hipStream_t ) override
    {
        if( mipLevel >= info_.numMipLevels )
            return false;
        const size_t count = static_cast<size_t>( width ) * height * info_.numChannels;
        switch( pattern_ )
        {
            case Pattern::Uniform:
                std::memset( dest, value_, count );
                break;
            case Pattern::Tiles:
                std::memset( dest, static_cast<int>( mipLevel ), count );
                break;
            case Pattern::Noise:
                fillNoise( dest, mipLevel, 0, 0, width, height, width );
                break;
        }
        numBytesRead_ += count;
        return true;
    }

    bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, hipStream_t ) override
    {
        if( mipLevel >= info_.numMipLevels )
            return false;
        const size_t   rowBytes      = static_cast<size_t>( tile.width ) * info_.numChannels;
        const uint32_t levelWidth    = std::max( info_.width >> mipLevel, 1u );
        const uint32_t levelHeight   = std::max( info_.height >> mipLevel, 1u );
        const uint32_t visibleWidth  = std::min( tile.width, levelWidth - tile.x * tile.width );
        const uint32_t visibleHeight = std::min( tile.height, levelHeight - tile.y * tile.height );

        for( size_t i = 0; i < rowBytes * tile.height; ++i )
            dest[i] = pattern_ == Pattern::Uniform ? static_cast<char>( value_ ) : static_cast<char>( i );
        if( pattern_ == Pattern::Tiles )
        {
            const std::vector<uint8_t> marker = tileMarker( mipLevel, tile.x, tile.y );
            std::memcpy( dest, marker.data(), std::min( marker.size(), rowBytes * tile.height ) );
        }
        else if( pattern_ == Pattern::Noise )
            fillNoise( dest, mipLevel, tile.x * tile.width, tile.y * tile.height, visibleWidth, visibleHeight, tile.width );

        // The ramp past the edge.
        for( uint32_t row = 0; row < tile.height; ++row )
        {
            char*        line   = dest + row * rowBytes;
            const size_t offset = row < visibleHeight ? visibleWidth * info_.numChannels : 0;
            for( size_t i = offset; i < rowBytes; ++i )
                line[i] = static_cast<char>( row + i );
        }
        numTilesRead_++;
        numBytesRead_ += rowBytes * tile.height;
        return true;
    }

    bool readMipTail( char* dest, unsigned int mipTailFirstLevel, unsigned int numMipLevels, const uint2* mipLevelDims, hipStream_t stream ) override
    {
        {
            std::lock_guard<std::mutex> lock( mutex_ );
            mipTailReads_.push_back( { mipTailFirstLevel, numMipLevels, std::vector<uint2>( mipLevelDims, mipLevelDims + numMipLevels ) } );
        }
        return ImageSource::readMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims, stream );
    }

    std::vector<MipTailRead> mipTailReads() const
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        return mipTailReads_;
    }

    bool               readBaseColor( float4& ) override { return false; }
    unsigned long long getNumTilesRead() const override { return numTilesRead_; }
    unsigned long long getNumBytesRead() const override { return numBytesRead_; }
    double             getTotalReadTime() const override { return 0.0; }

  private:
    void fillNoise( char* dest, uint32_t mipLevel, uint32_t x0, uint32_t y0, uint32_t width, uint32_t height, uint32_t pitchTexels )
    {
        for( uint32_t y = 0; y < height; ++y )
        {
            for( uint32_t x = 0; x < width; ++x )
            {
                char* texel = dest + ( static_cast<size_t>( y ) * pitchTexels + x ) * info_.numChannels;
                for( uint32_t channel = 0; channel < info_.numChannels; ++channel )
                    texel[channel] = static_cast<char>( noiseTexel( mipLevel, x0 + x, y0 + y, channel ) );
            }
        }
    }

    TextureInfo                     info_{};
    Pattern                         pattern_ = Pattern::Uniform;
    uint8_t                         value_   = 0;
    bool                            open_    = false;
    std::atomic<unsigned long long> numTilesRead_{ 0 };
    std::atomic<unsigned long long> numBytesRead_{ 0 };
    mutable std::mutex              mutex_;
    std::vector<MipTailRead>        mipTailReads_;
};

class VmmLoaderTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        int deviceCount = 0;
        if( hipGetDeviceCount( &deviceCount ) != hipSuccess || deviceCount == 0 )
            GTEST_SKIP() << "No HIP device";
        ASSERT_EQ( hipSetDevice( 0 ), hipSuccess );
        int vmmSupported = 0;
        ASSERT_EQ( hipDeviceGetAttribute( &vmmSupported, hipDeviceAttributeVirtualMemoryManagementSupported, 0 ), hipSuccess );
        if( !vmmSupported )
            GTEST_SKIP() << "No virtual memory management";
        ASSERT_EQ( hipStreamCreate( &stream_ ), hipSuccess );
    }

    void TearDown() override
    {
        loader_.reset();
        if( stream_ )
            ASSERT_EQ( hipStreamDestroy( stream_ ), hipSuccess );
    }

    // Small limits, no eviction, no cascades.
    static Options defaultOptions()
    {
        Options options{};
        options.maxVirtualPages         = 4096;
        options.maxPhysicalPages        = 4096;
        options.maxTextures             = 16;
        options.maxRequests             = 256;
        options.enableEviction          = false;
        options.maxThreads              = 2;
        options.coalesceWhiteBlackTiles = false;
        return options;
    }

    void createLoader( bool coalesceWhiteBlackTiles )
    {
        Options options                 = defaultOptions();
        options.coalesceWhiteBlackTiles = coalesceWhiteBlackTiles;
        createLoader( options );
    }

    void createLoader( const Options& options )
    {
        options_ = options;
        loader_  = createDemandTextureLoader( options );
        impl_    = dynamic_cast<DemandTextureLoaderImpl*>( loader_.get() );
        ASSERT_NE( impl_, nullptr );
        // Fills the context with the resource table the request ids are computed from.
        loader_->launchPrepare( stream_, context_ );
        ASSERT_EQ( hipStreamSynchronize( stream_ ), hipSuccess );
    }

    uint32_t textureInfoResourceId( uint32_t textureId ) const
    {
        return context_.resourceTable.textureInfos.getResourceId( textureId );
    }
    uint32_t pageResourceId( uint32_t pageId ) const { return context_.resourceTable.textureTiles.getResourceId( pageId ); }
    uint32_t cascadeResourceId( uint32_t textureId, uint32_t level ) const
    {
        return context_.resourceTable.cascades.getResourceId( textureId * MAX_TILED_MIP_LEVELS + level );
    }
    uint32_t tilePageId( const DeviceTextureInfo& info, uint32_t level, uint32_t x, uint32_t y ) const
    {
        const DeviceMipLevel mip = info.getMipLevel( level );
        return mip.startPage + y * mip.tilesX + x;
    }

    // Runs one request pass for the given resources, as a sampling kernel would request them, and returns its ticket.
    Ticket requestResources( const std::vector<uint32_t>& resourceIds )
    {
        loader_->launchPrepare( stream_, context_ );
        EXPECT_EQ( hipStreamSynchronize( stream_ ), hipSuccess );
        std::vector<uint32_t> words( context_.referenceBits.len, 0 );
        for( uint32_t resourceId : resourceIds )
            words[resourceId / 32] |= 1u << ( resourceId % 32 );
        EXPECT_EQ( hipMemcpy( context_.referenceBits.ptr, words.data(), words.size() * sizeof( uint32_t ), hipMemcpyHostToDevice ), hipSuccess );
        Ticket ticket = loader_->processRequests( stream_, context_ );
        ticket.wait();
        EXPECT_EQ( hipStreamSynchronize( stream_ ), hipSuccess );
        return ticket;
    }

    // A pass that references nothing.
    void runPass() { requestResources( {} ); }

    // Runs passes that reference the given resources until none of them is requested any more, or maxPasses.
    void requestUntilResident( const std::vector<uint32_t>& resourceIds, uint32_t maxPasses = 16 )
    {
        for( uint32_t pass = 0; pass < maxPasses; ++pass )
        {
            if( requestResources( resourceIds ).numTasksTotal() == 0 )
                return;
        }
    }

    DeviceTextureInfo readTextureInfo( uint32_t textureId )
    {
        DeviceTextureInfo info{};
        EXPECT_EQ( hipMemcpy( &info, context_.textureInfos.ptr + textureId, sizeof( info ), hipMemcpyDeviceToHost ), hipSuccess );
        return info;
    }

    DeviceTextureInfo loadTextureInfo( const DemandTexture& texture )
    {
        requestResources( { textureInfoResourceId( texture.getId() ) } );
        const DeviceTextureInfo info = readTextureInfo( texture.getId() );
        EXPECT_TRUE( info.isLoaded() );
        return info;
    }

    // Loads every tile of the finest level and returns their page ids.
    std::vector<uint32_t> loadLevel0( const DeviceTextureInfo& info )
    {
        return loadLevel0Tiles( info, []( uint32_t, uint32_t ) { return true; } );
    }

    // Loads the tiles of the finest level that the predicate accepts by their (x, y) and returns their page ids.
    template <class Predicate>
    std::vector<uint32_t> loadLevel0Tiles( const DeviceTextureInfo& info, Predicate accept )
    {
        return loadLevelTiles( info, 0, accept );
    }

    // Loads every tile of a tiled level and returns their page ids.
    std::vector<uint32_t> loadLevel( const DeviceTextureInfo& info, uint32_t level )
    {
        return loadLevelTiles( info, level, []( uint32_t, uint32_t ) { return true; } );
    }

    template <class Predicate>
    std::vector<uint32_t> loadLevelTiles( const DeviceTextureInfo& info, uint32_t level, Predicate accept )
    {
        const DeviceMipLevel  mip = info.getMipLevel( level );
        std::vector<uint32_t> pageIds, resourceIds;
        for( uint32_t y = 0; y < mip.tilesY; ++y )
        {
            for( uint32_t x = 0; x < mip.tilesX; ++x )
            {
                if( !accept( x, y ) )
                    continue;
                const uint32_t pageId = mip.startPage + y * mip.tilesX + x;
                pageIds.push_back( pageId );
                resourceIds.push_back( pageResourceId( pageId ) );
            }
        }
        requestResources( resourceIds );
        return pageIds;
    }

    // Loads every tiled level and the mip tail.
    void loadWholeTexture( const DeviceTextureInfo& info )
    {
        for( uint32_t level = info.cascadeLevel; level < info.mipTailFirstLevel(); ++level )
            loadLevel( info, level );
        if( info.mipTailFirstLevel() < info.mipCount() )
            requestResources( { pageResourceId( info.mipTailPage ) } );
    }

    bool isResourceResident( uint32_t resourceId )
    {
        uint32_t word = 0;
        EXPECT_EQ( hipMemcpy( &word, context_.residenceBits.ptr + resourceId / 32, sizeof( word ), hipMemcpyDeviceToHost ), hipSuccess );
        return ( word >> ( resourceId % 32 ) ) & 1u;
    }

    bool isResident( uint32_t pageId ) { return isResourceResident( pageResourceId( pageId ) ); }

    // The LRU counter of a page, when eviction is enabled.
    uint32_t lruValue( uint32_t pageId )
    {
        uint32_t word = 0;
        EXPECT_NE( context_.lru.ptr, nullptr );
        EXPECT_EQ( hipMemcpy( &word, context_.lru.ptr + pageId / 8, sizeof( word ), hipMemcpyDeviceToHost ), hipSuccess );
        return ( word >> ( 4 * ( pageId % 8 ) ) ) & 0xF;
    }

    // The first bytes of a page (all of them by default).
    std::vector<uint8_t> readPage( uint32_t pageId, size_t bytes = 0 )
    {
        std::vector<uint8_t> host( bytes ? bytes : context_.pageSize );
        EXPECT_EQ( hipMemcpy( host.data(), context_.pageMemory.ptr + pageId * context_.pageSize, host.size(), hipMemcpyDeviceToHost ), hipSuccess );
        return host;
    }

    // The value the first bytes of a page hold (all of them by default), or -1 when they hold more than one.
    int pageValue( uint32_t pageId, size_t bytes = 0 )
    {
        std::vector<uint8_t> host( bytes ? bytes : context_.pageSize );
        if( hipMemcpy( host.data(), context_.pageMemory.ptr + pageId * context_.pageSize, host.size(), hipMemcpyDeviceToHost ) != hipSuccess )
            return -2;
        for( uint8_t byte : host )
        {
            if( byte != host[0] )
                return -1;
        }
        return host[0];
    }

    hipStream_t                          stream_ = nullptr;
    Options                              options_{};
    std::unique_ptr<DemandTextureLoader> loader_;
    DemandTextureLoaderImpl*             impl_ = nullptr;
    DeviceContext                        context_{};
};


}  // namespace hip_demand::vmm::test
