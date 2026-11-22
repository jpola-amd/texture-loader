// SPDX-License-Identifier: MIT
// Tests for the VMM loader's request processing: decoding resource ids into tiles and mip tails, and the passes.

#include "VmmLoaderTestUtils.h"

namespace hip_demand::vmm::test {

namespace {

constexpr size_t pageBytes = PageSystem::PAGE_BYTES;

class VmmRequestTest : public VmmLoaderTest
{
  protected:
    std::shared_ptr<TestImage> tilesImage( uint32_t width = 1024, uint32_t height = 1024, uint32_t numMipLevels = 11 )
    {
        return std::make_shared<TestImage>( width, height, numMipLevels, TestImage::Pattern::Tiles );
    }

    // Loads one tile and returns its first bytes.
    std::vector<uint8_t> loadTile( const DeviceTextureInfo& info, uint32_t level, uint32_t x, uint32_t y )
    {
        const uint32_t pageId = tilePageId( info, level, x, y );
        requestResources( { pageResourceId( pageId ) } );
        EXPECT_TRUE( isResident( pageId ) ) << "tile " << level << ", " << x << ", " << y;
        return readPage( pageId, 4 );
    }
};

}  // namespace

TEST_F( VmmRequestTest, TilesDecodeToTheirLevelAndPosition )
{
    createLoader( defaultOptions() );
    const auto              imageA = tilesImage();
    const auto              imageB = tilesImage();
    const DeviceTextureInfo a      = loadTextureInfo( loader_->createTexture( imageA, TextureDescriptor{} ) );
    const DeviceTextureInfo b      = loadTextureInfo( loader_->createTexture( imageB, TextureDescriptor{} ) );

    EXPECT_EQ( loadTile( b, 1, 1, 0 ), TestImage::tileMarker( 1, 1, 0 ) );
    EXPECT_EQ( loadTile( a, 0, 3, 2 ), TestImage::tileMarker( 0, 3, 2 ) );
    EXPECT_EQ( loadTile( b, 2, 0, 0 ), TestImage::tileMarker( 2, 0, 0 ) );
    EXPECT_EQ( loadTile( a, 1, 1, 1 ), TestImage::tileMarker( 1, 1, 1 ) );
    EXPECT_EQ( imageA->getNumTilesRead(), 2u );
    EXPECT_EQ( imageB->getNumTilesRead(), 2u );
}

TEST_F( VmmRequestTest, EdgeTilesAreReadWhole )
{
    createLoader( defaultOptions() );
    // 1000x1000 with 256x256 tiles: the last tile of a row holds 232 visible texels and a ramp past the edge.
    const DeviceTextureInfo info = loadTextureInfo( loader_->createTexture( tilesImage( 1000, 1000, 1 ), TextureDescriptor{} ) );
    EXPECT_EQ( loadTile( info, 0, 3, 3 ), TestImage::tileMarker( 0, 3, 3 ) );
    const std::vector<uint8_t> page = readPage( tilePageId( info, 0, 3, 3 ) );
    EXPECT_EQ( page[232], static_cast<uint8_t>( 232 ) );  // row 0, the first texel past the edge
    EXPECT_EQ( loader_->getStatistics().bytesTransferredToDevice, pageBytes );
}

TEST_F( VmmRequestTest, MipTailIsReadPackedIntoOnePage )
{
    createLoader( defaultOptions() );
    const auto              image = tilesImage();
    const DeviceTextureInfo info  = loadTextureInfo( loader_->createTexture( image, TextureDescriptor{} ) );
    ASSERT_EQ( info.mipTailFirstLevel(), 3u );
    ASSERT_EQ( info.mipCount(), 11u );

    requestResources( { pageResourceId( info.mipTailPage ) } );
    EXPECT_TRUE( isResident( info.mipTailPage ) );

    // One readMipTail() with the tail levels and their sizes.
    const std::vector<TestImage::MipTailRead> reads = image->mipTailReads();
    ASSERT_EQ( reads.size(), 1u );
    EXPECT_EQ( reads[0].firstLevel, 3u );
    EXPECT_EQ( reads[0].numLevels, 11u );
    ASSERT_EQ( reads[0].levelDims.size(), 11u );
    for( uint32_t level = 3; level < 11; ++level )
    {
        EXPECT_EQ( reads[0].levelDims[level].x, 1024u >> level ) << "level " << level;
        EXPECT_EQ( reads[0].levelDims[level].y, 1024u >> level ) << "level " << level;
    }
    EXPECT_EQ( image->getNumTilesRead(), 0u );

    // The page holds the levels one after another, each filled with its number, and zeros after the last.
    const std::vector<uint8_t> page = readPage( info.mipTailPage );
    for( uint32_t level = 3; level < 11; ++level )
    {
        const uint32_t begin = info.mipTailLevelOffset( level );
        const uint32_t end   = info.mipTailLevelOffset( level + 1 );
        EXPECT_EQ( end - begin, ( 1024u >> level ) * ( 1024u >> level ) ) << "level " << level;
        EXPECT_TRUE( std::all_of( page.begin() + begin, page.begin() + end, [=]( uint8_t byte ) { return byte == level; } ) ) << "level " << level;
    }
    const uint32_t tailEnd = info.mipTailLevelOffset( 11 );
    EXPECT_EQ( tailEnd, 128u * 128 + 64 * 64 + 32 * 32 + 16 * 16 + 8 * 8 + 4 * 4 + 2 * 2 + 1 );
    EXPECT_TRUE( std::all_of( page.begin() + tailEnd, page.end(), []( uint8_t byte ) { return byte == 0; } ) );
    EXPECT_EQ( loader_->getStatistics().bytesTransferredToDevice, pageBytes );
}

TEST_F( VmmRequestTest, RequestsOutsideAnyTextureAreDropped )
{
    createLoader( defaultOptions() );
    const auto           image   = tilesImage();
    const DemandTexture& texture = loader_->createTexture( image, TextureDescriptor{} );

    // A page before its texture's metadata is loaded belongs to no level yet.
    requestResources( { pageResourceId( 0 ) } );
    EXPECT_FALSE( isResident( 0 ) );
    EXPECT_EQ( image->getNumTilesRead(), 0u );

    // A texture that does not exist, and a page no texture reaches.
    requestResources( { textureInfoResourceId( 5 ), pageResourceId( options_.maxVirtualPages - 1 ) } );
    EXPECT_FALSE( isResourceResident( textureInfoResourceId( 5 ) ) );
    EXPECT_FALSE( isResident( options_.maxVirtualPages - 1 ) );

    // Once the metadata is there, the page is a tile.
    const DeviceTextureInfo info = loadTextureInfo( texture );
    EXPECT_EQ( info.getMipLevel( 0 ).startPage, 0u );
    requestResources( { pageResourceId( 0 ) } );
    EXPECT_TRUE( isResident( 0 ) );
    EXPECT_EQ( readPage( 0, 4 ), TestImage::tileMarker( 0, 0, 0 ) );

    // A page past the texture's levels is dropped too.
    requestResources( { pageResourceId( info.mipTailPage + 1 ) } );
    EXPECT_FALSE( isResident( info.mipTailPage + 1 ) );
}

TEST_F( VmmRequestTest, ResidentResourcesAreNotRequestedAgain )
{
    createLoader( defaultOptions() );
    const auto                  image = tilesImage();
    const DeviceTextureInfo     info  = loadTextureInfo( loader_->createTexture( image, TextureDescriptor{} ) );
    const std::vector<uint32_t> pages = loadLevel0( info );
    EXPECT_EQ( image->getNumTilesRead(), 16u );

    std::vector<uint32_t> resources{ textureInfoResourceId( 0 ) };
    for( uint32_t pageId : pages )
        resources.push_back( pageResourceId( pageId ) );
    const Ticket ticket = requestResources( resources );
    EXPECT_EQ( ticket.numTasksTotal(), 0 );
    EXPECT_EQ( image->getNumTilesRead(), 16u );
}

TEST_F( VmmRequestTest, PassProcessesAtMostMaxRequests )
{
    Options options     = defaultOptions();
    options.maxRequests = 4;
    createLoader( options );
    EXPECT_EQ( context_.requestedResources.len, 4u );

    const auto              image = tilesImage();
    const DeviceTextureInfo info  = loadTextureInfo( loader_->createTexture( image, TextureDescriptor{} ) );
    std::vector<uint32_t>   resources;
    for( uint32_t y = 0; y < 4; ++y )
        for( uint32_t x = 0; x < 4; ++x )
            resources.push_back( pageResourceId( tilePageId( info, 0, x, y ) ) );

    const Ticket first = requestResources( resources );
    EXPECT_EQ( first.numTasksTotal(), 4 );
    EXPECT_EQ( first.numTasksRemaining(), 0 );
    EXPECT_EQ( image->getNumTilesRead(), 4u );

    uint32_t passes = 1;
    while( requestResources( resources ).numTasksTotal() > 0 )
    {
        ++passes;
        ASSERT_LE( passes, 8u );
    }
    EXPECT_EQ( passes, 4u );
    EXPECT_EQ( image->getNumTilesRead(), 16u );
    for( uint32_t y = 0; y < 4; ++y )
        for( uint32_t x = 0; x < 4; ++x )
            EXPECT_EQ( readPage( tilePageId( info, 0, x, y ), 4 ), TestImage::tileMarker( 0, x, y ) );
}

TEST_F( VmmRequestTest, TexturesOfOneImageLoadTheirOwnPages )
{
    createLoader( defaultOptions() );
    const auto              image = tilesImage();
    const DeviceTextureInfo a     = loadTextureInfo( loader_->createTexture( image, TextureDescriptor{} ) );
    const DeviceTextureInfo b     = loadTextureInfo( loader_->createTexture( image, TextureDescriptor{} ) );
    EXPECT_NE( a.getMipLevel( 0 ).startPage, b.getMipLevel( 0 ).startPage );

    EXPECT_EQ( loadTile( a, 0, 1, 1 ), TestImage::tileMarker( 0, 1, 1 ) );
    EXPECT_EQ( loadTile( b, 0, 1, 1 ), TestImage::tileMarker( 0, 1, 1 ) );
    EXPECT_EQ( image->getNumTilesRead(), 2u );
    EXPECT_EQ( impl_->physicalPageCount(), 2u );
    EXPECT_FALSE( isResident( tilePageId( a, 0, 0, 0 ) ) );
}

}  // namespace hip_demand::vmm::test
