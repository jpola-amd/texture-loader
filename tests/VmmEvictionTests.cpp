// SPDX-License-Identifier: MIT
// Tests for the VMM loader's eviction of tile pages (Options::enableEviction).

#include "VmmLoaderTestUtils.h"
#include <DemandLoading/vmm/Lru.h>

namespace hip_demand::vmm::test {

namespace {

constexpr size_t pageBytes = PageSystem::PAGE_BYTES;

class VmmEvictionTest : public VmmLoaderTest
{
  protected:
    static Options evictionOptions( bool coalesceWhiteBlackTiles = false )
    {
        Options options                 = defaultOptions();
        options.enableEviction          = true;
        options.maxEvictedPages         = 64;
        options.coalesceWhiteBlackTiles = coalesceWhiteBlackTiles;
        return options;
    }

    bool allNonResident( const std::vector<uint32_t>& pageIds )
    {
        return std::none_of( pageIds.begin(), pageIds.end(), [&]( uint32_t pageId ) { return isResident( pageId ); } );
    }

    // Runs passes that reference only the given resources until the pages are non-resident on the device, or maxPasses.
    // The LRU counter of a tile unreferenced for a pass grows to the eviction threshold in a few passes, the next pass
    // clears its residence bit and the one after that hands the page to the page system.
    uint32_t agePages( const std::vector<uint32_t>& pageIds, const std::vector<uint32_t>& referencedResources = {}, uint32_t maxPasses = 12 )
    {
        for( uint32_t pass = 1; pass <= maxPasses; ++pass )
        {
            requestResources( referencedResources );
            if( allNonResident( pageIds ) )
                return pass;
        }
        return maxPasses + 1;
    }
};

}  // namespace

TEST_F( VmmEvictionTest, UnreferencedTilesAreEvicted )
{
    createLoader( evictionOptions() );
    EXPECT_NE( context_.lru.ptr, nullptr );
    EXPECT_EQ( context_.evictionCandidates.len, 64u );
    EXPECT_TRUE( context_.requestIfResident );

    const auto                  image   = std::make_shared<TestImage>( 1024, 1024, 1, true, 0xAA );
    const DemandTexture&        texture = loader_->createTexture( image, TextureDescriptor{} );
    const std::vector<uint32_t> pages   = loadLevel0( loadTextureInfo( texture ) );
    ASSERT_EQ( pages.size(), 16u );
    for( uint32_t pageId : pages )
        EXPECT_EQ( lruValue( pageId ), 0u ) << "page " << pageId;

    // One pass without references: every tile's LRU counter leaves zero, none is a candidate yet.
    runPass();
    for( uint32_t pageId : pages )
    {
        EXPECT_EQ( lruValue( pageId ), 1u ) << "page " << pageId;
        EXPECT_TRUE( isResident( pageId ) ) << "page " << pageId;
    }

    const uint32_t passes = agePages( pages );
    EXPECT_LE( passes, 8u );
    EXPECT_TRUE( allNonResident( pages ) );

    // The pages are unmapped when the page system next maps one, and were released, as no pass that handed them over
    // requested anything to reuse them for.
    EXPECT_EQ( loader_->getStatistics().numEvictions, 0u );
    const DemandTexture&        other      = loader_->createTexture( std::make_shared<TestImage>( 256, 256, 1, true, 0xBB ), TextureDescriptor{} );
    const std::vector<uint32_t> otherPages = loadLevel0( loadTextureInfo( other ) );
    ASSERT_EQ( otherPages.size(), 1u );
    const Statistics stats = loader_->getStatistics();
    EXPECT_EQ( stats.numEvictions, 16u );
    EXPECT_EQ( impl_->physicalPageCount(), 1u );
    EXPECT_EQ( pageValue( otherPages[0] ), 0xBB );
}

TEST_F( VmmEvictionTest, ReferencedTilesStayResident )
{
    createLoader( evictionOptions() );
    const DemandTexture&        texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 1, true, 0xAA ), TextureDescriptor{} );
    const std::vector<uint32_t> pages   = loadLevel0( loadTextureInfo( texture ) );
    ASSERT_EQ( pages.size(), 16u );

    const std::vector<uint32_t> kept{ pages[0], pages[5] };
    const std::vector<uint32_t> rest( pages.begin() + 1, pages.begin() + 5 );
    agePages( rest, { pageResourceId( kept[0] ), pageResourceId( kept[1] ) } );

    for( uint32_t pageId : kept )
    {
        EXPECT_TRUE( isResident( pageId ) ) << "page " << pageId;
        EXPECT_EQ( lruValue( pageId ), 0u ) << "page " << pageId;
    }
    for( uint32_t pageId : pages )
    {
        if( pageId != kept[0] && pageId != kept[1] )
            EXPECT_FALSE( isResident( pageId ) ) << "page " << pageId;
    }
}

TEST_F( VmmEvictionTest, EvictedTilesAreReadAgainWhenRequested )
{
    createLoader( evictionOptions() );
    const auto                  image   = std::make_shared<TestImage>( 1024, 1024, 1, true, 0xAA );
    const DemandTexture&        texture = loader_->createTexture( image, TextureDescriptor{} );
    const DeviceTextureInfo     info    = loadTextureInfo( texture );
    const std::vector<uint32_t> pages   = loadLevel0( info );
    EXPECT_EQ( image->getNumTilesRead(), 16u );

    // Two passes without references: the tiles whose turn it is reach the eviction threshold and become candidates.
    runPass();
    runPass();

    // The pass that clears their residence bits hands them to the page system with its own request count as the
    // number of pages to reuse: another texture's tiles, requested in that pass, take the evicted physical pages.
    const auto                  otherImage = std::make_shared<TestImage>( 1024, 1024, 1, true, 0xBB );
    const DemandTexture&        other      = loader_->createTexture( otherImage, TextureDescriptor{} );
    const DeviceTextureInfo     otherInfo  = loadTextureInfo( other );
    const std::vector<uint32_t> otherPages = loadLevel0( otherInfo );
    EXPECT_EQ( otherImage->getNumTilesRead(), 16u );
    const Statistics stats = loader_->getStatistics();
    EXPECT_GT( stats.numEvictions, 0u );
    EXPECT_EQ( impl_->physicalPageCount() + stats.numEvictions, 32u ) << "every evicted page was reused";

    agePages( pages );
    EXPECT_TRUE( allNonResident( pages ) );

    // The evicted tiles are read again and land in their own pages, not in the ones the other texture now holds.
    const std::vector<uint32_t> reloaded = loadLevel0( info );
    EXPECT_EQ( reloaded, pages );
    EXPECT_EQ( image->getNumTilesRead(), 32u );
    EXPECT_EQ( loader_->getStatistics().numEvictions, 16u );
    for( uint32_t pageId : pages )
    {
        EXPECT_TRUE( isResident( pageId ) ) << "page " << pageId;
        EXPECT_EQ( pageValue( pageId ), 0xAA ) << "page " << pageId;
    }
    for( uint32_t pageId : otherPages )
    {
        EXPECT_TRUE( isResident( pageId ) ) << "page " << pageId;
        EXPECT_EQ( pageValue( pageId ), 0xBB ) << "page " << pageId;
    }
}

TEST_F( VmmEvictionTest, MipTailIsEvictedLikeATile )
{
    createLoader( evictionOptions() );
    const DemandTexture&    texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 11, true, 0xAA ), TextureDescriptor{} );
    const DeviceTextureInfo info    = loadTextureInfo( texture );
    ASSERT_NE( info.mipTailPage, INVALID_PAGE );
    requestResources( { pageResourceId( info.mipTailPage ) } );
    ASSERT_TRUE( isResident( info.mipTailPage ) );

    agePages( { info.mipTailPage } );
    EXPECT_FALSE( isResident( info.mipTailPage ) );
    // The texture's metadata is never evicted.
    EXPECT_TRUE( isResourceResident( textureInfoResourceId( texture.getId() ) ) );
    EXPECT_TRUE( readTextureInfo( texture.getId() ).isLoaded() );
}

TEST_F( VmmEvictionTest, SharedWhiteBlackPageSurvivesTheEvictionOfItsTiles )
{
    createLoader( evictionOptions( true ) );
    const DemandTexture&        texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 1, true, 255 ), TextureDescriptor{} );
    const DeviceTextureInfo     info    = loadTextureInfo( texture );
    const std::vector<uint32_t> pages   = loadLevel0( info );
    ASSERT_EQ( impl_->physicalPageCount(), 1u );
    ASSERT_EQ( impl_->whiteBlackTileCount(), 1u );
    const size_t transferred = loader_->getStatistics().bytesTransferredToDevice;
    EXPECT_EQ( transferred, pageBytes );

    agePages( pages );
    EXPECT_TRUE( allNonResident( pages ) );

    // Mapping another texture's tile unmaps the aliases; the retained shared page stays allocated.
    const DemandTexture&        grey      = loader_->createTexture( std::make_shared<TestImage>( 256, 256, 1, true, 128 ), TextureDescriptor{} );
    const std::vector<uint32_t> greyPages = loadLevel0( loadTextureInfo( grey ) );
    EXPECT_EQ( loader_->getStatistics().numEvictions, 16u );
    EXPECT_EQ( impl_->physicalPageCount(), 2u );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 1u );

    // The white tiles join it again without another transfer.
    EXPECT_EQ( loadLevel0( info ), pages );
    EXPECT_EQ( impl_->physicalPageCount(), 2u );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 1u );
    EXPECT_EQ( loader_->getStatistics().bytesTransferredToDevice, transferred + pageBytes );  // the grey tile
    for( uint32_t pageId : pages )
    {
        EXPECT_TRUE( isResident( pageId ) ) << "page " << pageId;
        EXPECT_EQ( pageValue( pageId ), 255 ) << "page " << pageId;
    }
    EXPECT_EQ( pageValue( greyPages[0] ), 128 );
}

TEST_F( VmmEvictionTest, NoCandidatesMeansNoEviction )
{
    Options options         = evictionOptions();
    options.maxEvictedPages = 0;
    createLoader( options );
    EXPECT_EQ( context_.lru.ptr, nullptr );
    EXPECT_EQ( context_.evictionCandidates.len, 0u );
    EXPECT_FALSE( context_.requestIfResident );

    const DemandTexture&        texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 1, true, 0xAA ), TextureDescriptor{} );
    const std::vector<uint32_t> pages   = loadLevel0( loadTextureInfo( texture ) );
    for( uint32_t pass = 0; pass < 8; ++pass )
        runPass();
    for( uint32_t pageId : pages )
        EXPECT_TRUE( isResident( pageId ) ) << "page " << pageId;
    EXPECT_EQ( loader_->getStatistics().numEvictions, 0u );
}

TEST_F( VmmEvictionTest, EvictionsAreCountedInTheStatistics )
{
    createLoader( evictionOptions() );
    const DemandTexture&        texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 1, true, 0xAA ), TextureDescriptor{} );
    const std::vector<uint32_t> pages   = loadLevel0( loadTextureInfo( texture ) );
    const size_t                contextBytes = loader_->getStatistics().deviceMemoryUsed - 16 * pageBytes;
    EXPECT_EQ( loader_->getStatistics().deviceMemoryUsed,
               contextBytes + 16 * pageBytes );

    agePages( pages );
    loadLevel0( loadTextureInfo( loader_->createTexture( std::make_shared<TestImage>( 256, 256, 1, true, 0xBB ), TextureDescriptor{} ) ) );

    const Statistics stats = loader_->getStatistics();
    EXPECT_EQ( stats.numEvictions, 16u );
    EXPECT_EQ( stats.deviceMemoryUsed, contextBytes + pageBytes );
    EXPECT_EQ( stats.bytesTransferredToDevice, 17 * pageBytes );
}

}  // namespace hip_demand::vmm::test
