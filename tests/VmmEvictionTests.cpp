// SPDX-License-Identifier: MIT
// Tests for the VMM loader's memory budget (Options::maxPhysicalPages): the eviction of tile pages that keeps to it,
// what happens when the tiles a launch reads do not fit, and tiles that cannot be read.

#include "VmmLoaderTestUtils.h"
#include <DemandLoading/vmm/Lru.h>

namespace hip_demand::vmm::test {

namespace {

constexpr size_t pageBytes = PageSystem::PAGE_BYTES;

// An image whose tiles cannot be read.
class UnreadableImage : public TestImage
{
  public:
    using TestImage::TestImage;
    bool readTile( char*, unsigned int, const Tile&, hipStream_t ) override { return false; }
};

class VmmEvictionTest : public VmmLoaderTest
{
  protected:
    static Options evictionOptions( uint32_t budgetPages, bool coalesceWhiteBlackTiles = false )
    {
        Options options                 = defaultOptions();
        options.maxPhysicalPages        = budgetPages;
        options.enableEviction          = true;
        options.maxEvictedPages         = 64;
        options.coalesceWhiteBlackTiles = coalesceWhiteBlackTiles;
        return options;
    }

    uint32_t residentCount( const std::vector<uint32_t>& pageIds )
    {
        return static_cast<uint32_t>( std::count_if( pageIds.begin(), pageIds.end(), [&]( uint32_t pageId ) { return isResident( pageId ); } ) );
    }

    std::vector<uint32_t> resourceIds( const std::vector<uint32_t>& pageIds ) const
    {
        std::vector<uint32_t> ids;
        for( uint32_t pageId : pageIds )
            ids.push_back( pageResourceId( pageId ) );
        return ids;
    }

    // A 1024x1024 texture of 16 tiles, all loaded.
    std::vector<uint32_t> loadSixteenTiles( uint8_t value = 0xAA )
    {
        image_                      = std::make_shared<TestImage>( 1024, 1024, 1, true, value );
        const DemandTexture& texture = loader_->createTexture( image_, TextureDescriptor{} );
        info_                       = loadTextureInfo( texture );
        std::vector<uint32_t> pages = loadLevel0( info_ );
        EXPECT_EQ( pages.size(), 16u );
        EXPECT_EQ( residentCount( pages ), 16u );
        return pages;
    }

    // A 256x256 texture, one page, with its metadata loaded. Returns its page.
    uint32_t createOnePageTexture( uint8_t value )
    {
        const DemandTexture&    texture = loader_->createTexture( std::make_shared<TestImage>( 256, 256, 1, true, value ), TextureDescriptor{} );
        const DeviceTextureInfo info    = loadTextureInfo( texture );
        return tilePageId( info, 0, 0, 0 );
    }

    std::shared_ptr<TestImage> image_;
    DeviceTextureInfo          info_{};
};

}  // namespace

TEST_F( VmmEvictionTest, NothingIsEvictedWhileTheBudgetHasRoom )
{
    createLoader( evictionOptions( 4096 ) );
    EXPECT_NE( context_.lru.ptr, nullptr );
    EXPECT_EQ( context_.evictionCandidates.len, 64u );
    EXPECT_TRUE( context_.requestIfResident );

    const std::vector<uint32_t> pages = loadSixteenTiles();
    for( uint32_t pageId : pages )
        EXPECT_EQ( lruValue( pageId ), 0u ) << "page " << pageId;

    for( uint32_t pass = 0; pass < 8; ++pass )
        runPass();
    for( uint32_t pageId : pages )
    {
        EXPECT_GE( lruValue( pageId ), 1u ) << "page " << pageId;
        EXPECT_TRUE( isResident( pageId ) ) << "page " << pageId;
    }
    const Statistics stats = loader_->getStatistics();
    EXPECT_EQ( stats.numEvictions, 0u );
    EXPECT_EQ( stats.numDeniedRequests, 0u );
    EXPECT_EQ( stats.residentTileBytes, 16 * pageBytes );
}

TEST_F( VmmEvictionTest, AnUnreadTileMakesRoomForARequest )
{
    createLoader( evictionOptions( 16 ) );
    const std::vector<uint32_t> pages     = loadSixteenTiles();
    const uint32_t              otherPage = createOnePageTexture( 0xBB );
    EXPECT_EQ( residentCount( pages ), 16u ) << "a pass with room evicts nothing";

    // The budget is full. The pass picks a victim among the tiles its launch did not read and puts the request off.
    requestResources( { pageResourceId( otherPage ) } );
    EXPECT_FALSE( isResident( otherPage ) );
    EXPECT_EQ( residentCount( pages ), 16u ) << "the victim stays resident until the next launch is prepared";
    EXPECT_EQ( loader_->getStatistics().numDeniedRequests, 1u );
    EXPECT_FALSE( loader_->getStatistics().budgetExhausted );

    // The next pass finds the victim gone and its page free.
    requestResources( { pageResourceId( otherPage ) } );
    EXPECT_TRUE( isResident( otherPage ) );
    EXPECT_EQ( pageValue( otherPage ), 0xBB );
    EXPECT_EQ( residentCount( pages ), 15u );

    const Statistics stats = loader_->getStatistics();
    EXPECT_EQ( stats.numEvictions, 1u );
    EXPECT_EQ( stats.numDeniedRequests, 1u );
    EXPECT_EQ( impl_->physicalPageCount(), 16u ) << "the evicted page was reused";
    EXPECT_EQ( stats.residentTileBytes, 16 * pageBytes );
    EXPECT_EQ( stats.peakResidentTileBytes, 16 * pageBytes );
    EXPECT_EQ( stats.bytesTransferredToDevice, 17 * pageBytes );
}

TEST_F( VmmEvictionTest, TheTileUnreadForLongestGoesFirst )
{
    createLoader( evictionOptions( 16 ) );
    const std::vector<uint32_t> pages     = loadSixteenTiles();
    const uint32_t              otherPage = createOnePageTexture( 0xBB );

    // The second half goes unread for some launches.
    const std::vector<uint32_t> recent( pages.begin(), pages.begin() + 8 );
    const std::vector<uint32_t> old( pages.begin() + 8, pages.end() );
    for( uint32_t pass = 0; pass < 5; ++pass )
        requestResources( resourceIds( recent ) );
    for( uint32_t pageId : recent )
        EXPECT_EQ( lruValue( pageId ), 0u ) << "page " << pageId;
    for( uint32_t pageId : old )
        EXPECT_GE( lruValue( pageId ), 2u ) << "page " << pageId;

    requestResources( { pageResourceId( otherPage ) } );
    requestResources( { pageResourceId( otherPage ) } );
    EXPECT_TRUE( isResident( otherPage ) );
    EXPECT_EQ( residentCount( recent ), 8u );
    EXPECT_EQ( residentCount( old ), 7u );
}

TEST_F( VmmEvictionTest, EvictedTilesAreReadAgainWhenRequested )
{
    createLoader( evictionOptions( 16 ) );
    const std::vector<uint32_t> pages     = loadSixteenTiles();
    const uint32_t              otherPage = createOnePageTexture( 0xBB );
    EXPECT_EQ( image_->getNumTilesRead(), 16u );

    requestResources( { pageResourceId( otherPage ) } );
    requestResources( { pageResourceId( otherPage ) } );
    const auto evicted = std::find_if( pages.begin(), pages.end(), [&]( uint32_t pageId ) { return !isResident( pageId ); } );
    ASSERT_NE( evicted, pages.end() );

    // The evicted tile comes back into its own page, for which another tile leaves.
    requestUntilResident( { pageResourceId( *evicted ) } );
    EXPECT_TRUE( isResident( *evicted ) );
    EXPECT_EQ( pageValue( *evicted ), 0xAA );
    EXPECT_EQ( image_->getNumTilesRead(), 17u );
    EXPECT_EQ( loader_->getStatistics().numEvictions, 2u );
    EXPECT_EQ( impl_->physicalPageCount(), 16u );
}

TEST_F( VmmEvictionTest, TilesTheLaunchReadAreNotEvictedAndTheBudgetIsExceeded )
{
    createLoader( evictionOptions( 16 ) );
    const std::vector<uint32_t> pages     = loadSixteenTiles();
    const uint32_t              otherPage = createOnePageTexture( 0xBB );

    // The launch reads all sixteen tiles and asks for one more: nothing can go, so the tile is mapped beyond the budget.
    std::vector<uint32_t> everything = resourceIds( pages );
    everything.push_back( pageResourceId( otherPage ) );
    requestResources( everything );
    EXPECT_TRUE( isResident( otherPage ) );
    EXPECT_EQ( residentCount( pages ), 16u );
    Statistics stats = loader_->getStatistics();
    EXPECT_EQ( stats.numDeniedRequests, 0u );
    EXPECT_EQ( stats.numEvictions, 0u );
    EXPECT_FALSE( stats.budgetExhausted );
    EXPECT_EQ( stats.residentTileBytes, 17 * pageBytes );
    EXPECT_EQ( stats.peakResidentTileBytes, 17 * pageBytes );

    // Once the launch reads fewer tiles, the excess goes without a request asking for room.
    requestResources( { pageResourceId( otherPage ) } );
    requestResources( { pageResourceId( otherPage ) } );
    EXPECT_TRUE( isResident( otherPage ) );
    EXPECT_EQ( residentCount( pages ), 15u );
}

TEST_F( VmmEvictionTest, AStrictLimitRefusesWhatDoesNotFit )
{
    Options options             = evictionOptions( 16 );
    options.strictPhysicalLimit = true;
    createLoader( options );
    const std::vector<uint32_t> pages     = loadSixteenTiles();
    const uint32_t              otherPage = createOnePageTexture( 0xBB );

    std::vector<uint32_t> everything = resourceIds( pages );
    everything.push_back( pageResourceId( otherPage ) );
    for( uint32_t pass = 0; pass < 3; ++pass )
        requestResources( everything );
    EXPECT_FALSE( isResident( otherPage ) );
    EXPECT_EQ( residentCount( pages ), 16u );
    Statistics stats = loader_->getStatistics();
    EXPECT_EQ( stats.numDeniedRequests, 3u );
    EXPECT_TRUE( stats.budgetExhausted );
    EXPECT_EQ( stats.peakResidentTileBytes, 16 * pageBytes );

    // A launch that reads less leaves tiles to evict.
    requestResources( { pageResourceId( otherPage ) } );
    EXPECT_FALSE( loader_->getStatistics().budgetExhausted );
    requestResources( { pageResourceId( otherPage ) } );
    EXPECT_TRUE( isResident( otherPage ) );
    EXPECT_EQ( residentCount( pages ), 15u );
    EXPECT_EQ( loader_->getStatistics().peakResidentTileBytes, 16 * pageBytes );
}

TEST_F( VmmEvictionTest, MipTailIsEvictedLikeATile )
{
    createLoader( evictionOptions( 1 ) );
    const DemandTexture&    texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 11, true, 0xAA ), TextureDescriptor{} );
    const DeviceTextureInfo info    = loadTextureInfo( texture );
    ASSERT_NE( info.mipTailPage, INVALID_PAGE );
    ASSERT_GT( info.mipTailFirstLevel(), 0u );
    requestResources( { pageResourceId( info.mipTailPage ) } );
    ASSERT_TRUE( isResident( info.mipTailPage ) );

    const uint32_t tile = tilePageId( info, 0, 0, 0 );
    requestResources( { pageResourceId( tile ) } );
    requestResources( { pageResourceId( tile ) } );
    EXPECT_TRUE( isResident( tile ) );
    EXPECT_FALSE( isResident( info.mipTailPage ) );
    // The texture's metadata is never evicted.
    EXPECT_TRUE( isResourceResident( textureInfoResourceId( texture.getId() ) ) );
    EXPECT_TRUE( readTextureInfo( texture.getId() ).isLoaded() );
}

TEST_F( VmmEvictionTest, SharedWhiteBlackPageSurvivesTheEvictionOfItsTiles )
{
    createLoader( evictionOptions( 2, true ) );
    const std::vector<uint32_t> white = loadSixteenTiles( 255 );
    ASSERT_EQ( impl_->physicalPageCount(), 1u );
    ASSERT_EQ( impl_->whiteBlackTileCount(), 1u );
    const uint32_t greyPage = createOnePageTexture( 128 );
    requestResources( { pageResourceId( greyPage ) } );
    ASSERT_TRUE( isResident( greyPage ) );
    ASSERT_EQ( impl_->physicalPageCount(), 2u );

    // A third page needs room. Evicting a white tile frees none, as the tiles share one retained page.
    const uint32_t otherPage = createOnePageTexture( 0x55 );
    requestUntilResident( { pageResourceId( otherPage ) }, 40 );
    EXPECT_TRUE( isResident( otherPage ) );
    EXPECT_EQ( pageValue( otherPage ), 0x55 );
    EXPECT_FALSE( isResident( greyPage ) );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 1u );

    // The white tiles join their page again.
    requestUntilResident( resourceIds( white ), 40 );
    for( uint32_t pageId : white )
    {
        EXPECT_TRUE( isResident( pageId ) ) << "page " << pageId;
        EXPECT_EQ( pageValue( pageId ), 255 ) << "page " << pageId;
    }
    EXPECT_EQ( impl_->whiteBlackTileCount(), 1u );
}

TEST_F( VmmEvictionTest, WithoutEvictionTheBudgetIsExceeded )
{
    Options options         = evictionOptions( 16 );
    options.maxEvictedPages = 0;
    createLoader( options );
    EXPECT_EQ( context_.lru.ptr, nullptr );
    EXPECT_EQ( context_.evictionCandidates.len, 0u );
    EXPECT_FALSE( context_.requestIfResident );

    const std::vector<uint32_t> pages     = loadSixteenTiles();
    const uint32_t              otherPage = createOnePageTexture( 0xBB );
    requestResources( { pageResourceId( otherPage ) } );
    EXPECT_TRUE( isResident( otherPage ) );
    for( uint32_t pass = 0; pass < 8; ++pass )
        runPass();
    EXPECT_EQ( residentCount( pages ), 16u );
    const Statistics stats = loader_->getStatistics();
    EXPECT_EQ( stats.numEvictions, 0u );
    EXPECT_EQ( stats.residentTileBytes, 17 * pageBytes );
}

TEST_F( VmmEvictionTest, ATileThatCannotBeReadIsResidentAndReadsAsZero )
{
    createLoader( defaultOptions() );
    const DemandTexture&        texture = loader_->createTexture( std::make_shared<UnreadableImage>( 1024, 1024, 1, true, 0xAA ), TextureDescriptor{} );
    const std::vector<uint32_t> pages   = loadLevel0( loadTextureInfo( texture ) );
    ASSERT_EQ( pages.size(), 16u );
    for( uint32_t pageId : pages )
    {
        EXPECT_TRUE( isResident( pageId ) ) << "page " << pageId;
        EXPECT_EQ( pageValue( pageId ), 0 ) << "page " << pageId;
    }
    EXPECT_EQ( loader_->getStatistics().numFailedReads, 16u );

    // Resident tiles are not requested again.
    EXPECT_EQ( requestResources( resourceIds( pages ) ).numTasksTotal(), 0 );
}

}  // namespace hip_demand::vmm::test