// SPDX-License-Identifier: MIT
// Tests for the VMM loader's preloading (Options::preloadTextures): the metadata and the mip tail of a texture are
// loaded by the first launchPrepare() after its creation, instead of by the requests of a launch.

#include "VmmLoaderTestUtils.h"

namespace hip_demand::vmm::test {

namespace {

constexpr size_t pageBytes = PageSystem::PAGE_BYTES;

class VmmPreloadTest : public VmmLoaderTest
{
  protected:
    static Options preloadOptions()
    {
        Options options         = defaultOptions();
        options.preloadTextures = true;
        return options;
    }

    // A budget of resident pages, kept by eviction.
    static Options preloadOptions( uint32_t budgetPages )
    {
        Options options          = preloadOptions();
        options.maxPhysicalPages = budgetPages;
        options.enableEviction   = true;
        options.maxEvictedPages  = 64;
        return options;
    }

    // 1024x1024 with 11 levels and one channel: levels 0 to 2 are tiled, levels 3 to 10 are the mip tail.
    static std::shared_ptr<TestImage> tilesImage()
    {
        return std::make_shared<TestImage>( 1024, 1024, 11, TestImage::Pattern::Tiles );
    }

    // What a renderer does before a launch.
    void prepareLaunch()
    {
        loader_->launchPrepare( stream_, context_ );
        ASSERT_EQ( hipStreamSynchronize( stream_ ), hipSuccess );
    }

    bool isTextureLoaded( uint32_t textureId )
    {
        return isResourceResident( textureInfoResourceId( textureId ) ) && readTextureInfo( textureId ).isLoaded();
    }
};

}  // namespace

TEST_F( VmmPreloadTest, MetadataAndMipTailAreResidentAfterLaunchPrepare )
{
    createLoader( preloadOptions() );
    const auto           image   = tilesImage();
    const DemandTexture& texture = loader_->createTexture( image, TextureDescriptor{} );
    EXPECT_FALSE( image->isOpen() ) << "nothing is loaded before the launch is prepared";

    prepareLaunch();
    EXPECT_TRUE( image->isOpen() );
    ASSERT_TRUE( isTextureLoaded( texture.getId() ) );
    const DeviceTextureInfo info = readTextureInfo( texture.getId() );
    ASSERT_EQ( info.mipTailFirstLevel(), 3u );
    ASSERT_TRUE( isResident( info.mipTailPage ) );

    // The page holds the tail levels one after another, each filled with its number.
    const std::vector<uint8_t> page = readPage( info.mipTailPage );
    for( uint32_t level = 3; level < 11; ++level )
    {
        const uint32_t begin = info.mipTailLevelOffset( level );
        const uint32_t end   = info.mipTailLevelOffset( level + 1 );
        EXPECT_TRUE( std::all_of( page.begin() + begin, page.begin() + end, [=]( uint8_t byte ) { return byte == level; } ) )
            << "level " << level;
    }
    EXPECT_EQ( image->mipTailReads().size(), 1u );
    EXPECT_EQ( image->getNumTilesRead(), 0u ) << "tiles are loaded on request";
    EXPECT_FALSE( isResident( tilePageId( info, 0, 0, 0 ) ) );

    const Statistics stats = loader_->getStatistics();
    EXPECT_EQ( stats.residentTileBytes, pageBytes );
    EXPECT_EQ( stats.peakResidentTileBytes, pageBytes );
    EXPECT_EQ( stats.bytesTransferredToDevice, pageBytes );

    // A launch that reads both finds nothing missing, and one that reads a tile gets it in its own pass.
    EXPECT_EQ( requestResources( { textureInfoResourceId( texture.getId() ), pageResourceId( info.mipTailPage ) } ).numTasksTotal(), 0 );
    EXPECT_EQ( requestResources( { pageResourceId( tilePageId( info, 0, 1, 2 ) ) } ).numTasksTotal(), 1 );
    EXPECT_EQ( readPage( tilePageId( info, 0, 1, 2 ), 4 ), TestImage::tileMarker( 0, 1, 2 ) );
    EXPECT_EQ( image->mipTailReads().size(), 1u );
}

TEST_F( VmmPreloadTest, TexturesCreatedLaterAreLoadedByTheNextLaunchPrepare )
{
    createLoader( preloadOptions() );
    const auto           firstImage = tilesImage();
    const DemandTexture& first      = loader_->createTexture( firstImage, TextureDescriptor{} );
    prepareLaunch();
    ASSERT_TRUE( isTextureLoaded( first.getId() ) );

    const auto           secondImage = tilesImage();
    const DemandTexture& second      = loader_->createTexture( secondImage, TextureDescriptor{} );
    EXPECT_FALSE( isTextureLoaded( second.getId() ) );
    runPass();
    ASSERT_TRUE( isTextureLoaded( second.getId() ) );
    EXPECT_TRUE( isResident( readTextureInfo( second.getId() ).mipTailPage ) );

    // The first texture was loaded once, and its pages are where they were.
    EXPECT_EQ( firstImage->mipTailReads().size(), 1u );
    EXPECT_EQ( secondImage->mipTailReads().size(), 1u );
    EXPECT_EQ( readTextureInfo( first.getId() ).mipTailPage, 21u );
    EXPECT_EQ( readTextureInfo( second.getId() ).mipTailPage, 43u );
    EXPECT_EQ( loader_->getStatistics().residentTileBytes, 2 * pageBytes );
}

TEST_F( VmmPreloadTest, MoreResourcesThanOnePassTakesArePublished )
{
    Options options     = preloadOptions();
    options.maxRequests = 4;
    createLoader( options );
    ASSERT_EQ( context_.processedResources.len, 4u );

    // Six textures give twelve resources.
    for( uint32_t i = 0; i < 6; ++i )
        loader_->createTexture( tilesImage(), TextureDescriptor{} );
    prepareLaunch();
    for( uint32_t textureId = 0; textureId < 6; ++textureId )
    {
        ASSERT_TRUE( isTextureLoaded( textureId ) ) << "texture " << textureId;
        EXPECT_TRUE( isResident( readTextureInfo( textureId ).mipTailPage ) ) << "texture " << textureId;
    }
}

TEST_F( VmmPreloadTest, UdimCellsAreLoaded )
{
    createLoader( preloadOptions() );
    // A 2x1 grid whose second cell is empty.
    const DemandTexture& entry =
        loader_->createUdimTexture( { tilesImage(), nullptr }, std::vector<TextureDescriptor>( 2 ), 2, 1 );
    prepareLaunch();

    ASSERT_TRUE( isTextureLoaded( entry.getId() ) );
    const DeviceTextureInfo cell = readTextureInfo( entry.getId() );
    EXPECT_TRUE( cell.isUdim() );
    EXPECT_FALSE( cell.isEmpty() );
    EXPECT_TRUE( isResident( cell.mipTailPage ) );

    ASSERT_TRUE( isTextureLoaded( entry.getId() + 1 ) );
    const DeviceTextureInfo empty = readTextureInfo( entry.getId() + 1 );
    EXPECT_TRUE( empty.isUdim() );
    EXPECT_TRUE( empty.isEmpty() );
    EXPECT_EQ( loader_->getStatistics().residentTileBytes, pageBytes ) << "an empty cell has no mip tail";
}

TEST_F( VmmPreloadTest, AnImageThatDoesNotLoadIsLeftToTheRequests )
{
    createLoader( preloadOptions() );
    const auto invalid             = tilesImage();
    invalid->mutableInfo().isValid = false;
    const DemandTexture& broken    = loader_->createTexture( invalid, TextureDescriptor{} );
    const DemandTexture& valid     = loader_->createTexture( tilesImage(), TextureDescriptor{} );
    prepareLaunch();

    EXPECT_FALSE( isTextureLoaded( broken.getId() ) );
    ASSERT_TRUE( isTextureLoaded( valid.getId() ) );
    const DeviceTextureInfo info = readTextureInfo( valid.getId() );
    EXPECT_EQ( info.getMipLevel( 0 ).startPage, 0u ) << "the texture that failed got no pages";
    EXPECT_TRUE( isResident( info.mipTailPage ) );

    // A launch that samples the texture requests its metadata, as without preloading.
    EXPECT_EQ( requestResources( { textureInfoResourceId( broken.getId() ) } ).numTasksTotal(), 1 );
    EXPECT_FALSE( isTextureLoaded( broken.getId() ) );
}

TEST_F( VmmPreloadTest, MipTailsBeyondTheBudgetAreLoadedOnRequest )
{
    createLoader( preloadOptions( 2 ) );
    std::vector<uint32_t> tails;
    for( uint32_t i = 0; i < 3; ++i )
        loader_->createTexture( tilesImage(), TextureDescriptor{} );
    prepareLaunch();
    for( uint32_t textureId = 0; textureId < 3; ++textureId )
    {
        ASSERT_TRUE( isTextureLoaded( textureId ) ) << "texture " << textureId;
        tails.push_back( readTextureInfo( textureId ).mipTailPage );
    }

    // The budget has room for two mip tails.
    EXPECT_TRUE( isResident( tails[0] ) );
    EXPECT_TRUE( isResident( tails[1] ) );
    EXPECT_FALSE( isResident( tails[2] ) );
    Statistics stats = loader_->getStatistics();
    EXPECT_EQ( stats.peakResidentTileBytes, 2 * pageBytes );
    EXPECT_EQ( stats.numDeniedRequests, 0u );

    // The third comes with the launch that reads it, for which a preloaded one that the launch does not read leaves.
    requestUntilResident( { pageResourceId( tails[2] ) } );
    EXPECT_TRUE( isResident( tails[2] ) );
    EXPECT_NE( isResident( tails[0] ), isResident( tails[1] ) );
    stats = loader_->getStatistics();
    EXPECT_EQ( stats.numEvictions, 1u );
    EXPECT_EQ( stats.residentTileBytes, 2 * pageBytes );
    EXPECT_EQ( stats.peakResidentTileBytes, 2 * pageBytes );

    // A texture created now finds the budget full: its metadata is loaded, its mip tail is not.
    const DemandTexture& fourth = loader_->createTexture( tilesImage(), TextureDescriptor{} );
    runPass();
    ASSERT_TRUE( isTextureLoaded( fourth.getId() ) );
    EXPECT_FALSE( isResident( readTextureInfo( fourth.getId() ).mipTailPage ) );
    EXPECT_EQ( loader_->getStatistics().peakResidentTileBytes, 2 * pageBytes );
}

}  // namespace hip_demand::vmm::test
