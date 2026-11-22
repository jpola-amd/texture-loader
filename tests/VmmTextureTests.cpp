// SPDX-License-Identifier: MIT
// Tests for the VMM loader's texture creation, image validation and the page layout of a loaded texture.

#include "VmmLoaderTestUtils.h"
#include <stdexcept>

namespace hip_demand::vmm::test {

namespace {

class VmmTextureTest : public VmmLoaderTest
{
};

std::shared_ptr<TestImage> image1024( uint32_t numChannels = 1, uint32_t numMipLevels = 11 )
{
    return std::make_shared<TestImage>( 1024, 1024, numMipLevels, true, 0xAA, numChannels );
}

}  // namespace

// --- creation ---

TEST_F( VmmTextureTest, TexturesGetSequentialIds )
{
    createLoader( defaultOptions() );
    for( uint32_t i = 0; i < options_.maxTextures; ++i )
        EXPECT_EQ( loader_->createTexture( image1024(), TextureDescriptor{} ).getId(), i );
    EXPECT_EQ( loader_->getStatistics().numTextures, options_.maxTextures );
}

TEST_F( VmmTextureTest, NullImageIsRejected )
{
    createLoader( defaultOptions() );
    EXPECT_THROW( loader_->createTexture( nullptr, TextureDescriptor{} ), std::invalid_argument );
}

TEST_F( VmmTextureTest, TextureCountIsLimited )
{
    createLoader( defaultOptions() );
    for( uint32_t i = 0; i < options_.maxTextures; ++i )
        loader_->createTexture( image1024(), TextureDescriptor{} );
    EXPECT_THROW( loader_->createTexture( image1024(), TextureDescriptor{} ), std::runtime_error );
    // Nor does a UDIM grid fit any more.
    EXPECT_THROW( loader_->createUdimTexture( { image1024() }, { TextureDescriptor{} }, 1, 1 ), std::runtime_error );
}

TEST_F( VmmTextureTest, UdimGridIsValidated )
{
    createLoader( defaultOptions() );
    const std::vector<std::shared_ptr<ImageSource>> images( 4, image1024() );
    const std::vector<TextureDescriptor>            descriptors( 4 );

    EXPECT_THROW( loader_->createUdimTexture( images, descriptors, 0, 4 ), std::invalid_argument );
    EXPECT_THROW( loader_->createUdimTexture( images, descriptors, 4, 0 ), std::invalid_argument );
    EXPECT_THROW( loader_->createUdimTexture( images, descriptors, 0x10000, 1 ), std::invalid_argument );
    EXPECT_THROW( loader_->createUdimTexture( images, descriptors, 1, 0x10000 ), std::invalid_argument );
    EXPECT_THROW( loader_->createUdimTexture( images, descriptors, 2, 2, 0 ), std::invalid_argument );
    EXPECT_THROW( loader_->createUdimTexture( images, descriptors, 2, 2, 256 ), std::invalid_argument );
    // One image and one descriptor per cell and channel.
    EXPECT_THROW( loader_->createUdimTexture( images, descriptors, 2, 1 ), std::invalid_argument );
    EXPECT_THROW( loader_->createUdimTexture( images, descriptors, 2, 2, 2 ), std::invalid_argument );
    EXPECT_THROW( loader_->createUdimTexture( images, std::vector<TextureDescriptor>( 3 ), 2, 2 ), std::invalid_argument );
    // 3x3 cells of 2 channels exceed the 16 textures.
    EXPECT_THROW( loader_->createUdimTexture( std::vector<std::shared_ptr<ImageSource>>( 18, image1024() ),
                                              std::vector<TextureDescriptor>( 18 ), 3, 3, 2 ),
                  std::runtime_error );

    // Subtextures must use normalized coordinates; an empty cell's descriptor is not looked at.
    std::vector<TextureDescriptor> unnormalized( 4 );
    unnormalized[3].normalizedCoords = false;
    EXPECT_THROW( loader_->createUdimTexture( images, unnormalized, 2, 2 ), std::invalid_argument );
    std::vector<std::shared_ptr<ImageSource>> withEmptyCell = images;
    withEmptyCell[3]                                        = nullptr;
    EXPECT_NO_THROW( loader_->createUdimTexture( withEmptyCell, unnormalized, 2, 2 ) );

    // Nothing was created by the rejected calls.
    EXPECT_EQ( loader_->createTexture( image1024(), TextureDescriptor{} ).getId(), 4u );
}

TEST_F( VmmTextureTest, UdimCellsCarryTheGrid )
{
    createLoader( defaultOptions() );
    loader_->createTexture( image1024(), TextureDescriptor{} );  // Texture 0, so the grid does not start at 0.

    // A 2x1 grid of 2 channels: cell 1's second channel is empty.
    const std::vector<std::shared_ptr<ImageSource>> images{ image1024(), image1024( 4 ), image1024(), nullptr };
    const DemandTexture& entry = loader_->createUdimTexture( images, std::vector<TextureDescriptor>( 4 ), 2, 1, 2 );
    EXPECT_EQ( entry.getId(), 1u );
    EXPECT_EQ( loader_->getStatistics().numTextures, 4u );

    for( uint32_t id = 1; id <= 4; ++id )
    {
        requestResources( { textureInfoResourceId( id ) } );
        const DeviceTextureInfo info = readTextureInfo( id );
        EXPECT_TRUE( isResourceResident( textureInfoResourceId( id ) ) ) << "texture " << id;
        EXPECT_TRUE( info.isLoaded() ) << "texture " << id;
        EXPECT_TRUE( info.isUdim() ) << "texture " << id;
        EXPECT_EQ( info.udimStartTexture, 1u ) << "texture " << id;
        EXPECT_EQ( info.udim(), 2u ) << "texture " << id;
        EXPECT_EQ( info.vdim(), 1u ) << "texture " << id;
        EXPECT_EQ( info.udimChannels(), 2u ) << "texture " << id;
        EXPECT_EQ( info.isEmpty(), id == 4 ) << "texture " << id;
    }
    const DeviceTextureInfo empty = readTextureInfo( 4 );
    EXPECT_EQ( empty.mipCount(), 0u );
    EXPECT_EQ( empty.mipTailPage, INVALID_PAGE );
    EXPECT_EQ( empty.width, 0u );
    EXPECT_EQ( readTextureInfo( 2 ).numChannels(), 4u );

    // A plain texture is not part of a grid.
    requestResources( { textureInfoResourceId( 0 ) } );
    EXPECT_FALSE( readTextureInfo( 0 ).isUdim() );
}

TEST_F( VmmTextureTest, AllEmptyUdimGridIsValid )
{
    createLoader( defaultOptions() );
    const DemandTexture& entry = loader_->createUdimTexture( { nullptr, nullptr }, std::vector<TextureDescriptor>( 2 ), 2, 1 );
    EXPECT_EQ( loader_->getStatistics().numTextures, 0u );
    requestResources( { textureInfoResourceId( entry.getId() ) } );
    const DeviceTextureInfo info = readTextureInfo( entry.getId() );
    EXPECT_TRUE( info.isLoaded() );
    EXPECT_TRUE( info.isEmpty() );
    EXPECT_TRUE( info.isUdim() );
    EXPECT_EQ( info.udim(), 2u );
}

// --- validation ---

TEST_F( VmmTextureTest, InvalidImagesDoNotLoad )
{
    createLoader( defaultOptions() );
    const auto broken = []( auto&& change ) {
        auto image = image1024();
        change( image->mutableInfo() );
        return image;
    };
    const std::vector<std::shared_ptr<TestImage>> images{
        broken( []( TextureInfo& info ) { info.isValid = false; } ),
        broken( []( TextureInfo& info ) { info.width = 0; } ),
        broken( []( TextureInfo& info ) { info.height = 0; } ),
        broken( []( TextureInfo& info ) { info.numMipLevels = 0; } ),
        broken( []( TextureInfo& info ) { info.numChannels = 0; } ),
        broken( []( TextureInfo& info ) { info.numChannels = 5; } ),
        broken( []( TextureInfo& info ) { info.format = static_cast<hipArray_Format>( 0xFF ); } ),
    };
    for( const auto& image : images )
    {
        const DemandTexture& texture = loader_->createTexture( image, TextureDescriptor{} );
        requestResources( { textureInfoResourceId( texture.getId() ) } );
        EXPECT_FALSE( isResourceResident( textureInfoResourceId( texture.getId() ) ) ) << "texture " << texture.getId();
        EXPECT_FALSE( readTextureInfo( texture.getId() ).isLoaded() ) << "texture " << texture.getId();
    }

    // The loader goes on serving valid textures, from the first page: the failed ones got none.
    const DemandTexture&    texture = loader_->createTexture( image1024(), TextureDescriptor{} );
    const DeviceTextureInfo info    = loadTextureInfo( texture );
    EXPECT_EQ( info.getMipLevel( 0 ).startPage, 0u );
    EXPECT_EQ( loadLevel0( info ).size(), 16u );
}

TEST_F( VmmTextureTest, TextureBeyondTheVirtualPagesDoesNotLoad )
{
    Options options         = defaultOptions();
    options.maxVirtualPages = 8;
    createLoader( options );

    // 16 pages do not fit; after the failure, a texture of 4 pages and one of 1 do, and the next 4 do not.
    const DemandTexture& large = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 1, true, 0xAA ), TextureDescriptor{} );
    requestResources( { textureInfoResourceId( large.getId() ) } );
    EXPECT_FALSE( isResourceResident( textureInfoResourceId( large.getId() ) ) );
    EXPECT_FALSE( readTextureInfo( large.getId() ).isLoaded() );

    const DemandTexture&    medium     = loader_->createTexture( std::make_shared<TestImage>( 512, 512, 1, true, 0xAA ), TextureDescriptor{} );
    const DeviceTextureInfo mediumInfo = loadTextureInfo( medium );
    EXPECT_EQ( mediumInfo.getMipLevel( 0 ).startPage, 0u );
    EXPECT_EQ( loadLevel0( mediumInfo ).size(), 4u );

    const DemandTexture&    small     = loader_->createTexture( std::make_shared<TestImage>( 256, 256, 1, true, 0xAA ), TextureDescriptor{} );
    const DeviceTextureInfo smallInfo = loadTextureInfo( small );
    EXPECT_EQ( smallInfo.mipTailPage, 4u );

    const DemandTexture& last = loader_->createTexture( std::make_shared<TestImage>( 512, 512, 1, true, 0xAA ), TextureDescriptor{} );
    requestResources( { textureInfoResourceId( last.getId() ) } );
    EXPECT_FALSE( readTextureInfo( last.getId() ).isLoaded() );
}

// --- layout ---

TEST_F( VmmTextureTest, DescriptorIsEncodedInTheTextureInfo )
{
    createLoader( defaultOptions() );
    TextureDescriptor descriptor{};
    descriptor.addressMode[0]   = hipAddressModeClamp;
    descriptor.addressMode[1]   = hipAddressModeMirror;
    descriptor.filterMode       = hipFilterModePoint;
    descriptor.mipmapFilterMode = hipFilterModePoint;
    descriptor.normalizedCoords = false;
    descriptor.maxAnisotropy    = 8;
    const DeviceTextureInfo info = loadTextureInfo( loader_->createTexture( image1024(), descriptor ) );
    EXPECT_EQ( info.addressMode( 0 ), static_cast<uint32_t>( hipAddressModeClamp ) );
    EXPECT_EQ( info.addressMode( 1 ), static_cast<uint32_t>( hipAddressModeMirror ) );
    EXPECT_EQ( info.filterMode(), static_cast<uint32_t>( hipFilterModePoint ) );
    EXPECT_EQ( info.mipmapFilterMode(), static_cast<uint32_t>( hipFilterModePoint ) );
    EXPECT_FALSE( info.normalizedCoords() );
    EXPECT_EQ( info.maxAnisotropy(), 8u );
    EXPECT_FALSE( info.wideOffsets() );
    EXPECT_FALSE( info.isEmpty() );
    EXPECT_EQ( info.width, 1024u );
    EXPECT_EQ( info.height, 1024u );

    const DeviceTextureInfo defaults = loadTextureInfo( loader_->createTexture( image1024(), TextureDescriptor{} ) );
    EXPECT_EQ( defaults.addressMode( 0 ), static_cast<uint32_t>( hipAddressModeWrap ) );
    EXPECT_EQ( defaults.addressMode( 1 ), static_cast<uint32_t>( hipAddressModeWrap ) );
    EXPECT_EQ( defaults.filterMode(), static_cast<uint32_t>( hipFilterModeLinear ) );
    EXPECT_EQ( defaults.mipmapFilterMode(), static_cast<uint32_t>( hipFilterModeLinear ) );
    EXPECT_TRUE( defaults.normalizedCoords() );
    EXPECT_EQ( defaults.maxAnisotropy(), 1u );
}

TEST_F( VmmTextureTest, AnisotropyIsClampedTo1To16 )
{
    createLoader( defaultOptions() );
    TextureDescriptor descriptor{};
    descriptor.maxAnisotropy = 0;
    EXPECT_EQ( loadTextureInfo( loader_->createTexture( image1024(), descriptor ) ).maxAnisotropy(), 1u );
    descriptor.maxAnisotropy = 32;
    EXPECT_EQ( loadTextureInfo( loader_->createTexture( image1024(), descriptor ) ).maxAnisotropy(), 16u );
    descriptor.maxAnisotropy = 16;
    EXPECT_EQ( loadTextureInfo( loader_->createTexture( image1024(), descriptor ) ).maxAnisotropy(), 16u );
}

TEST_F( VmmTextureTest, TileShapeFollowsTheTexelSize )
{
    createLoader( defaultOptions() );
    struct Case
    {
        uint32_t channels, tileWidth, tileHeight, mipTailFirstLevel;
    };
    // A tile fills a 64 KiB page, as square as a power of two allows; an rgb8 tile fills 48 KiB of it.
    // The tail starts at the first level whose chain of coarser levels fits a page.
    const Case cases[] = { { 1, 256, 256, 3 }, { 2, 256, 128, 3 }, { 3, 128, 128, 3 }, { 4, 128, 128, 4 } };
    for( const Case& c : cases )
    {
        const DeviceTextureInfo info = loadTextureInfo( loader_->createTexture( image1024( c.channels ), TextureDescriptor{} ) );
        EXPECT_EQ( info.numChannels(), c.channels );
        EXPECT_EQ( info.bytesPerTexel(), c.channels );
        EXPECT_EQ( info.format(), HIP_AD_FORMAT_UNSIGNED_INT8 );
        EXPECT_EQ( info.tileWidth(), c.tileWidth ) << c.channels << " channels";
        EXPECT_EQ( info.tileHeight(), c.tileHeight ) << c.channels << " channels";
        EXPECT_EQ( info.mipCount(), 11u );
        EXPECT_EQ( info.mipTailFirstLevel(), c.mipTailFirstLevel ) << c.channels << " channels";
        EXPECT_LE( info.mipTailLevelOffset( info.mipCount() ), PageSystem::PAGE_BYTES );
    }
}

TEST_F( VmmTextureTest, TextureWithoutMipLevelsHasNoTail )
{
    createLoader( defaultOptions() );
    const DeviceTextureInfo info = loadTextureInfo( loader_->createTexture( image1024( 1, 1 ), TextureDescriptor{} ) );
    EXPECT_EQ( info.mipCount(), 1u );
    EXPECT_EQ( info.mipTailFirstLevel(), 1u );
    EXPECT_EQ( info.mipTailPage, INVALID_PAGE );
    EXPECT_EQ( info.getMipLevel( 0 ).pageCount(), 16u );

    // A level that fits a page on its own is a tail.
    const DeviceTextureInfo tiny = loadTextureInfo( loader_->createTexture( std::make_shared<TestImage>( 256, 256, 1, true, 0 ), TextureDescriptor{} ) );
    EXPECT_EQ( tiny.mipTailFirstLevel(), 0u );
    EXPECT_EQ( tiny.mipTailPage, 16u );
}

TEST_F( VmmTextureTest, LevelsOfTexturesArePackedInOrder )
{
    createLoader( defaultOptions() );
    // 1024x1024 with 256x256 tiles: 16, 4 and 1 pages, then the tail. 512x512: 4 and 1 pages, then the tail.
    const DeviceTextureInfo a = loadTextureInfo( loader_->createTexture( image1024(), TextureDescriptor{} ) );
    const DeviceTextureInfo b = loadTextureInfo( loader_->createTexture( std::make_shared<TestImage>( 512, 512, 10, true, 0 ), TextureDescriptor{} ) );

    EXPECT_EQ( a.cascadeLevel, 0u );
    EXPECT_EQ( a.getMipLevel( 0 ).startPage, 0u );
    EXPECT_EQ( a.getMipLevel( 0 ).tilesX, 4u );
    EXPECT_EQ( a.getMipLevel( 0 ).tilesY, 4u );
    EXPECT_EQ( a.getMipLevel( 1 ).startPage, 16u );
    EXPECT_EQ( a.getMipLevel( 2 ).startPage, 20u );
    EXPECT_EQ( a.mipTailPage, 21u );
    EXPECT_TRUE( a.getMipLevel( 3 ).mipTail );
    EXPECT_EQ( a.getMipLevel( 3 ).startPage, 21u );

    EXPECT_EQ( b.mipTailFirstLevel(), 2u );
    EXPECT_EQ( b.getMipLevel( 0 ).startPage, 22u );
    EXPECT_EQ( b.getMipLevel( 1 ).startPage, 26u );
    EXPECT_EQ( b.mipTailPage, 27u );
}

TEST_F( VmmTextureTest, EdgeTilesCountTowardsTheLevel )
{
    createLoader( defaultOptions() );
    // 1000x600 with 256x256 tiles: 4x3 tiles on level 0, 2x2 on level 1 (500x300); level 2 (250x150) and down fit the tail.
    const DeviceTextureInfo info = loadTextureInfo( loader_->createTexture( std::make_shared<TestImage>( 1000, 600, 10, true, 0 ), TextureDescriptor{} ) );
    EXPECT_EQ( info.getMipLevel( 0 ).tilesX, 4u );
    EXPECT_EQ( info.getMipLevel( 0 ).tilesY, 3u );
    EXPECT_EQ( info.getMipLevel( 0 ).pageCount(), 12u );
    EXPECT_EQ( info.getMipLevel( 1 ).width, 500u );
    EXPECT_EQ( info.getMipLevel( 1 ).height, 300u );
    EXPECT_EQ( info.getMipLevel( 1 ).pageCount(), 4u );
    EXPECT_EQ( info.getMipLevel( 1 ).startPage, 12u );
    EXPECT_EQ( info.mipTailFirstLevel(), 2u );
    EXPECT_EQ( info.mipTailPage, 16u );
}

TEST_F( VmmTextureTest, MoreTiledLevelsThanWordsAreClamped )
{
    // Virtual pages are only reserved: 2^19 pages are 32 GiB of address space.
    Options options         = defaultOptions();
    options.maxVirtualPages = 1u << 19;
    options.maxTextures     = 2;
    createLoader( options );

    // 131072x131072 8-bit: levels 0..9 would be tiled (level 9 is one 256x256 tile), so the texture keeps its first
    // nine levels and loses the tail. Its finest level spans 16 GiB, more than 32-bit offsets reach.
    const DemandTexture&    texture = loader_->createTexture( std::make_shared<TestImage>( 131072, 131072, 18, true, 0 ), TextureDescriptor{} );
    const DeviceTextureInfo info    = loadTextureInfo( texture );
    EXPECT_EQ( info.mipCount(), MAX_TILED_MIP_LEVELS );
    EXPECT_EQ( info.mipTailFirstLevel(), MAX_TILED_MIP_LEVELS );
    EXPECT_EQ( info.mipTailPage, INVALID_PAGE );
    EXPECT_TRUE( info.wideOffsets() );
    uint32_t page = 0;
    for( uint32_t level = 0; level < MAX_TILED_MIP_LEVELS; ++level )
    {
        const DeviceMipLevel mip = info.getMipLevel( level );
        EXPECT_FALSE( mip.mipTail ) << "level " << level;
        EXPECT_EQ( mip.startPage, page ) << "level " << level;
        EXPECT_EQ( mip.tilesX, 512u >> level ) << "level " << level;
        page += mip.pageCount();
    }
    EXPECT_EQ( page, 349524u );

    const DeviceTextureInfo small = loadTextureInfo( loader_->createTexture( image1024(), TextureDescriptor{} ) );
    EXPECT_EQ( small.getMipLevel( 0 ).startPage, page );
    EXPECT_FALSE( small.wideOffsets() );
}

}  // namespace hip_demand::vmm::test
