// SPDX-License-Identifier: MIT
// Tests for the VMM loader's cascade textures (Options::enableCascades): pages given to mip levels on demand.

#include "VmmLoaderTestUtils.h"

namespace hip_demand::vmm::test {

namespace {

class VmmCascadeTest : public VmmLoaderTest
{
  protected:
    static Options cascadeOptions()
    {
        Options options        = defaultOptions();
        options.enableCascades = true;
        return options;
    }

    // A 1024x1024 8-bit texture with 256x256 tiles: levels 0..2 are tiled (16, 4 and 1 pages), levels 3..10 the tail.
    const DemandTexture& createTexture( TestImage::Pattern pattern = TestImage::Pattern::Uniform )
    {
        return loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 11, pattern, 0xAA ), TextureDescriptor{} );
    }

    DeviceTextureInfo requestCascade( const DemandTexture& texture, uint32_t level )
    {
        requestResources( { cascadeResourceId( texture.getId(), level ) } );
        return readTextureInfo( texture.getId() );
    }
};

}  // namespace

TEST_F( VmmCascadeTest, ResourceTableHasCascadesOnlyWhenEnabled )
{
    createLoader( defaultOptions() );
    EXPECT_EQ( context_.resourceTable.cascades.count, 0u );
    EXPECT_EQ( context_.resourceTable.count(), options_.maxVirtualPages + options_.maxTextures );

    createLoader( cascadeOptions() );
    EXPECT_EQ( context_.resourceTable.cascades.start, context_.resourceTable.textureInfos.end() );
    EXPECT_EQ( context_.resourceTable.cascades.count, options_.maxTextures * MAX_TILED_MIP_LEVELS );
}

TEST_F( VmmCascadeTest, LoadedTextureStartsWithTheMipTail )
{
    createLoader( cascadeOptions() );
    const DemandTexture&    texture = createTexture();
    const DeviceTextureInfo info    = loadTextureInfo( texture );
    EXPECT_EQ( info.mipTailFirstLevel(), 3u );
    EXPECT_EQ( info.cascadeLevel, 3u );
    EXPECT_EQ( info.mipTailPage, 0u );

    // The next texture's pages follow right after the one page of the tail.
    const DemandTexture&    next     = createTexture();
    const DeviceTextureInfo nextInfo = loadTextureInfo( next );
    EXPECT_EQ( nextInfo.cascadeLevel, 3u );
    EXPECT_EQ( nextInfo.mipTailPage, 1u );

    requestResources( { pageResourceId( info.mipTailPage ) } );
    EXPECT_TRUE( isResident( info.mipTailPage ) );
    EXPECT_EQ( pageValue( info.mipTailPage, info.mipTailLevelOffset( info.mipCount() ) ), 0xAA );
}

TEST_F( VmmCascadeTest, TextureHeldByItsTailNeedsNoCascade )
{
    createLoader( cascadeOptions() );
    // 64x64 8-bit: every level fits the tail.
    const DemandTexture&    texture = loader_->createTexture( std::make_shared<TestImage>( 64, 64, 7, true, 0xAA ), TextureDescriptor{} );
    const DeviceTextureInfo info    = loadTextureInfo( texture );
    EXPECT_EQ( info.mipTailFirstLevel(), 0u );
    EXPECT_EQ( info.cascadeLevel, 0u );
    EXPECT_EQ( info.mipTailPage, 0u );
}

TEST_F( VmmCascadeTest, CascadeRequestExtendsTheTextureToItsLevel )
{
    createLoader( cascadeOptions() );
    const DemandTexture&    texture = createTexture();
    const DeviceTextureInfo initial = loadTextureInfo( texture );
    ASSERT_EQ( initial.cascadeLevel, 3u );

    // Level 1 brings levels 1 and 2, in that order after the tail page.
    DeviceTextureInfo info = requestCascade( texture, 1 );
    EXPECT_TRUE( isResourceResident( cascadeResourceId( texture.getId(), 1 ) ) );
    EXPECT_EQ( info.cascadeLevel, 1u );
    EXPECT_EQ( info.mipTailPage, 0u );
    EXPECT_EQ( info.getMipLevel( 1 ).startPage, 1u );
    EXPECT_EQ( info.getMipLevel( 1 ).pageCount(), 4u );
    EXPECT_EQ( info.getMipLevel( 2 ).startPage, 5u );
    EXPECT_EQ( info.getMipLevel( 2 ).pageCount(), 1u );

    // A coarser request afterwards changes nothing.
    info = requestCascade( texture, 2 );
    EXPECT_TRUE( isResourceResident( cascadeResourceId( texture.getId(), 2 ) ) );
    EXPECT_EQ( info.cascadeLevel, 1u );
    EXPECT_EQ( info.getMipLevel( 1 ).startPage, 1u );
    EXPECT_EQ( info.getMipLevel( 2 ).startPage, 5u );

    // The finest level comes last and gets the pages after the others.
    info = requestCascade( texture, 0 );
    EXPECT_EQ( info.cascadeLevel, 0u );
    EXPECT_EQ( info.getMipLevel( 0 ).startPage, 6u );
    EXPECT_EQ( info.getMipLevel( 0 ).pageCount(), 16u );
    EXPECT_EQ( info.getMipLevel( 1 ).startPage, 1u );
    EXPECT_EQ( info.getMipLevel( 2 ).startPage, 5u );

    // The next texture's tail follows the extended one.
    EXPECT_EQ( loadTextureInfo( createTexture() ).mipTailPage, 22u );
}

TEST_F( VmmCascadeTest, CascadeRequestsOfOnePassExtendTheTextureOnce )
{
    createLoader( cascadeOptions() );
    const DemandTexture& texture = createTexture();
    loadTextureInfo( texture );

    requestResources( { cascadeResourceId( texture.getId(), 0 ), cascadeResourceId( texture.getId(), 2 ) } );
    const DeviceTextureInfo info = readTextureInfo( texture.getId() );
    EXPECT_EQ( info.cascadeLevel, 0u );
    EXPECT_TRUE( isResourceResident( cascadeResourceId( texture.getId(), 0 ) ) );
    EXPECT_TRUE( isResourceResident( cascadeResourceId( texture.getId(), 2 ) ) );

    // Whichever request came first, the levels got disjoint page ranges after the tail, 21 pages in all.
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    for( uint32_t level = 0; level < 3; ++level )
    {
        const DeviceMipLevel mip = info.getMipLevel( level );
        ranges.emplace_back( mip.startPage, mip.startPage + mip.pageCount() );
    }
    std::sort( ranges.begin(), ranges.end() );
    EXPECT_EQ( ranges[0].first, 1u );
    EXPECT_EQ( ranges[0].second, ranges[1].first );
    EXPECT_EQ( ranges[1].second, ranges[2].first );
    EXPECT_EQ( ranges[2].second, 22u );
}

TEST_F( VmmCascadeTest, TilesOfExtendedLevelsLoad )
{
    createLoader( cascadeOptions() );
    const DemandTexture& texture = createTexture( TestImage::Pattern::Tiles );
    loadTextureInfo( texture );

    // A tile of level 1 after the texture reaches it: decoded as its own level and position.
    DeviceTextureInfo info   = requestCascade( texture, 1 );
    const uint32_t    page11 = tilePageId( info, 1, 1, 0 );
    requestResources( { pageResourceId( page11 ) } );
    EXPECT_TRUE( isResident( page11 ) );
    EXPECT_EQ( readPage( page11, 4 ), TestImage::tileMarker( 1, 1, 0 ) );

    // Level 0 is added after level 1 and 2, so its pages come after theirs; its tiles decode all the same.
    info                  = requestCascade( texture, 0 );
    const uint32_t page32 = tilePageId( info, 0, 3, 2 );
    EXPECT_GT( page32, page11 );
    requestResources( { pageResourceId( page32 ) } );
    EXPECT_TRUE( isResident( page32 ) );
    EXPECT_EQ( readPage( page32, 4 ), TestImage::tileMarker( 0, 3, 2 ) );
    EXPECT_TRUE( isResident( page11 ) );
}

TEST_F( VmmCascadeTest, WithoutCascadesEveryLevelHasPagesAtOnce )
{
    createLoader( defaultOptions() );
    const DemandTexture&    texture = createTexture();
    const DeviceTextureInfo info    = loadTextureInfo( texture );
    EXPECT_EQ( info.cascadeLevel, 0u );
    EXPECT_EQ( info.getMipLevel( 0 ).startPage, 0u );
    EXPECT_EQ( info.getMipLevel( 1 ).startPage, 16u );
    EXPECT_EQ( info.getMipLevel( 2 ).startPage, 20u );
    EXPECT_EQ( info.mipTailPage, 21u );
}

}  // namespace hip_demand::vmm::test
