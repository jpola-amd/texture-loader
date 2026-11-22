// SPDX-License-Identifier: MIT
// Tests for the VMM loader's sharing of white and black tiles (Options::coalesceWhiteBlackTiles).

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <DemandLoading/VmmDemandTextureLoader.h>
#include <DemandLoading/vmm/DemandTextureLoaderImpl.h>
#include <DemandLoading/vmm/WhiteBlackTile.h>
#include "VmmLoaderTestUtils.h"
#include <ImageSource/ImageSource.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace hip_demand::vmm::test {

namespace {

constexpr size_t pageBytes = PageSystem::PAGE_BYTES;

template <class T>
std::vector<uint8_t> texelBytes( std::initializer_list<T> channels )
{
    std::vector<uint8_t> texel( channels.size() * sizeof( T ) );
    std::memcpy( texel.data(), channels.begin(), texel.size() );
    return texel;
}

std::vector<uint8_t> pageOf( const std::vector<uint8_t>& texel )
{
    std::vector<uint8_t> page;
    page.reserve( pageBytes );
    while( page.size() < pageBytes )
        page.insert( page.end(), texel.begin(), texel.end() );
    return page;
}

// The key of a page of the texel, taken as one row, or "" when it is not a white or black tile.
std::string keyOfPage( const std::vector<uint8_t>& page, hipArray_Format format, uint32_t numChannels )
{
    const uint32_t texels = static_cast<uint32_t>( page.size() / ( getBytesPerChannel( format ) * numChannels ) );
    std::string    key;
    return whiteBlackTileKey( page.data(), texels, texels, 1, format, numChannels, key ) ? key : "";
}

// Whether a page of the texel is a white or black tile, and that its key is the texel.
template <class T>
bool isWhiteBlack( hipArray_Format format, std::initializer_list<T> channels )
{
    const std::vector<uint8_t> texel = texelBytes( channels );
    const std::string          key   = keyOfPage( pageOf( texel ), format, static_cast<uint32_t>( channels.size() ) );
    if( key.empty() )
        return false;
    EXPECT_EQ( key, std::string( texel.begin(), texel.end() ) );
    return true;
}

}  // namespace

TEST( VmmWhiteBlackTileKeyTest, WhiteAndBlackTexelsOfEveryChannelCount )
{
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 0 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 0, 0 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255, 255 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 0, 0, 0 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255, 255, 255 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 0, 0, 0, 0 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 0, 0, 0, 255 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255, 255, 255, 0 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255, 255, 255, 255 } ) );
}

TEST( VmmWhiteBlackTileKeyTest, OneIsWhatTheSamplerNormalizesToOne )
{
    EXPECT_TRUE( isWhiteBlack<int8_t>( HIP_AD_FORMAT_SIGNED_INT8, { 127, 127, 127, 0 } ) );
    EXPECT_TRUE( isWhiteBlack<uint16_t>( HIP_AD_FORMAT_UNSIGNED_INT16, { 65535 } ) );
    EXPECT_TRUE( isWhiteBlack<int16_t>( HIP_AD_FORMAT_SIGNED_INT16, { 32767, 32767 } ) );
    EXPECT_TRUE( isWhiteBlack<uint32_t>( HIP_AD_FORMAT_UNSIGNED_INT32, { 0xFFFFFFFFu } ) );
    EXPECT_TRUE( isWhiteBlack<int32_t>( HIP_AD_FORMAT_SIGNED_INT32, { 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF } ) );
    EXPECT_TRUE( isWhiteBlack<uint16_t>( HIP_AD_FORMAT_HALF, { 0x3C00, 0x3C00, 0x3C00, 0x0000 } ) );  // 1.0h
    EXPECT_TRUE( isWhiteBlack<float>( HIP_AD_FORMAT_FLOAT, { 1.f, 1.f, 1.f, 1.f } ) );
    EXPECT_TRUE( isWhiteBlack<float>( HIP_AD_FORMAT_FLOAT, { 0.f } ) );

    EXPECT_FALSE( isWhiteBlack<int8_t>( HIP_AD_FORMAT_SIGNED_INT8, { -128 } ) );
    EXPECT_FALSE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_SIGNED_INT8, { 255 } ) );
    EXPECT_FALSE( isWhiteBlack<uint16_t>( HIP_AD_FORMAT_HALF, { 0x3800 } ) );  // 0.5h
    EXPECT_FALSE( isWhiteBlack<float>( HIP_AD_FORMAT_FLOAT, { -1.f } ) );
}

TEST( VmmWhiteBlackTileKeyTest, OtherTexelsAreNotWhiteOrBlack )
{
    EXPECT_FALSE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 128 } ) );
    EXPECT_FALSE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255, 0 } ) );
    EXPECT_FALSE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255, 255, 0 } ) );
    EXPECT_FALSE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255, 0, 0, 255 } ) );
    EXPECT_FALSE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 0, 0, 0, 128 } ) );
    EXPECT_FALSE( isWhiteBlack<float>( HIP_AD_FORMAT_FLOAT, { 0.5f, 0.5f, 0.5f, 1.f } ) );
}

TEST( VmmWhiteBlackTileKeyTest, OtherPagesHaveNoKey )
{
    std::vector<uint8_t> page = pageOf( { 255 } );
    page[pageBytes - 1]       = 254;
    EXPECT_EQ( keyOfPage( page, HIP_AD_FORMAT_UNSIGNED_INT8, 1 ), "" );
    page[pageBytes - 1] = 255;
    page[0]             = 0;
    EXPECT_EQ( keyOfPage( page, HIP_AD_FORMAT_UNSIGNED_INT8, 1 ), "" );

    // A page of alternating texels is not uniform.
    EXPECT_EQ( keyOfPage( pageOf( { 0, 255 } ), HIP_AD_FORMAT_UNSIGNED_INT8, 1 ), "" );

    const std::vector<uint8_t> white = pageOf( { 255 } );
    std::string                key;
    EXPECT_FALSE( whiteBlackTileKey( nullptr, pageBytes, pageBytes, 1, HIP_AD_FORMAT_UNSIGNED_INT8, 1, key ) );
    EXPECT_FALSE( whiteBlackTileKey( white.data(), pageBytes, 0, 1, HIP_AD_FORMAT_UNSIGNED_INT8, 1, key ) );
    EXPECT_FALSE( whiteBlackTileKey( white.data(), pageBytes, pageBytes, 0, HIP_AD_FORMAT_UNSIGNED_INT8, 1, key ) );
    EXPECT_FALSE( whiteBlackTileKey( white.data(), 16, 17, 1, HIP_AD_FORMAT_UNSIGNED_INT8, 1, key ) );
    EXPECT_EQ( keyOfPage( white, HIP_AD_FORMAT_UNSIGNED_INT8, 0 ), "" );
    EXPECT_EQ( keyOfPage( white, HIP_AD_FORMAT_UNSIGNED_INT8, 5 ), "" );
    EXPECT_EQ( keyOfPage( white, static_cast<hipArray_Format>( 0xFF ), 1 ), "" );
}

TEST( VmmWhiteBlackTileKeyTest, OnlyTheVisibleRectangleIsLookedAt )
{
    // A 4x4 rgba8 tile whose top left 2x2 texels are white and the rest a ramp.
    std::vector<uint8_t> tile( 4 * 4 * 4 );
    for( size_t i = 0; i < tile.size(); ++i )
        tile[i] = static_cast<uint8_t>( i );
    for( uint32_t y = 0; y < 2; ++y )
        for( uint32_t x = 0; x < 2; ++x )
            std::fill_n( tile.begin() + ( y * 4 + x ) * 4, 4, 255 );

    std::string key;
    EXPECT_TRUE( whiteBlackTileKey( tile.data(), 4, 2, 2, HIP_AD_FORMAT_UNSIGNED_INT8, 4, key ) );
    EXPECT_EQ( key, std::string( 4, '\xff' ) );
    EXPECT_TRUE( whiteBlackTileKey( tile.data(), 4, 1, 1, HIP_AD_FORMAT_UNSIGNED_INT8, 4, key ) );
    EXPECT_FALSE( whiteBlackTileKey( tile.data(), 4, 3, 2, HIP_AD_FORMAT_UNSIGNED_INT8, 4, key ) );
    EXPECT_FALSE( whiteBlackTileKey( tile.data(), 4, 2, 3, HIP_AD_FORMAT_UNSIGNED_INT8, 4, key ) );
    EXPECT_FALSE( whiteBlackTileKey( tile.data(), 4, 4, 4, HIP_AD_FORMAT_UNSIGNED_INT8, 4, key ) );
}

class VmmWhiteBlackTileTest : public VmmLoaderTest
{
};

TEST_F( VmmWhiteBlackTileTest, WhiteTilesShareOnePhysicalPage )
{
    createLoader( true );
    // 1024x1024 8-bit tiles are 256x256: 16 tiles on the finest level.
    const DemandTexture&        texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 1, true, 255 ), TextureDescriptor{} );
    const DeviceTextureInfo     info    = loadTextureInfo( texture );
    const std::vector<uint32_t> pages   = loadLevel0( info );
    ASSERT_EQ( pages.size(), 16u );

    EXPECT_EQ( impl_->physicalPageCount(), 1u );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 1u );
    for( uint32_t pageId : pages )
    {
        EXPECT_TRUE( isResident( pageId ) ) << "page " << pageId;
        EXPECT_EQ( pageValue( pageId ), 255 ) << "page " << pageId;
    }
}

TEST_F( VmmWhiteBlackTileTest, RgbWhiteTilesShareOnePhysicalPage )
{
    createLoader( true );
    // 1024x1024 rgb8 tiles are 128x128 and fill 48 KiB of their 64 KiB page: 64 tiles on the finest level.
    const DemandTexture&        texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 1, true, 255, 3 ), TextureDescriptor{} );
    const DeviceTextureInfo     info    = loadTextureInfo( texture );
    const std::vector<uint32_t> pages   = loadLevel0( info );
    ASSERT_EQ( pages.size(), 64u );
    const size_t tileBytes = static_cast<size_t>( info.tileWidth() ) * info.tileHeight() * info.bytesPerTexel();
    ASSERT_EQ( tileBytes, 48u * 1024 );

    EXPECT_EQ( impl_->physicalPageCount(), 1u );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 1u );
    EXPECT_EQ( loader_->getStatistics().bytesTransferredToDevice, tileBytes );
    for( uint32_t pageId : pages )
    {
        EXPECT_TRUE( isResident( pageId ) ) << "page " << pageId;
        EXPECT_EQ( pageValue( pageId, tileBytes ), 255 ) << "page " << pageId;
    }
}

// 1000x1000 8-bit tiles are 256x256: 4x4 tiles, of which the 3x3 inner ones lie inside the level.
bool insideLevel1000( uint32_t x, uint32_t y )
{
    return x < 3 && y < 3;
}

TEST_F( VmmWhiteBlackTileTest, EdgeTilesJoinThePageOfAnInsideTile )
{
    createLoader( true );
    const DemandTexture&        texture = loader_->createTexture( std::make_shared<TestImage>( 1000, 1000, 1, true, 255 ), TextureDescriptor{} );
    const DeviceTextureInfo     info    = loadTextureInfo( texture );
    const std::vector<uint32_t> inside  = loadLevel0Tiles( info, insideLevel1000 );
    ASSERT_EQ( inside.size(), 9u );
    EXPECT_EQ( impl_->physicalPageCount(), 1u );

    const std::vector<uint32_t> edge = loadLevel0Tiles( info, []( uint32_t x, uint32_t y ) { return !insideLevel1000( x, y ); } );
    ASSERT_EQ( edge.size(), 7u );
    EXPECT_EQ( impl_->physicalPageCount(), 1u );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 1u );
    for( uint32_t pageId : edge )
    {
        EXPECT_TRUE( isResident( pageId ) ) << "page " << pageId;
        EXPECT_EQ( pageValue( pageId ), 255 ) << "page " << pageId;
    }
}

TEST_F( VmmWhiteBlackTileTest, EdgeTilesStartNoSharedPage )
{
    createLoader( true );
    const DemandTexture&        texture = loader_->createTexture( std::make_shared<TestImage>( 1000, 1000, 1, true, 255 ), TextureDescriptor{} );
    const DeviceTextureInfo     info    = loadTextureInfo( texture );
    const std::vector<uint32_t> edge    = loadLevel0Tiles( info, []( uint32_t x, uint32_t y ) { return !insideLevel1000( x, y ); } );
    ASSERT_EQ( edge.size(), 7u );
    EXPECT_EQ( impl_->physicalPageCount(), 7u );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 0u );
    for( uint32_t pageId : edge )
        EXPECT_EQ( pageValue( pageId ), -1 ) << "page " << pageId;  // the ramp past the edge is uploaded as read

    // The inside tiles then share one page of their own; the edge tiles keep theirs.
    const std::vector<uint32_t> inside = loadLevel0Tiles( info, insideLevel1000 );
    ASSERT_EQ( inside.size(), 9u );
    EXPECT_EQ( impl_->physicalPageCount(), 8u );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 1u );
}

TEST_F( VmmWhiteBlackTileTest, TexturesShareThePageOfATexel )
{
    createLoader( true );
    const DemandTexture& white         = loader_->createTexture( std::make_shared<TestImage>( 1024, 512, 1, true, 255 ), TextureDescriptor{} );
    const DemandTexture& whiteToo      = loader_->createTexture( std::make_shared<TestImage>( 512, 512, 1, true, 255 ), TextureDescriptor{} );
    const DemandTexture& black         = loader_->createTexture( std::make_shared<TestImage>( 512, 512, 1, true, 0 ), TextureDescriptor{} );
    const auto           whitePages    = loadLevel0( loadTextureInfo( white ) );
    const auto           whiteTooPages = loadLevel0( loadTextureInfo( whiteToo ) );
    const auto           blackPages    = loadLevel0( loadTextureInfo( black ) );
    ASSERT_EQ( whitePages.size(), 8u );
    ASSERT_EQ( whiteTooPages.size(), 4u );
    ASSERT_EQ( blackPages.size(), 4u );

    EXPECT_EQ( impl_->physicalPageCount(), 2u );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 2u );
    for( uint32_t pageId : whiteTooPages )
        EXPECT_EQ( pageValue( pageId ), 255 ) << "page " << pageId;
    for( uint32_t pageId : blackPages )
        EXPECT_EQ( pageValue( pageId ), 0 ) << "page " << pageId;
}

TEST_F( VmmWhiteBlackTileTest, GreyTilesGetTheirOwnPages )
{
    createLoader( true );
    const DemandTexture&        texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 1, true, 128 ), TextureDescriptor{} );
    const std::vector<uint32_t> pages   = loadLevel0( loadTextureInfo( texture ) );
    ASSERT_EQ( pages.size(), 16u );

    EXPECT_EQ( impl_->physicalPageCount(), 16u );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 0u );
    for( uint32_t pageId : pages )
        EXPECT_EQ( pageValue( pageId ), 128 ) << "page " << pageId;
}

TEST_F( VmmWhiteBlackTileTest, OtherTilesGetTheirOwnPages )
{
    createLoader( true );
    const DemandTexture&        texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 1, false, 0 ), TextureDescriptor{} );
    const std::vector<uint32_t> pages   = loadLevel0( loadTextureInfo( texture ) );
    ASSERT_EQ( pages.size(), 16u );

    EXPECT_EQ( impl_->physicalPageCount(), 16u );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 0u );
    for( uint32_t pageId : pages )
        EXPECT_EQ( pageValue( pageId ), -1 ) << "page " << pageId;
}

TEST_F( VmmWhiteBlackTileTest, SharingIsOffByDefault )
{
    createLoader( false );
    const DemandTexture&        texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 1, true, 255 ), TextureDescriptor{} );
    const std::vector<uint32_t> pages   = loadLevel0( loadTextureInfo( texture ) );

    EXPECT_EQ( impl_->physicalPageCount(), pages.size() );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 0u );
}

}  // namespace hip_demand::vmm::test
