// SPDX-License-Identifier: MIT
// Unit tests for the level geometry of the VMM loader's DeviceTextureInfo

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <DemandLoading/DeviceContext.h>

namespace hip_demand::vmm::test {

namespace {

constexpr uint32_t pageBytes   = 64 * 1024;
constexpr uint32_t mipTailPage = 77;

DeviceTextureInfo makeInfo( uint32_t width, uint32_t height, uint32_t bytesPerTexel, uint32_t tileShift, uint32_t mipCount, uint32_t mipTailFirstLevel )
{
    DeviceTextureInfo info{};
    info.width  = width;
    info.height = height;
    info.setTexel( HIP_AD_FORMAT_UNSIGNED_INT8, 4, bytesPerTexel );
    info.setLayout( tileShift, tileShift, mipCount, mipTailFirstLevel );
    info.mipTailPage = mipTailPage;
    for( uint32_t level = 0; level < mipTailFirstLevel; ++level )
        info.mips[level] = 1000 + level;
    return info;
}

// The offset readMipTail() gives a tail level: the sizes of the tail levels before it, packed without padding.
size_t packedTailOffset( const DeviceTextureInfo& info, uint32_t level )
{
    size_t offset = 0;
    for( uint32_t tailLevel = info.mipTailFirstLevel(); tailLevel < level; ++tailLevel )
    {
        const uint32_t width  = std::max( info.width >> tailLevel, 1u );
        const uint32_t height = std::max( info.height >> tailLevel, 1u );
        offset += static_cast<size_t>( width ) * height * info.bytesPerTexel();
    }
    return offset;
}

void expectTailLevelsPacked( const DeviceTextureInfo& info )
{
    for( uint32_t level = info.mipTailFirstLevel(); level < info.mipCount(); ++level )
    {
        const DeviceMipLevel mip = info.getMipLevel( level );
        EXPECT_TRUE( mip.mipTail ) << "level " << level;
        EXPECT_EQ( mip.startPage, mipTailPage ) << "level " << level;
        EXPECT_EQ( mip.mipTailOffset, packedTailOffset( info, level ) ) << "level " << level;
        EXPECT_EQ( mip.width, std::max( info.width >> level, 1u ) ) << "level " << level;
        EXPECT_EQ( mip.height, std::max( info.height >> level, 1u ) ) << "level " << level;
    }
    EXPECT_EQ( info.mipTailLevelOffset( info.mipCount() ), packedTailOffset( info, info.mipCount() ) );
    EXPECT_LE( info.mipTailLevelOffset( info.mipCount() ), pageBytes );
}

}  // namespace

TEST( VmmDeviceTextureInfoTest, TiledLevelsReadTheirWord )
{
    // 4096x4096 RGBA8 with 128x128 tiles: 13 levels, the tail from 64x64 (level 6) down.
    const DeviceTextureInfo info = makeInfo( 4096, 4096, 4, 7, 13, 6 );
    for( uint32_t level = 0; level < info.mipTailFirstLevel(); ++level )
    {
        const DeviceMipLevel mip = info.getMipLevel( level );
        EXPECT_FALSE( mip.mipTail ) << "level " << level;
        EXPECT_EQ( mip.startPage, 1000 + level ) << "level " << level;
        EXPECT_EQ( mip.mipTailOffset, 0u ) << "level " << level;
        EXPECT_EQ( mip.tilesX, std::max( ( 4096u >> level ) / 128u, 1u ) ) << "level " << level;
    }
    EXPECT_EQ( info.mipTailLevelOffset( info.mipTailFirstLevel() ), 0u );
}

TEST( VmmDeviceTextureInfoTest, TailLevelsArePackedInOrder )
{
    const DeviceTextureInfo info = makeInfo( 4096, 4096, 4, 7, 13, 6 );
    expectTailLevelsPacked( info );
    // 64x64 RGBA8 is 16 KiB, so the second tail level starts there.
    EXPECT_EQ( info.getMipLevel( 7 ).mipTailOffset, 64u * 64u * 4u );
    EXPECT_EQ( info.getMipLevel( 8 ).mipTailOffset, ( 64u * 64u + 32u * 32u ) * 4u );
}

TEST( VmmDeviceTextureInfoTest, NonPowerOfTwoTailLevelsArePacked )
{
    // 1000x600 with 2 bytes per texel: 10 levels, dimensions round down and stop at 1.
    const DeviceTextureInfo info = makeInfo( 1000, 600, 2, 7, 10, 3 );
    expectTailLevelsPacked( info );
    EXPECT_EQ( info.getMipLevel( 9 ).width, 1u );
    EXPECT_EQ( info.getMipLevel( 9 ).height, 1u );
}

TEST( VmmDeviceTextureInfoTest, TailMayHoldMoreLevelsThanTiledSlots )
{
    // 32768x32768 RGBA8: 16 levels, nine of them tiled, the tail from 64x64 (level 9) down to 1x1 (level 15).
    const DeviceTextureInfo info = makeInfo( 32768, 32768, 4, 7, 16, MAX_TILED_MIP_LEVELS );
    ASSERT_GT( info.mipCount(), MAX_TILED_MIP_LEVELS );
    expectTailLevelsPacked( info );

    const DeviceMipLevel last = info.getMipLevel( info.mipCount() - 1 );
    EXPECT_EQ( last.width, 1u );
    EXPECT_EQ( last.startPage, mipTailPage );
    EXPECT_EQ( last.mipTailOffset, packedTailOffset( info, info.mipCount() - 1 ) );
}

}  // namespace hip_demand::vmm::test
