// SPDX-License-Identifier: MIT
// Tests for the VMM loader's getStatistics().

#include "VmmLoaderTestUtils.h"
#include <ImageSource/TextureInfo.h>

namespace hip_demand::vmm::test {

namespace {

constexpr size_t pageBytes = PageSystem::PAGE_BYTES;

class VmmStatisticsTest : public VmmLoaderTest
{
};

}  // namespace

TEST_F( VmmStatisticsTest, FreshLoaderHasNoActivity )
{
    createLoader( false );
    const Statistics stats = loader_->getStatistics();
    EXPECT_EQ( stats.numTextures, 0u );
    EXPECT_EQ( stats.numTilesRead, 0u );
    EXPECT_EQ( stats.numBytesRead, 0u );
    EXPECT_EQ( stats.readTime, 0.0 );
    EXPECT_EQ( stats.virtualTextureBytes, 0u );
    EXPECT_EQ( stats.bytesTransferredToDevice, 0u );
    EXPECT_EQ( stats.numEvictions, 0u );
    EXPECT_EQ( stats.deviceMemoryUsed, context_.residenceBits.sizeInBytes() + options_.maxTextures * sizeof( DeviceTextureInfo )
                                           + context_.referenceBits.sizeInBytes() + context_.requestedResources.sizeInBytes()
                                           + context_.processedResources.sizeInBytes() + context_.counters.sizeInBytes() );
}

TEST_F( VmmStatisticsTest, LoadedTilesAreCounted )
{
    createLoader( false );
    const auto                  image   = std::make_shared<TestImage>( 1024, 1024, 1, false, 0 );
    const DemandTexture&        texture = loader_->createTexture( image, TextureDescriptor{} );
    const Statistics            before  = loader_->getStatistics();
    const size_t                contextBytes = before.deviceMemoryUsed;
    EXPECT_EQ( before.numTextures, 1u );
    EXPECT_EQ( before.numTilesRead, 0u );
    EXPECT_EQ( before.bytesTransferredToDevice, 0u );

    const DeviceTextureInfo     info  = loadTextureInfo( texture );
    const std::vector<uint32_t> pages = loadLevel0( info );
    ASSERT_EQ( pages.size(), 16u );

    const Statistics stats = loader_->getStatistics();
    EXPECT_EQ( stats.numTextures, 1u );
    EXPECT_EQ( stats.numTilesRead, 16u );
    EXPECT_EQ( stats.numBytesRead, 1024u * 1024u );
    EXPECT_EQ( stats.virtualTextureBytes, getTextureSizeInBytes( image->getInfo() ) );
    EXPECT_EQ( stats.bytesTransferredToDevice, 16 * pageBytes );
    EXPECT_EQ( stats.deviceMemoryUsed, contextBytes + 16 * pageBytes );
    EXPECT_EQ( stats.numEvictions, 0u );
}

TEST_F( VmmStatisticsTest, SharedImagesAreCountedOnce )
{
    createLoader( false );
    const auto image = std::make_shared<TestImage>( 1024, 1024, 1, false, 0 );
    loader_->createTexture( image, TextureDescriptor{} );
    const DemandTexture& second = loader_->createTexture( image, TextureDescriptor{} );
    loadLevel0( loadTextureInfo( second ) );

    const Statistics stats = loader_->getStatistics();
    EXPECT_EQ( stats.numTextures, 2u );
    EXPECT_EQ( stats.numTilesRead, 16u );
    EXPECT_EQ( stats.virtualTextureBytes, getTextureSizeInBytes( image->getInfo() ) );
}

TEST_F( VmmStatisticsTest, EmptyUdimCellsAreNotTextures )
{
    createLoader( false );
    const auto image = std::make_shared<TestImage>( 1024, 1024, 1, false, 0 );
    loader_->createUdimTexture( { image, nullptr, nullptr, image }, std::vector<TextureDescriptor>( 4 ), 2, 2 );

    const Statistics stats = loader_->getStatistics();
    EXPECT_EQ( stats.numTextures, 2u );
    EXPECT_EQ( stats.virtualTextureBytes, getTextureSizeInBytes( image->getInfo() ) );
}

TEST_F( VmmStatisticsTest, SharedWhiteTilesAreTransferredOnce )
{
    createLoader( true );
    const DemandTexture& texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 1, true, 255 ), TextureDescriptor{} );
    const size_t         contextBytes = loader_->getStatistics().deviceMemoryUsed;
    loadLevel0( loadTextureInfo( texture ) );

    const Statistics stats = loader_->getStatistics();
    EXPECT_EQ( stats.numTilesRead, 16u );
    EXPECT_EQ( stats.bytesTransferredToDevice, pageBytes );
    EXPECT_EQ( stats.deviceMemoryUsed, contextBytes + pageBytes );
}

}  // namespace hip_demand::vmm::test
