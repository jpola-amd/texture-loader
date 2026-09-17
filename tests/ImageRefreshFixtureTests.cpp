// SPDX-License-Identifier: MIT
#include "ImageRefreshTestSupport.h"

namespace hip_demand {
namespace test {
namespace {
namespace rr = refresh;
class ImageRefreshFixture : public ::testing::Test
{
  protected:
    std::filesystem::path directory;
    bool                  owned = false;
    void                  SetUp() override
    {
        directory = std::filesystem::absolute(
                        testFileRoot()
                        / ( "refresh-host-" + std::string( ::testing::UnitTest::GetInstance()->current_test_info()->name() ) ) )
                        .make_preferred();
        ASSERT_FALSE( std::filesystem::exists( directory ) );
        owned = std::filesystem::create_directory( directory );
        ASSERT_TRUE( owned );
    }
    void TearDown() override
    {
        EXPECT_EQ( rr::liveSources, 0u );
        if( owned )
        {
            std::error_code error;
            std::filesystem::remove_all( directory, error );
            EXPECT_FALSE( error );
        }
    }
};
TEST_F( ImageRefreshFixture, TgaHasExactFourChannelTopOriginBytes )
{
    const auto image = rr::pattern( false );
    const auto bytes = rr::tga( image );
    ASSERT_EQ( bytes.size(), 18 + 64u );
    EXPECT_EQ( bytes[2], 2 );
    EXPECT_EQ( bytes[16], 32 );
    EXPECT_EQ( bytes[17], 0x28 );
    for( unsigned p = 0; p < 16; ++p )
    {
        EXPECT_EQ( bytes[18 + 4 * p], image.levels[0][4 * p + 2] );
        EXPECT_EQ( bytes[18 + 4 * p + 2], image.levels[0][4 * p] );
        EXPECT_EQ( bytes[18 + 4 * p + 3], image.levels[0][4 * p + 3] );
    }
}
TEST_F( ImageRefreshFixture, FreshSnapshotReopensSameBytesWithRestoredTimestamp )
{
    const auto path    = directory / "same.hdtref";
    uint64_t   oldHash = 0, oldIdentity = 0;
    rr::writeFile( path, rr::snapshotFile( rr::solid( false ) ) );
    const auto timestamp = std::filesystem::last_write_time( path );
    {
        rr::FileSnapshot source( path );
        TextureInfo      info;
        source.open( &info );
        oldHash     = source.contentIdentity();
        oldIdentity = source.identity;
        EXPECT_EQ( source.getHash(), 0u );
        EXPECT_EQ( info.width, 8u );
    }
    EXPECT_EQ( rr::liveSources, 0u );
    rr::writeFile( path, rr::snapshotFile( rr::solid( true ) ) );
    std::filesystem::last_write_time( path, timestamp );
    {
        rr::FileSnapshot source( path );
        TextureInfo      info;
        source.open( &info );
        EXPECT_NE( source.contentIdentity(), oldHash );
        EXPECT_NE( source.identity, oldIdentity );
        std::vector<char> pixels( 8 * 8 * 4 );
        ASSERT_TRUE( source.readMipLevel( pixels.data(), 0, 8, 8, nullptr ) );
        EXPECT_EQ( uint8_t( pixels[0] ), 0 );
        EXPECT_EQ( uint8_t( pixels[1] ), 255 );
        EXPECT_EQ( uint8_t( pixels[3] ), 255 );
    }
    EXPECT_EQ( std::filesystem::last_write_time( path ), timestamp );
}
TEST_F( ImageRefreshFixture, AuthoredFloatSnapshotAndMalformedPayloads )
{
    const auto image   = rr::solid( true, 8, 8, true, true );
    const auto bytes   = rr::snapshotFile( image );
    const auto decoded = rr::decodeSnapshot( bytes );
    EXPECT_EQ( decoded.levels, image.levels );
    auto truncated = bytes;
    truncated.pop_back();
    EXPECT_THROW( rr::decodeSnapshot( truncated ), std::runtime_error );
    auto trailing = bytes;
    trailing.push_back( 0 );
    EXPECT_THROW( rr::decodeSnapshot( trailing ), std::runtime_error );
    auto malformed = bytes;
    malformed[0]   = 0;
    EXPECT_THROW( rr::decodeSnapshot( malformed ), std::runtime_error );
    EXPECT_THROW( rr::readFile( directory / "missing" ), std::runtime_error );
}
TEST_F( ImageRefreshFixture, IndependentPatternMipRounding )
{
    const auto expected = rr::expected( rr::pattern( false ), 1 );
    ASSERT_EQ( expected.size(), 4u );
    EXPECT_DOUBLE_EQ( expected[0][0], 56. / 255 );
    EXPECT_DOUBLE_EQ( expected[0][3], 56. / 255 );
    EXPECT_DOUBLE_EQ( expected[3][0], 152. / 255 );
    EXPECT_DOUBLE_EQ( expected[3][3], 216. / 255 );
    EXPECT_EQ( rr::expected( rr::pattern( false ), 2 ).size(), 1u );
}
}  // namespace
}  // namespace test
}  // namespace hip_demand
