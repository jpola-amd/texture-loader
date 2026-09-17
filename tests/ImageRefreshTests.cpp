// SPDX-License-Identifier: MIT
#include "ImageRefreshTestSupport.h"
#include "TestUtils.h"
#include "TextureSamplingHarness.h"
#include "WholeMipTestHarness.h"
#include <DemandLoading/Internal/HipCalls.h>
#include <DemandLoading/Logging.h>
#include <DemandLoading/WholeMipTexture.h>
#include <cctype>

namespace hip_demand {
namespace test {
namespace {
namespace rr = refresh;
namespace cv = contract_v1;
namespace wm = whole_mip_v1;
using cv::Outcome;
enum class Route
{
    Filename,
    Snapshot,
    WholeMip
};
struct Id
{
    uint32_t   number = InvalidTextureId;
    cv::GpuKey key{};
};
TextureDesc refreshDescriptor()
{
    TextureDesc d;
    d.filterMode       = hipFilterModePoint;
    d.mipmapFilterMode = hipFilterModePoint;
    d.addressMode[0] = d.addressMode[1] = hipAddressModeClamp;
    return d;
}
struct FaultScope
{
    std::shared_ptr<internal::HipFaultState> state = std::make_shared<internal::HipFaultState>();
    FaultScope() { internal::setHipFaultState( state ); }
    ~FaultScope() { internal::setHipFaultState( nullptr ); }
};
struct QuietScope
{
    LogLevel old = getLogLevel();
    QuietScope() { setLogLevel( LogLevel::Off ); }
    ~QuietScope() { setLogLevel( old ); }
};

class ImageRefresh : public HipTestFixture, public ::testing::WithParamInterface<Route>
{
  protected:
    TextureSamplingHarness legacyHarness;
    WholeMipHarness        wholeHarness;
    std::filesystem::path  directory, file;
    bool                   ownsDirectory = false;
    double                 maximumError  = 0;
    uint64_t               samples       = 0;
    hipStream_t            stream() const
    {
        return GetParam() == Route::WholeMip ? wholeHarness.stream() : legacyHarness.stream();
    }
    void SetUp() override
    {
        HipTestFixture::SetUp();
        if( HasFatalFailure() )
            return;
        if( GetParam() == Route::WholeMip )
        {
            ASSERT_EQ( wholeHarness.open(), hipSuccess );
            std::cout << "REFRESH_MODULE=" << wholeMipModulePath().string() << '\n';
        }
        else
        {
            ASSERT_EQ( legacyHarness.open( samplingModulePath() ), hipSuccess );
            std::cout << "REFRESH_MODULE=" << samplingModulePath().string() << '\n';
        }
        std::string name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        for( char& c : name )
            if( !std::isalnum( static_cast<unsigned char>( c ) ) )
                c = '_';
        directory =
            std::filesystem::absolute( testFileRoot() / ( "image-refresh-" + name ) ).make_preferred();
        ASSERT_FALSE( std::filesystem::exists( directory ) );
        ownsDirectory = std::filesystem::create_directory( directory );
        ASSERT_TRUE( ownsDirectory );
        file = directory / ( GetParam() == Route::Filename ? "same-image.tga" : "same-image.hdtref" );
        ASSERT_EQ( rr::liveSources, 0u );
        ASSERT_EQ( rr::liveOwners, 0u );
    }
    void TearDown() override
    {
        EXPECT_EQ( rr::liveSources, 0u );
        EXPECT_EQ( rr::liveOwners, 0u );
        EXPECT_EQ( legacyHarness.close(), hipSuccess );
        EXPECT_EQ( wholeHarness.close(), hipSuccess );
        if( ownsDirectory )
        {
            std::error_code error;
            std::filesystem::remove_all( directory, error );
            EXPECT_FALSE( error );
        }
        std::cout << std::setprecision( 12 ) << "REFRESH_MAX_ERROR=" << maximumError << " SAMPLES=" << samples << '\n';
        HipTestFixture::TearDown();
    }
    rr::Bytes write( const rr::Image& image, const std::filesystem::path& target = {} )
    {
        const auto bytes = GetParam() == Route::Filename ? rr::tga( image ) : rr::snapshotFile( image );
        rr::writeFile( target.empty() ? file : target, bytes );
        return bytes;
    }
    void pixel( float4 actual, const rr::Pixel& expected )
    {
        const rr::Pixel value{ actual.x, actual.y, actual.z, actual.w };
        ++samples;
        for( unsigned c = 0; c < 4; ++c )
        {
            maximumError = std::max( maximumError, std::abs( value[c] - expected[c] ) );
            EXPECT_NEAR( value[c], expected[c], image_refresh_data::pointTolerance ) << "channel=" << c;
        }
        std::cout << "REFRESH_PIXEL=" << actual.x << ',' << actual.y << ',' << actual.z << ',' << actual.w << '\n';
    }
    struct Owner
    {
        ImageRefresh&                                  fixture;
        const uint64_t                                 identity = rr::nextOwner++;
        std::unique_ptr<DemandTextureLoader>           legacy;
        std::unique_ptr<wm::Texture>                   whole;
        std::vector<std::shared_ptr<rr::FileSnapshot>> sources;
        DeviceContext                                  legacyContext{};
        wm::DeviceContext                              wholeContext{};
        TextureDesc                                    descriptor;
        std::filesystem::path                          path;
        Id                                             primary;
        bool                                           prepared     = false;
        Outcome                                        registration = Outcome::Pending;
        Owner( ImageRefresh& f )
            : fixture( f )
        {
            ++rr::liveOwners;
            std::cout << "REFRESH_OWNER_BEGIN=" << identity << '\n';
        }
        ~Owner()
        {
            finish();
            --rr::liveOwners;
            std::cout << "REFRESH_OWNER_END=" << identity << " liveOwners=" << rr::liveOwners
                      << " liveSources=" << rr::liveSources << '\n';
        }
        void open( TextureDesc desc = refreshDescriptor(), bool failRead = false, const std::filesystem::path& target = {} )
        {
            descriptor = desc;
            path       = target.empty() ? fixture.file : target;
            if( fixture.GetParam() != Route::WholeMip )
            {
                LoaderOptions opts;
                opts.maxTextures          = 8;
                opts.maxThreads           = 1;
                opts.maxRequestsPerLaunch = 128;
                legacy                    = std::make_unique<DemandTextureLoader>( opts );
                ASSERT_EQ( legacy->getLastError(), LoaderError::Success );
            }
            if( fixture.GetParam() != Route::Filename )
            {
                sources.push_back( std::make_shared<rr::FileSnapshot>( path, failRead ) );
                if( fixture.GetParam() == Route::WholeMip )
                {
                    whole             = std::make_unique<wm::Texture>( sources.front(), desc );
                    const auto result = whole->addSampler( desc );
                    primary.key       = result.key;
                    registration      = result.outcome;
                }
                else
                {
                    const auto result = legacy->createTextureV1( sources.front(), desc );
                    primary.number    = result.id;
                    registration      = result.valid ? Outcome::Success : Outcome::SourceFailure;
                    if( !result.valid )
                        EXPECT_EQ( result.error, LoaderError::ImageLoadFailed );
                }
            }
            else
            {
                const auto result = legacy->createTexture( path.string(), desc );
                primary.number    = result.id;
                registration      = result.valid ? Outcome::Success : Outcome::SourceFailure;
            }
            std::cout << "REFRESH_REGISTRATION owner=" << identity << " result=" << unsigned( registration )
                      << " id=" << primary.number << " key=" << primary.key.slot << ',' << primary.key.incarnation << '\n';
        }
        Id variant()
        {
            auto desc           = descriptor;
            desc.addressMode[0] = hipAddressModeWrap;
            Id result;
            if( whole )
            {
                const auto r = whole->addSampler( desc );
                EXPECT_EQ( r.outcome, Outcome::Success );
                result.key = r.key;
                EXPECT_FALSE( result.key == primary.key );
            }
            else
            {
                const auto r = fixture.GetParam() == Route::Filename ? legacy->createTexture( path.string(), desc ) :
                                                                       legacy->createTextureV1( sources.front(), desc );
                EXPECT_TRUE( r.valid );
                result.number = r.id;
                EXPECT_NE( result.number, primary.number );
            }
            return result;
        }
        Id unrelated( const std::filesystem::path& other )
        {
            Id result;
            if( fixture.GetParam() == Route::Filename )
            {
                const auto h = legacy->createTexture( other.string(), descriptor );
                EXPECT_TRUE( h.valid );
                result.number = h.id;
            }
            else
            {
                sources.push_back( std::make_shared<rr::FileSnapshot>( other ) );
                const auto h = legacy->createTextureV1( sources.back(), descriptor );
                EXPECT_TRUE( h.valid );
                result.number = h.id;
            }
            return result;
        }
        void prepare()
        {
            ASSERT_FALSE( prepared );
            if( whole )
                ASSERT_EQ( whole->prepare( fixture.stream(), wholeContext ), Outcome::Success );
            else
            {
                legacy->launchPrepare( fixture.stream() );
                legacyContext = legacy->getDeviceContext();
            }
            prepared = true;
        }
        Outcome process()
        {
            EXPECT_TRUE( prepared );
            EXPECT_EQ( hipStreamSynchronize( fixture.stream() ), hipSuccess );
            prepared = false;
            if( whole )
                return whole->processRequests();
            legacy->processRequests( fixture.stream(), legacyContext );
            if( legacy->hadRequestOverflow() )
                return Outcome::RequestOverflow;
            capability_v1::Status state;
            EXPECT_EQ( legacy->getTextureStatusV1( primary.number, state ), Outcome::Success );
            return state.state == capability_v1::State::Failed ? state.primary.outcome : Outcome::Success;
        }
        void state( unsigned width, unsigned height, unsigned first = 0 )
        {
            if( whole )
            {
                wm::Status s;
                ASSERT_EQ( whole->getStatus( s ), Outcome::Success );
                EXPECT_EQ( s.mips.originalWidth, width );
                EXPECT_EQ( s.mips.originalHeight, height );
                EXPECT_EQ( s.mips.firstResidentMip, first );
                EXPECT_EQ( s.mips.originalLevels, cv::fullMipCount( width, height ) );
                EXPECT_EQ( s.mips.resourceLevels, cv::fullMipCount( width, height ) - first );
                EXPECT_EQ( s.mips.resourceWidth, cv::mipDimension( width, first ) );
                EXPECT_EQ( s.mips.resourceHeight, cv::mipDimension( height, first ) );
                EXPECT_EQ( s.primary.outcome, Outcome::Success );
                std::cout << "REFRESH_STATUS owner=" << identity << " dimensions=" << width << 'x' << height
                          << " first=" << s.mips.firstResidentMip << " levels=" << s.mips.originalLevels
                          << " bytes=" << s.residentBytes << '\n';
            }
            else
            {
                capability_v1::Status s;
                ASSERT_EQ( legacy->getTextureStatusV1( primary.number, s ), Outcome::Success );
                EXPECT_EQ( s.originalWidth, width );
                EXPECT_EQ( s.originalHeight, height );
                EXPECT_EQ( s.originalLevels, cv::fullMipCount( width, height ) );
                EXPECT_EQ( s.resourceLevels, cv::fullMipCount( width, height ) );
                EXPECT_EQ( s.state, capability_v1::State::Resident );
                EXPECT_EQ( s.published, 1u );
                EXPECT_EQ( s.firstResidentMip, 0u );
                std::cout << "REFRESH_STATUS owner=" << identity << " dimensions=" << s.originalWidth << 'x'
                          << s.originalHeight << " first=" << s.firstResidentMip << " levels=" << s.originalLevels
                          << " bytes=" << s.payloadBytes << '\n';
            }
            for( const auto& source : sources )
                std::cout << "REFRESH_READS identity=" << source->identity << " calls=" << source->reads()
                          << " bytes=" << source->getNumBytesRead() << " fnv=" << source->contentIdentity() << '\n';
        }
        void verifyVariant( Id id )
        {
            hipTextureDesc actual{};
            if( whole )
            {
                prepare();
                wm::Entry entry;
                ASSERT_EQ( hipMemcpy( &entry, wholeContext.entries + id.key.slot, sizeof( entry ), hipMemcpyDeviceToHost ), hipSuccess );
                ASSERT_EQ( hipGetTextureObjectTextureDesc( &actual, reinterpret_cast<hipTextureObject_t>( entry.texture.textureObject ) ),
                           hipSuccess );
                ASSERT_EQ( process(), Outcome::Success );
            }
            else
            {
                capability_v1::Status status;
                ASSERT_EQ( legacy->getTextureStatusV1( id.number, status ), Outcome::Success );
                ASSERT_EQ( status.returned, 1u );
                actual = status.returnedSampler;
            }
            EXPECT_EQ( actual.addressMode[0], hipAddressModeWrap );
            EXPECT_EQ( actual.addressMode[1], hipAddressModeClamp );
            EXPECT_EQ( actual.filterMode, hipFilterModePoint );
            EXPECT_EQ( actual.mipmapFilterMode, hipFilterModePoint );
            EXPECT_EQ( actual.normalizedCoords, 1 );
            EXPECT_EQ( actual.sRGB, 0 );
        }
        void finish()
        {
            if( !legacy && !whole )
                return;
            EXPECT_EQ( hipStreamSynchronize( fixture.stream() ), hipSuccess );
            if( prepared )
                EXPECT_EQ( process(), Outcome::Success );
            legacyContext = {};
            wholeContext  = {};
            if( whole )
            {
                wm::Status s;
                EXPECT_EQ( whole->getStatus( s ), Outcome::Success );
                if( s.initialization == Outcome::Success )
                    EXPECT_EQ( whole->unload(), Outcome::Success );
                EXPECT_EQ( whole->getStatus( s ), Outcome::Success );
                EXPECT_EQ( s.residentBytes, 0u );
                EXPECT_EQ( s.retiringBytes, 0u );
                EXPECT_EQ( s.pendingBytes, 0u );
            }
            if( legacy )
            {
                legacy->unloadAll();
                EXPECT_EQ( legacy->getTotalTextureMemory(), 0u );
            }
            whole.reset();
            legacy.reset();
            for( auto& source : sources )
                source->close();
            sources.clear();
        }
    };
    std::vector<float4> sample( Owner&           owner,
                                Id               id,
                                const rr::Image& image,
                                unsigned         mip,
                                bool             complete,
                                Outcome          expectedOutcome = Outcome::Success,
                                bool             delayed         = false )
    {
        owner.prepare();
        if( HasFatalFailure() )
            return {};
        const unsigned      w = cv::mipDimension( image.width, mip ), h = cv::mipDimension( image.height, mip );
        std::vector<float4> values;
        if( owner.whole )
        {
            std::vector<WholeMipInput> in;
            for( unsigned y = 0; y < h; ++y )
                for( unsigned x = 0; x < w; ++x )
                {
                    WholeMipInput p;
                    p.key          = id.key;
                    p.lod          = float( mip );
                    p.u            = ( x + .5f ) / w;
                    p.v            = ( y + .5f ) / h;
                    p.defaultColor = { 0, 0, 0, 0 };
                    if( delayed )
                        p.path = WholeMipPath::DelayedSnapshot;
                    in.push_back( p );
                }
            std::vector<WholeMipResult> out;
            EXPECT_EQ( wholeHarness.sample( owner.wholeContext, in, out ), hipSuccess );
            for( const auto& p : out )
            {
                EXPECT_EQ( p.decision.validity, complete ? cv::SampleValidity::Complete : cv::SampleValidity::Missing );
                if( complete )
                    EXPECT_EQ( p.decision.outcome, Outcome::Success );
                values.push_back( p.value );
            }
        }
        else
        {
            std::vector<SamplingInput> in;
            for( unsigned y = 0; y < h; ++y )
                for( unsigned x = 0; x < w; ++x )
                {
                    SamplingInput p;
                    p.textureId    = id.number;
                    p.path         = delayed ? SamplingPath::DelayedSnapshot : SamplingPath::Lod;
                    p.lod          = float( mip );
                    p.u            = ( x + .5f ) / w;
                    p.v            = ( y + .5f ) / h;
                    p.defaultColor = { 0, 0, 0, 0 };
                    in.push_back( p );
                }
            std::vector<SamplingResult> out;
            EXPECT_EQ( legacyHarness.sample( owner.legacyContext, in, out ), hipSuccess );
            for( const auto& p : out )
            {
                EXPECT_EQ( p.resident, complete ? 1u : 0u );
                values.push_back( p.value );
            }
        }
        EXPECT_EQ( owner.process(), expectedOutcome );
        const auto expected = complete ? rr::expected( image, mip ) : std::vector<rr::Pixel>( w * h, rr::Pixel{ 0, 0, 0, 0 } );
        EXPECT_EQ( values.size(), expected.size() );
        for( size_t i = 0; i < values.size(); ++i )
            pixel( values[i], expected[i] );
        return values;
    }
    void load( Owner& owner, const rr::Image& image, unsigned first = 0 )
    {
        ASSERT_EQ( owner.registration, Outcome::Success );
        sample( owner, owner.primary, image, first, false );
        if( HasFatalFailure() )
            return;
        sample( owner, owner.primary, image, first, true );
        sample( owner, owner.primary, image, first, true );
        owner.state( image.width, image.height, owner.whole ? first : 0 );
    }
    void cycle( const rr::Image& image, bool restore = false, std::filesystem::file_time_type timestamp = {}, bool authored = false, unsigned first = 0 )
    {
        ASSERT_EQ( rr::liveSources, 0u );
        ASSERT_EQ( rr::liveOwners, 0u );
        write( image );
        if( restore )
        {
            std::filesystem::last_write_time( file, timestamp );
            EXPECT_EQ( std::filesystem::last_write_time( file ), timestamp );
        }
        {
            Owner owner( *this );
            auto  desc           = refreshDescriptor();
            desc.generateMipmaps = !authored;
            ASSERT_NO_FATAL_FAILURE( owner.open( desc ) );
            ASSERT_NO_FATAL_FAILURE( load( owner, image, first ) );
            const unsigned last = cv::fullMipCount( image.width, image.height ) - 1;
            for( unsigned level = first; level <= last; ++level )
                sample( owner, owner.primary, image, level, true );
            if( owner.whole && first > 0 )
            {
                sample( owner, owner.primary, image, 0, false );
                for( unsigned level = 0; level < first; ++level )
                    sample( owner, owner.primary, image, level, true );
                owner.state( image.width, image.height, 0 );
            }
        }
        EXPECT_EQ( rr::liveSources, 0u );
        EXPECT_EQ( rr::liveOwners, 0u );
    }
};

TEST_P( ImageRefresh, StoppedRedGreenColdRetryWarm )
{
    ASSERT_NO_FATAL_FAILURE( cycle( rr::solid( false ) ) );
    const auto before = rr::readFile( file );
    ASSERT_NO_FATAL_FAILURE( cycle( rr::solid( true ) ) );
    EXPECT_NE( rr::digest( before ), rr::digest( rr::readFile( file ) ) );
    EXPECT_GT( 1.0, 100 * image_refresh_data::pointTolerance );
}
TEST_P( ImageRefresh, TenAlternationsReturnAllFixtureOwnersToZero )
{
    for( unsigned i = 0; i <= image_refresh_data::alternations; ++i )
    {
        SCOPED_TRACE( i );
        ASSERT_NO_FATAL_FAILURE( cycle( rr::solid( ( i & 1 ) != 0 ) ) );
    }
}
TEST_P( ImageRefresh, RestoredTimestampAndSameAbsoluteFilename )
{
    ASSERT_NO_FATAL_FAILURE( cycle( rr::solid( false ) ) );
    const auto time     = std::filesystem::last_write_time( file );
    const auto absolute = std::filesystem::absolute( file );
    const auto old      = rr::readFile( file );
    ASSERT_NO_FATAL_FAILURE( cycle( rr::solid( true ), true, time ) );
    EXPECT_EQ( std::filesystem::absolute( file ), absolute );
    EXPECT_EQ( std::filesystem::last_write_time( file ), time );
    EXPECT_NE( old, rr::readFile( file ) );
    std::cout << "REFRESH_TIMESTAMP preserved=1 ticks=" << time.time_since_epoch().count() << '\n';
}
TEST_P( ImageRefresh, ChangedSizePatternAndAuthoredGeneratedSuffixRefinement )
{
    ASSERT_NO_FATAL_FAILURE( cycle( rr::solid( false ) ) );
    ASSERT_NO_FATAL_FAILURE( cycle( rr::pattern( true ) ) );
    ASSERT_NO_FATAL_FAILURE( cycle( rr::pattern( false ) ) );
    if( GetParam() != Route::Filename )
    {
        ASSERT_NO_FATAL_FAILURE( cycle( rr::solid( false, 8, 8, true ), false, {}, true, GetParam() == Route::WholeMip ? 2 : 0 ) );
        ASSERT_NO_FATAL_FAILURE( cycle( rr::solid( true, 8, 8, true ), false, {}, true, GetParam() == Route::WholeMip ? 2 : 0 ) );
    }
    if( GetParam() == Route::WholeMip )
    {
        const auto green = rr::pattern( true );
        write( green );
        Owner owner( *this );
        ASSERT_NO_FATAL_FAILURE( owner.open() );
        ASSERT_NO_FATAL_FAILURE( load( owner, green, 1 ) );
        sample( owner, owner.primary, green, 0, false );
        sample( owner, owner.primary, green, 0, true );
        owner.state( 4, 4, 0 );
    }
}
TEST_P( ImageRefresh, DescriptorsAlphaUnrelatedImageAndFloatSnapshots )
{
    for( bool green : { false, true } )
    {
        const auto image = rr::pattern( green );
        write( image );
        rr::Image blue = rr::solid( false );
        for( size_t p = 0; p < blue.levels[0].size(); p += 4 )
            std::copy( image_refresh_data::blue.begin(), image_refresh_data::blue.end(), blue.levels[0].begin() + p );
        const auto other = directory / ( GetParam() == Route::Filename ? "unrelated.tga" : "unrelated.hdtref" );
        write( blue, other );
        {
            Owner owner( *this );
            ASSERT_NO_FATAL_FAILURE( owner.open() );
            const auto second = owner.variant();
            if( GetParam() == Route::WholeMip )
            {
                Owner unrelated( *this );
                ASSERT_NO_FATAL_FAILURE( unrelated.open( refreshDescriptor(), false, other ) );
                ASSERT_NO_FATAL_FAILURE( load( unrelated, blue ) );
                ASSERT_NO_FATAL_FAILURE( load( owner, image ) );
                sample( owner, second, image, 0, true );
                ASSERT_NO_FATAL_FAILURE( owner.verifyVariant( second ) );
                owner.finish();
                sample( unrelated, unrelated.primary, blue, 0, true );
            }
            else
            {
                const auto unrelated = owner.unrelated( other );
                ASSERT_NO_FATAL_FAILURE( load( owner, image ) );
                sample( owner, second, image, 0, false );
                sample( owner, second, image, 0, true );
                ASSERT_NO_FATAL_FAILURE( owner.verifyVariant( second ) );
                sample( owner, unrelated, blue, 0, false );
                sample( owner, unrelated, blue, 0, true );
            }
        }
        EXPECT_EQ( rr::liveSources, 0u );
        EXPECT_EQ( rr::liveOwners, 0u );
    }
    if( GetParam() != Route::Filename )
    {
        ASSERT_NO_FATAL_FAILURE( cycle( rr::solid( false, 8, 8, false, true ) ) );
        ASSERT_NO_FATAL_FAILURE( cycle( rr::solid( true, 8, 8, false, true ) ) );
    }
}
TEST_P( ImageRefresh, MissingTruncatedReaderAndUploadFailuresRecoverGreen )
{
    QuietScope quiet;
    for( unsigned failure = 0; failure < 4; ++failure )
    {
        if( failure == 2 && GetParam() == Route::Filename )
            continue;  // Reader seam belongs to the explicit snapshot route.
        ASSERT_NO_FATAL_FAILURE( cycle( rr::solid( false ) ) );
        FaultScope faults;
        if( failure == 0 )
            ASSERT_TRUE( std::filesystem::remove( file ) );
        else if( failure == 1 )
            rr::writeFile( file, rr::Bytes{ 0, 1, 2 } );
        else
            write( rr::solid( true ) );
        {
            Owner owner( *this );
            ASSERT_NO_FATAL_FAILURE( owner.open( refreshDescriptor(), failure == 2 ) );
            if( owner.registration != Outcome::Success )
            {
                EXPECT_NE( GetParam(), Route::Filename );
                EXPECT_EQ( owner.registration, Outcome::SourceFailure );
                if( owner.whole )
                {
                    wm::Status s;
                    ASSERT_EQ( owner.whole->getStatus( s ), Outcome::Success );
                    EXPECT_EQ( s.initialization, Outcome::SourceFailure );
                    EXPECT_EQ( s.device.published, 0u );
                    std::cout << "REFRESH_FAILURE kind=" << failure
                              << " initialization=" << unsigned( s.initialization ) << " published=0\n";
                }
            }
            else
            {
                if( failure == 3 )
                    faults.state->fail( internal::HipOperation::Upload, hipErrorOutOfMemory );
                sample( owner, owner.primary, rr::solid( true ), 0, false,
                        failure == 3 ? Outcome::DeviceOutOfMemory : Outcome::SourceFailure );
                if( owner.whole )
                {
                    wm::Status s;
                    ASSERT_EQ( owner.whole->getStatus( s ), Outcome::Success );
                    EXPECT_EQ( s.device.published, 0u );
                    EXPECT_EQ( s.residentBytes, 0u );
                    EXPECT_EQ( s.primary.outcome, failure == 3 ? Outcome::DeviceOutOfMemory : Outcome::SourceFailure );
                    std::cout << "REFRESH_FAILURE kind=" << failure << " outcome=" << unsigned( s.primary.outcome ) << " published=0\n";
                }
                else
                {
                    capability_v1::Status s;
                    ASSERT_EQ( owner.legacy->getTextureStatusV1( owner.primary.number, s ), Outcome::Success );
                    EXPECT_EQ( s.state, capability_v1::State::Failed );
                    EXPECT_EQ( s.published, 0u );
                    EXPECT_EQ( s.primary.outcome, failure == 3 ? Outcome::DeviceOutOfMemory : Outcome::SourceFailure );
                    std::cout << "REFRESH_FAILURE kind=" << failure << " outcome=" << unsigned( s.primary.outcome ) << " published=0\n";
                }
            }
        }
        EXPECT_EQ( rr::liveSources, 0u );
        EXPECT_EQ( rr::liveOwners, 0u );
        ASSERT_NO_FATAL_FAILURE( cycle( rr::solid( true ) ) );
    }
}
TEST_P( ImageRefresh, StoppedOrderingAndOldIdentityOnlyAgainstLiveNewContext )
{
    cv::GpuKey oldKey;
    uint64_t   oldOwner = 0;
    uint32_t   oldId    = InvalidTextureId;
    const auto red      = rr::solid( false );
    write( red );
    {
        Owner owner( *this );
        ASSERT_NO_FATAL_FAILURE( owner.open() );
        ASSERT_NO_FATAL_FAILURE( load( owner, red ) );
        oldKey   = owner.primary.key;
        oldId    = owner.primary.number;
        oldOwner = owner.identity;
        sample( owner, owner.primary, red, 3, true, Outcome::Success, true );
        owner.finish();
        EXPECT_EQ( owner.legacyContext.textures, nullptr );
        EXPECT_EQ( owner.wholeContext.entries, nullptr );
    }
    EXPECT_EQ( rr::liveSources, 0u );
    EXPECT_EQ( rr::liveOwners, 0u );
    const auto green = rr::solid( true );
    write( green );
    Owner owner( *this );
    ASSERT_NO_FATAL_FAILURE( owner.open() );
    ASSERT_NO_FATAL_FAILURE( load( owner, green ) );
    EXPECT_NE( owner.identity, oldOwner );  // Caller cache token, not a legacy API guarantee.
    if( owner.whole )
    {
        owner.prepare();
        WholeMipInput input;
        input.key          = oldKey;
        input.defaultColor = { 0, 0, 0, 0 };
        std::vector<WholeMipResult> out;
        ASSERT_EQ( wholeHarness.sample( owner.wholeContext, { input }, out ), hipSuccess );
        EXPECT_EQ( out[0].decision.outcome, Outcome::InvalidKey );
        EXPECT_EQ( out[0].value.x, 0 );
        EXPECT_EQ( out[0].value.y, 0 );
        EXPECT_EQ( out[0].value.z, 0 );
        EXPECT_EQ( out[0].value.w, 0 );
        uint32_t count = 99;
        ASSERT_EQ( hipMemcpy( &count, owner.wholeContext.requestCount, 4, hipMemcpyDeviceToHost ), hipSuccess );
        EXPECT_EQ( count, 0u );
        EXPECT_EQ( owner.process(), Outcome::Success );
    }
    else
        EXPECT_EQ( owner.primary.number, oldId );  // Numeric zero may be reused; old context is never used.
    sample( owner, owner.primary, green, 0, true );
}
INSTANTIATE_TEST_SUITE_P( AllRoutes, ImageRefresh, ::testing::Values( Route::Filename, Route::Snapshot, Route::WholeMip ) );
}  // namespace
}  // namespace test
}  // namespace hip_demand
