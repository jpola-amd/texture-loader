// SPDX-License-Identifier: MIT
#include "CubicExpected.h"
#include "CubicNativeReference.h"
#include "CubicReference.h"
#include "CubicSmartExpected.h"
#include "TestPaths.h"
#include "CubicTestData.h"
#include "ImageDataTestUtils.h"
#include "TestUtils.h"
#include "TextureSamplingHarness.h"
#include <DemandLoading/Internal/HipCalls.h>
#include <DemandLoading/Internal/TextureRuntime.h>
#include <DemandLoading/WholeMipTexture.h>
#include <fstream>
#include <iomanip>
#include <set>

namespace hip_demand {
namespace test {
namespace {
namespace cv  = contract_v1;
namespace cb  = cubic_v1;
namespace wm  = whole_mip_v1;
namespace ref = cubic_reference;
using cv::Outcome;

class AnisoEnvironment
{
    bool        present_;
    std::string previous_;

  public:
    explicit AnisoEnvironment( const char* value )
    {
        const char* old = std::getenv( "HDT_DISABLE_TEXTURE_ANISO_OVERRIDE" );
        present_        = old != nullptr;
        if( old )
            previous_ = old;
        set( value );
    }
    ~AnisoEnvironment() { set( present_ ? previous_.c_str() : nullptr ); }
    static void set( const char* value )
    {
#ifdef _WIN32
        EXPECT_EQ( _putenv_s( "HDT_DISABLE_TEXTURE_ANISO_OVERRIDE", value ? value : "" ), 0 );
#else
        EXPECT_EQ( value ? setenv( "HDT_DISABLE_TEXTURE_ANISO_OVERRIDE", value, 1 ) : unsetenv( "HDT_DISABLE_TEXTURE_ANISO_OVERRIDE" ), 0 );
#endif
    }
};

struct Faults
{
    std::shared_ptr<internal::HipFaultState> state = std::make_shared<internal::HipFaultState>();
    Faults() { internal::setHipFaultState( state ); }
    ~Faults() { internal::setHipFaultState( nullptr ); }
};

std::vector<ref::Level> patterns( unsigned w, unsigned h, unsigned count = 0, bool constant = false )
{
    if( !count )
        count = cv::fullMipCount( w, h );
    std::vector<ref::Level> result;
    for( unsigned m = 0; m < count; ++m )
    {
        ref::Level l{ cv::mipDimension( w, m ), cv::mipDimension( h, m ), {} };
        for( unsigned y = 0; y < l.height; ++y )
            for( unsigned x = 0; x < l.width; ++x )
                l.pixels.push_back( constant ? ref::Pixel{ .25, -.5, 2, .75 } :
                                               ref::Pixel{ double( int( x % 5 ) - 2 ) / 2 + double( m ) / 8,
                                                           double( int( y % 5 ) - 2 ) / 2, double( ( x + y ) % 2 ),
                                                           double( ( x + 2 * y ) % 7 ) / 8 } );
        result.push_back( std::move( l ) );
    }
    return result;
}
std::shared_ptr<TypedImageSource> sourceFor( const std::vector<ref::Level>& levels )
{
    auto source               = std::make_shared<TypedImageSource>();
    source->info.width        = levels[0].width;
    source->info.height       = levels[0].height;
    source->info.numChannels  = 4;
    source->info.format       = HIP_AD_FORMAT_FLOAT;
    source->info.numMipLevels = unsigned( levels.size() );
    source->info.isValid      = true;
    for( const auto& l : levels )
    {
        auto& bytes = source->mipPixels.emplace_back( l.pixels.size() * sizeof( float4 ) );
        for( size_t i = 0; i < l.pixels.size(); ++i )
        {
            const float p[4]{ float( l.pixels[i][0] ), float( l.pixels[i][1] ), float( l.pixels[i][2] ), float( l.pixels[i][3] ) };
            std::memcpy( bytes.data() + i * sizeof( p ), p, sizeof( p ) );
        }
    }
    source->pixels = source->mipPixels.front();
    return source;
}
TextureDesc descriptor( bool linear = true )
{
    TextureDesc d;
    d.addressMode[0] = d.addressMode[1] = hipAddressModeClamp;
    d.mipmapFilterMode                  = linear ? hipFilterModeLinear : hipFilterModePoint;
    return d;
}
ref::Address referenceAddress( hipTextureAddressMode mode )
{
    switch( mode )
    {
        case hipAddressModeWrap:
            return ref::Wrap;
        case hipAddressModeClamp:
            return ref::Clamp;
        case hipAddressModeMirror:
            return ref::Mirror;
        default:
            return ref::Border;
    }
}
LoaderOptions loaderOptions()
{
    LoaderOptions o;
    o.maxTextures          = 32;
    o.maxRequestsPerLaunch = 512;
    o.maxThreads           = 1;
    return o;
}

struct Backing
{
    std::unique_ptr<DemandTextureLoader> legacy;
    std::unique_ptr<wm::Texture>         whole;
    cv::GpuKey                           key{};
    uint32_t                             id = InvalidTextureId;
    void                                 open( bool                              suffix,
                                               std::shared_ptr<TypedImageSource> source,
                                               const TextureDesc&                desc,
                                               const anisotropy_v1::Request&     a = anisotropy_v1::Request::legacy(),
                                               unsigned                          maxRequests = 512 )
    {
        if( suffix )
        {
            wm::Options o;
            o.maxSamplers     = 8;
            o.maxRequests     = maxRequests;
            whole             = std::make_unique<wm::Texture>( source, desc, o );
            auto registration = whole->addSampler( desc, cv::SamplingPolicy::Strict, a );
            ASSERT_EQ( registration.outcome, Outcome::Success );
            key = registration.key;
            ASSERT_EQ( whole->enableCubicV1( key ), Outcome::Success );
        }
        else
        {
            auto o                 = loaderOptions();
            o.maxRequestsPerLaunch = maxRequests;
            legacy                 = std::make_unique<DemandTextureLoader>( o );
            auto handle            = legacy->createTextureAnisotropyV1( source, desc, a );
            ASSERT_TRUE( handle.valid );
            id                = handle.id;
            auto registration = legacy->enableCubicV1( id );
            ASSERT_EQ( registration.outcome, Outcome::Success );
            key = registration.key;
        }
    }
    cb::DeviceContext prepare( hipStream_t stream )
    {
        cb::DeviceContext context;
        if( whole )
            EXPECT_EQ( whole->prepareCubicV1( stream, context ), Outcome::Success );
        else
        {
            legacy->launchPrepare( stream );
            context = legacy->getCubicContextV1();
        }
        return context;
    }
    Outcome process( hipStream_t stream, const cb::DeviceContext& context )
    {
        if( whole )
            return whole->processRequests();
        legacy->processRequests( stream, context.legacy );
        if( legacy->hadRequestOverflow() )
            return Outcome::RequestOverflow;
        capability_v1::Status status;
        EXPECT_EQ( legacy->getTextureStatusV1( id, status ), Outcome::Success );
        return status.state == capability_v1::State::Failed ? status.primary.outcome : Outcome::Success;
    }
};

class CubicSampling : public HipTestFixture
{
  protected:
    hipModule_t   module_        = nullptr;
    hipFunction_t kernel_        = nullptr;
    hipFunction_t nativeKernel_  = nullptr;
    hipStream_t   stream_        = nullptr;
    CubicInput*   input_         = nullptr;
    CubicOutput*  output_        = nullptr;
    double        maxValueError_ = 0, maxRawDerivativeError_ = 0, maxNormalizedDerivativeError_ = 0;
    uint64_t      sampleCount_ = 0, nativeSampleCount_ = 0;
    void          SetUp() override
    {
        HipTestFixture::SetUp();
        if( HasFatalFailure() )
            return;
        const auto            name = std::filesystem::path( CUBIC_KERNEL_PATH ).filename();
        std::filesystem::path path;
        ASSERT_NO_THROW( path = findSamplingModule(
                             std::getenv( "HIP_DEMAND_TEST_KERNEL_DIR" ) ?
                                 std::vector<std::filesystem::path>{
                                     std::filesystem::path( std::getenv( "HIP_DEMAND_TEST_KERNEL_DIR" ) ) / name } :
                                 std::vector<std::filesystem::path>{ samplingModulePath().parent_path() / name, CUBIC_KERNEL_PATH } ) );
        std::cout << "Cubic module=" << path.string() << '\n';
        ASSERT_EQ( hipModuleLoad( &module_, path.string().c_str() ), hipSuccess );
        ASSERT_EQ( hipModuleGetFunction( &kernel_, module_, "sampleCubic" ), hipSuccess );
        ASSERT_EQ( hipModuleGetFunction( &nativeKernel_, module_, "sampleCubicNativePrimitive" ), hipSuccess );
        ASSERT_EQ( hipStreamCreateWithFlags( &stream_, hipStreamNonBlocking ), hipSuccess );
        ASSERT_EQ( hipMalloc( &input_, 512 * sizeof( CubicInput ) ), hipSuccess );
        ASSERT_EQ( hipMalloc( &output_, 512 * sizeof( CubicOutput ) ), hipSuccess );
    }
    void TearDown() override
    {
        if( stream_ )
            EXPECT_EQ( hipStreamSynchronize( stream_ ), hipSuccess );
        if( input_ )
            EXPECT_EQ( hipFree( input_ ), hipSuccess );
        if( output_ )
            EXPECT_EQ( hipFree( output_ ), hipSuccess );
        if( module_ )
            EXPECT_EQ( hipModuleUnload( module_ ), hipSuccess );
        if( stream_ )
            EXPECT_EQ( hipStreamDestroy( stream_ ), hipSuccess );
        std::cout << std::setprecision( 12 ) << "Cubic errors: value=" << maxValueError_ << " rawDerivative=" << maxRawDerivativeError_
                  << " normalizedDerivative=" << maxNormalizedDerivativeError_ << '\n';
        std::cout << "Cubic GPU invocations=" << sampleCount_ << " native control invocations=" << nativeSampleCount_ << '\n';
        HipTestFixture::TearDown();
    }
    std::vector<CubicOutput> run( cb::DeviceContext              context,
                                  const std::vector<CubicInput>& inputs,
                                  const std::function<void()>&   afterLaunch = {} )
    {
        EXPECT_FALSE( inputs.empty() );
        EXPECT_LE( inputs.size(), 512u );
        sampleCount_ += inputs.size();
        uint32_t                 count = uint32_t( inputs.size() );
        std::vector<CubicOutput> out( count );
        EXPECT_EQ( hipMemcpyAsync( input_, inputs.data(), count * sizeof( CubicInput ), hipMemcpyHostToDevice, stream_ ), hipSuccess );
        void* args[]{ &context, &input_, &output_, &count };
        EXPECT_EQ( hipModuleLaunchKernel( kernel_, ( count + 63 ) / 64, 1, 1, 64, 1, 1, 0, stream_, args, nullptr ), hipSuccess );
        if( afterLaunch )
            afterLaunch();
        EXPECT_EQ( hipMemcpyAsync( out.data(), output_, count * sizeof( CubicOutput ), hipMemcpyDeviceToHost, stream_ ), hipSuccess );
        EXPECT_EQ( hipStreamSynchronize( stream_ ), hipSuccess );
        return out;
    }
    std::vector<CubicOutput> batch( Backing& backing, std::vector<CubicInput> inputs, bool process = true )
    {
        for( auto& in : inputs )
            in.key = backing.key;
        auto context = backing.prepare( stream_ );
        auto out     = run( context, inputs );
        if( process )
            EXPECT_EQ( backing.process( stream_, context ), Outcome::Success );
        return out;
    }
    std::vector<CubicOutput> nativeRun( TextureObject object, const std::vector<CubicInput>& inputs )
    {
        nativeSampleCount_ += inputs.size();
        uint32_t                 count = uint32_t( inputs.size() );
        std::vector<CubicOutput> result( count );
        auto                     handle = reinterpret_cast<hipTextureObject_t>( object );
        EXPECT_EQ( hipMemcpyAsync( input_, inputs.data(), count * sizeof( CubicInput ), hipMemcpyHostToDevice, stream_ ), hipSuccess );
        void* args[]{ &handle, &input_, &output_, &count };
        EXPECT_EQ( hipModuleLaunchKernel( nativeKernel_, ( count + 63 ) / 64, 1, 1, 64, 1, 1, 0, stream_, args, nullptr ), hipSuccess );
        EXPECT_EQ( hipMemcpyAsync( result.data(), output_, count * sizeof( CubicOutput ), hipMemcpyDeviceToHost, stream_ ), hipSuccess );
        EXPECT_EQ( hipStreamSynchronize( stream_ ), hipSuccess );
        return result;
    }
    void warm( Backing& b )
    {
        CubicInput in;
        const auto cold = batch( b, { in } );
        ASSERT_EQ( cold[0].decision.validity, cv::SampleValidity::Missing );
        const auto hot = batch( b, { in } );
        ASSERT_EQ( hot[0].decision.validity, cv::SampleValidity::Complete );
    }
    void check( const CubicOutput& actual, const ref::Result& expected, unsigned w, unsigned h, unsigned mask = 7 )
    {
        const std::array<float, 4> v{ actual.value.x, actual.value.y, actual.value.z, actual.value.w };
        const std::array<float, 4> s{ actual.ds.x, actual.ds.y, actual.ds.z, actual.ds.w };
        const std::array<float, 4> t{ actual.dt.x, actual.dt.y, actual.dt.z, actual.dt.w };
        EXPECT_EQ( actual.decision.validity, cv::SampleValidity::Complete );
        for( unsigned k = 0; k < 4; ++k )
        {
            if( mask & 1 )
            {
                maxValueError_ = std::max( maxValueError_, std::abs( v[k] - expected.value[k] ) );
                EXPECT_NEAR( v[k], expected.value[k], ref::valueTolerance( expected.value[k] ) ) << "channel=" << k;
            }
            else
                EXPECT_EQ( v[k], 0 );
            for( unsigned axis = 0; axis < 2; ++axis )
            {
                const double got = axis ? t[k] : s[k], want = axis ? expected.dt[k] : expected.ds[k];
                const double dimension = axis ? h : w;
                if( mask & ( axis ? 4 : 2 ) )
                {
                    maxRawDerivativeError_ = std::max( maxRawDerivativeError_, std::abs( got - want ) );
                    maxNormalizedDerivativeError_ = std::max( maxNormalizedDerivativeError_, std::abs( got - want ) / dimension );
                    EXPECT_NEAR( got / dimension, want / dimension, ref::derivativeTolerance( want / dimension ) )
                        << "channel=" << k << " axis=" << axis << " raw=" << got << " expected=" << want;
                }
                else
                    EXPECT_EQ( got, 0 );
            }
        }
    }
};
class CubicBackings : public CubicSampling, public ::testing::WithParamInterface<bool>
{
};

TEST_P( CubicBackings, KernelCoordinatesAndSensitivity )
{
    AnisoEnvironment env( "0" );
    auto             levels = patterns( 8, 8, 1 );
    for( auto& p : levels[0].pixels )
        p = { 0, 0, 0, 0 };
    levels[0].pixels[27] = { 1, 1, 1, 1 };
    auto desc            = descriptor();
    desc.maxMipLevel     = 1;
    Backing b;
    ASSERT_NO_FATAL_FAILURE( b.open( GetParam(), sourceFor( levels ), desc ) );
    ASSERT_NO_FATAL_FAILURE( warm( b ) );
    std::vector<CubicInput> inputs;
    for( double a : cubic_expected::fractions )
        for( double c : cubic_expected::fractions )
        {
            CubicInput in;
            in.nativeControl = 1;
            in.s             = float( ( 3.5 + a ) / 8 );
            in.t             = float( ( 3.5 + c ) / 8 );
            inputs.push_back( in );
        }
    CubicInput center;
    center.s = std::nextafter( .4375f, 0.f );
    inputs.push_back( center );
    center.s = std::nextafter( .4375f, 1.f );
    inputs.push_back( center );
    const auto out = batch( b, inputs );
    for( size_t i = 0; i < inputs.size(); ++i )
        check( out[i], ref::sample( levels, inputs[i].s, inputs[i].t, 0, true ), 8, 8 );
    EXPECT_NEAR( out[0].value.x, cubic_expected::impulseCenter, ref::valueTolerance( cubic_expected::impulseCenter ) );
    EXPECT_GT( std::abs( out[0].nativeValue.x - out[0].value.x ), 100 * ref::valueTolerance( cubic_expected::impulseCenter ) );
    std::cout << std::setprecision( 12 ) << "SENSITIVITY cubic=" << out[0].value.x << " bilinear=" << out[0].nativeValue.x
              << " tolerance=" << ref::valueTolerance( cubic_expected::impulseCenter ) << " separation_over_tolerance="
              << std::abs( out[0].nativeValue.x - out[0].value.x ) / ref::valueTolerance( cubic_expected::impulseCenter )
              << '\n';
    for( bool constant : { false, true } )
    {
        for( unsigned y = 0; y < 8; ++y )
            for( unsigned x = 0; x < 8; ++x )
                levels[0].pixels[8 * y + x] =
                    constant ? ref::Pixel{ .25, -.5, 2, .75 } :
                               ref::Pixel{ double( x ) / 8, double( y ) / 8, double( ( x + y ) % 2 ), double( x + y ) / 16 };
        Backing ramps;
        ASSERT_NO_FATAL_FAILURE( ramps.open( GetParam(), sourceFor( levels ), desc ) );
        ASSERT_NO_FATAL_FAILURE( warm( ramps ) );
        const auto pixels = batch( ramps, inputs );
        for( size_t i = 0; i < inputs.size(); ++i )
            check( pixels[i], ref::sample( levels, inputs[i].s, inputs[i].t, 0, true ), 8, 8 );
    }
}

TEST_P( CubicBackings, MipsDimensionsAddressJitter )
{
    AnisoEnvironment env( "0" );
    for( const auto& wh : cubic_expected::dimensions )
        for( bool linear : { false, true } )
        {
            auto levels = patterns( wh[0], wh[1] );
            for( auto au : { hipAddressModeWrap, hipAddressModeClamp, hipAddressModeMirror, hipAddressModeBorder } )
                for( auto av : { hipAddressModeWrap, hipAddressModeClamp, hipAddressModeMirror, hipAddressModeBorder } )
                {
                    auto desc           = descriptor( linear );
                    desc.addressMode[0] = au;
                    desc.addressMode[1] = av;
                    Backing b;
                    ASSERT_NO_FATAL_FAILURE( b.open( GetParam(), sourceFor( levels ), desc ) );
                    ASSERT_NO_FATAL_FAILURE( warm( b ) );
                    std::vector<CubicInput> inputs;
                    const float             last = float( levels.size() - 1 );
                    for( float lod : { -1.f, 0.f, .25f, .5f, .75f, 1.f, last, last + 1, std::nextafter( .5f, 0.f ),
                                       std::nextafter( .5f, 1.f ) } )
                        for( float s : { -.17f, .4375f, 1.23f } )
                        {
                            CubicInput in;
                            in.lod    = lod;
                            in.s      = s;
                            in.t      = 1 - s;
                            in.jitter = make_float2( -.25f, .375f );
                            inputs.push_back( in );
                        }
                    const auto out = batch( b, inputs );
                    for( size_t i = 0; i < inputs.size(); ++i )
                    {
                        SCOPED_TRACE( ::testing::Message() << wh[0] << 'x' << wh[1] << " linear=" << linear
                                                           << " address=" << au << ',' << av << " lod=" << inputs[i].lod );
                        const double s = double( inputs[i].s ) - .25 / wh[0], t = double( inputs[i].t ) + .375 / wh[1];
                        const auto   u = referenceAddress( au ), v = referenceAddress( av );
                        check( out[i], ref::sample( levels, s, t, inputs[i].lod, linear, u, v ), wh[0], wh[1] );
                        double lod = std::clamp( double( inputs[i].lod ), 0., double( last ) );
                        if( !linear )
                            lod = std::ceil( lod - .5 );
                        const unsigned low = unsigned( std::max( 0., lod ) );
                        EXPECT_EQ( out[i].decision.required.first, low );
                        EXPECT_EQ( out[i].decision.required.last, linear && lod > low ? low + 1 : low );
                    }
                }
        }
}

TEST_P( CubicBackings, SmartThresholdsGradientsAndEightOutputs )
{
    AnisoEnvironment env( "0" );
    auto             levels = patterns( 17, 9, 0, true );
    for( unsigned a : cubic_expected::anisotropy )
        for( bool conservative : { false, true } )
        {
            anisotropy_v1::Request request;
            request.maxAnisotropy = a;
            Backing b;
            ASSERT_NO_FATAL_FAILURE( b.open( GetParam(), sourceFor( levels ), descriptor( false ), request ) );
            ASSERT_NO_FATAL_FAILURE( warm( b ) );
            std::vector<CubicInput> inputs;
            for( float rho : { 0.f, .5f, 1.f, 1.5f, 2.f, 4.f, std::nextafter( 1.f, 0.f ), std::nextafter( 1.f, 2.f ),
                               std::nextafter( 2.f, 1.f ), std::nextafter( 2.f, 3.f ) } )
                for( unsigned mask = 0; mask < 8; ++mask )
                {
                    CubicInput in;
                    in.gradient             = 1;
                    in.outputs              = mask;
                    in.config.spatial       = cb::SpatialMode::SmartBicubic;
                    in.config.maxAnisotropy = float( a );
                    in.config.conservative  = conservative;
                    in.dx                   = make_float2( rho / 17, 0 );
                    in.dy                   = {};
                    inputs.push_back( in );
                }
            for( const auto& pair : std::vector<std::array<float, 4>>{ { 0, .1f, .2f, 0 },
                                                                       { -.1f, .2f, .3f, .1f },
                                                                       { .1f, .2f, -.1f, -.2f },
                                                                       { 0, 0, 0, 0 },
                                                                       { 100, 0, 0, .001f },
                                                                       { .2f, 0, 0, .01f } } )
            {
                CubicInput in;
                in.gradient             = 1;
                in.config.spatial       = cb::SpatialMode::SmartBicubic;
                in.config.maxAnisotropy = float( a );
                in.config.conservative  = conservative;
                in.dx                   = { pair[0], pair[1] };
                in.dy                   = { pair[2], pair[3] };
                inputs.push_back( in );
            }
            const auto out = batch( b, inputs );
            for( size_t i = 0; i < inputs.size(); ++i )
            {
                const auto& in = inputs[i];
                const auto  f  = ref::footprint( { in.dx.x, in.dx.y }, { in.dy.x, in.dy.y }, 17, 9, a, conservative );
                EXPECT_NEAR( out[i].weight, f.weight, 1e-12 );
                const double lod =
                    ( in.outputs & 6 ) ? std::clamp( std::ceil( f.lod - .5 ), 0., double( levels.size() - 1 ) ) : 0;
                EXPECT_EQ( out[i].low, unsigned( lod ) );
                EXPECT_EQ( out[i].high, unsigned( lod ) );
                EXPECT_EQ( out[i].decision.required.first, f.weight < 1 ? 0 : unsigned( lod ) );
                EXPECT_EQ( out[i].decision.required.last, f.weight < 1 ? levels.size() - 1 : unsigned( lod ) );
                ref::Result expected;
                expected.value = { .25, -.5, 2, .75 };
                check( out[i], expected, 17, 9, in.outputs );
            }
        }
}

TEST_P( CubicBackings, InvalidInputsNoRequestsAndColdDerivativeOnly )
{
    AnisoEnvironment env( "0" );
    Backing          b;
    ASSERT_NO_FATAL_FAILURE( b.open( GetParam(), sourceFor( patterns( 8, 8 ) ), descriptor() ) );
    auto                    ctx = b.prepare( stream_ );
    std::vector<CubicInput> inputs( 9 );
    for( auto& in : inputs )
        in.key = b.key;
    ++inputs[0].key.generation;
    inputs[1].revision = 2;
    inputs[2].key.incarnation++;
    inputs[3].s = std::numeric_limits<float>::quiet_NaN();
    inputs[4].config.abi.version++;
    inputs[5].config.abi.byteSize++;
    inputs[6].config.maxAnisotropy = 0;
    inputs[7].config.maxAnisotropy = std::numeric_limits<float>::infinity();
    inputs[8].config.spatial       = cb::SpatialMode::SmartBicubic;
    const auto invalid             = run( ctx, inputs );
    for( const auto& out : invalid )
    {
        EXPECT_EQ( out.decision.validity, cv::SampleValidity::Invalid );
        EXPECT_EQ( out.value.x, 0 );
        EXPECT_EQ( out.ds.x, 0 );
        EXPECT_EQ( out.dt.x, 0 );
    }
    uint32_t count = 999;
    ASSERT_EQ( hipMemcpy( &count, GetParam() ? ctx.whole.requestCount : ctx.legacy.requestCount, 4, hipMemcpyDeviceToHost ), hipSuccess );
    EXPECT_EQ( count, 0u );
    auto bad = ctx;
    bad.abi.version++;
    EXPECT_EQ( run( bad, { inputs[0] } )[0].decision.outcome, Outcome::AbiMismatch );
    ASSERT_EQ( b.process( stream_, ctx ), Outcome::Success );
    CubicInput in;
    in.outputs = 2;
    auto cold  = batch( b, { in } );
    EXPECT_EQ( cold[0].decision.validity, cv::SampleValidity::Missing );
    EXPECT_EQ( cold[0].ds.x, 0 );
    auto hot = batch( b, { in } );
    EXPECT_EQ( hot[0].decision.validity, cv::SampleValidity::Complete );
}

TEST_P( CubicBackings, PureGradientNonconstantOracleAndFiniteDifference )
{
    AnisoEnvironment env( "0" );
    auto             levels = patterns( 19, 19 );
    for( bool linear : { false, true } )
    {
        Backing b;
        ASSERT_NO_FATAL_FAILURE( b.open( GetParam(), sourceFor( levels ), descriptor( linear ) ) );
        ASSERT_NO_FATAL_FAILURE( warm( b ) );
        std::vector<CubicInput> inputs;
        for( unsigned a : cubic_expected::anisotropy )
            for( bool conservative : { false, true } )
                for( const auto& g : std::vector<std::array<float, 4>>{ { 0, 0, 0, 0 },
                                                                        { .1f, 0, 0, .02f },
                                                                        { -.1f, .04f, .02f, .08f },
                                                                        { .02f, .08f, -.1f, .04f },
                                                                        { 100, 0, 0, .001f },
                                                                        { 1e30f, 1e30f, -1e30f, 1e30f } } )
                {
                    CubicInput in;
                    in.gradient             = 1;
                    in.s                    = .413f;
                    in.t                    = .527f;
                    in.config.maxAnisotropy = float( a );
                    in.config.conservative  = conservative;
                    in.dx                   = { g[0], g[1] };
                    in.dy                   = { g[2], g[3] };
                    inputs.push_back( in );
                }
        auto out = batch( b, inputs );
        for( size_t i = 0; i < inputs.size(); ++i )
        {
            const auto& in = inputs[i];
            const auto  f = ref::footprint( { in.dx.x, in.dx.y }, { in.dy.x, in.dy.y }, 19, 19, in.config.maxAnisotropy,
                                            in.config.conservative != 0 );
            check( out[i], ref::sample( levels, in.s, in.t, f.lod, linear ), 19, 19 );
            EXPECT_NEAR( out[i].dx.x, f.dx.x, 1e-6 * ( 1 + std::abs( f.dx.x ) ) );
            EXPECT_NEAR( out[i].dy.y, f.dy.y, 1e-6 * ( 1 + std::abs( f.dy.y ) ) );
        }
        CubicInput center;
        center.s             = .413f;
        center.t             = .527f;
        CubicInput      left = center, right = center, down = center, up = center;
        constexpr float e = 1.f / 4096;
        left.s -= e;
        right.s += e;
        down.t -= e;
        up.t += e;
        out                 = batch( b, { center, left, right, down, up } );
        const auto expected = ref::sample( levels, center.s, center.t, 0, linear );
        check( out[0], expected, 19, 19 );
        // Fixed central-difference step; its cubic truncation error is bounded
        // independently by the sampled polynomial's third derivative.
        EXPECT_NEAR( ( out[2].value.x - out[1].value.x ) / ( 2 * e ), out[0].ds.x, .002 );
        EXPECT_NEAR( ( out[4].value.y - out[3].value.y ) / ( 2 * e ), out[0].dt.y, .002 );
    }
}

TEST_P( CubicBackings, FormatsChannelsAndImmutableSource )
{
    AnisoEnvironment env( "0" );
    for( unsigned f = 0; f < formats.size(); ++f )
        for( unsigned channels = 1; channels <= 4; ++channels )
        {
            auto       source = std::make_shared<TypedImageSource>( makeSource( f, channels ) );
            const auto before = source->pixels;
            ref::Level decoded{ 3, 2, {} };
            for( unsigned p = 0; p < 6; ++p )
            {
                ref::Pixel value{ 0, 0, 0, 1 };
                for( unsigned c = 0; c < channels; ++c )
                    value[c] = formats[f].expected[( p + c ) % 4];
                decoded.pixels.push_back( value );
            }
            auto desc        = descriptor();
            desc.maxMipLevel = 1;
            Backing b;
            ASSERT_NO_FATAL_FAILURE( b.open( GetParam(), source, desc ) );
            ASSERT_NO_FATAL_FAILURE( warm( b ) );
            std::vector<CubicInput> inputs;
            for( float s : { -.1f, .25f, .75f, 1.1f } )
            {
                CubicInput in;
                in.s = s;
                in.t = .37f;
                inputs.push_back( in );
            }
            const auto out = batch( b, inputs );
            for( size_t i = 0; i < inputs.size(); ++i )
            {
                SCOPED_TRACE( ::testing::Message() << "format=" << f << " channels=" << channels );
                check( out[i], ref::sample( { decoded }, inputs[i].s, inputs[i].t, 0, true ), 3, 2 );
            }
            EXPECT_EQ( source->pixels, before );
            EXPECT_EQ( source->baseColorReads, 0u );
            auto      context = b.prepare( stream_ );
            cb::Entry entry;
            ASSERT_EQ( hipMemcpy( &entry, context.entries + b.key.slot, sizeof( entry ), hipMemcpyDeviceToHost ), hipSuccess );
            hipResourceDesc resource{};
            ASSERT_EQ( hipGetTextureObjectResourceDesc( &resource, reinterpret_cast<hipTextureObject_t>( entry.points[0] ) ), hipSuccess );
            const size_t               pixelBytes = f == 0 ? 4 : 16;
            std::vector<unsigned char> actual( 6 * pixelBytes );
            ASSERT_EQ( hipMemcpy2DFromArray( actual.data(), 3 * pixelBytes, resource.res.array.array, 0, 0,
                                             3 * pixelBytes, 2, hipMemcpyDeviceToHost ),
                       hipSuccess );
            if( f == 0 )
            {
                std::vector<unsigned char> expected( 24, 0 );
                for( unsigned p = 0; p < 6; ++p )
                {
                    expected[4 * p + 3] = 255;
                    for( unsigned c = 0; c < channels; ++c )
                        expected[4 * p + c] = before[p * channels + c];
                }
                EXPECT_EQ( actual, expected );
            }
            else
            {
                std::vector<float> expected;
                for( const auto& p : decoded.pixels )
                    for( double c : p )
                        expected.push_back( float( c ) );
                EXPECT_EQ( 0, std::memcmp( actual.data(), expected.data(), actual.size() ) );
            }
            EXPECT_EQ( b.process( stream_, context ), Outcome::Success );
        }
}

TEST_P( CubicBackings, ColorConversionOnceAndIndependentAlpha )
{
    AnisoEnvironment env( "0" );
    for( bool srgb : { false, true } )
        for( bool bytes : { false, true } )
        {
            auto levels = patterns( 8, 8, 1 );
            for( unsigned i = 0; i < 64; ++i )
                levels[0].pixels[i] = { double( i % 4 ) / 3, double( ( i + 1 ) % 4 ) / 3, double( ( i + 2 ) % 4 ) / 3,
                                        double( i % 7 ) / 7 };
            auto source = sourceFor( levels );
            if( bytes )
            {
                source->info.format = HIP_AD_FORMAT_UNSIGNED_INT8;
                source->pixels.resize( 64 * 4 );
                for( unsigned i = 0; i < 64; ++i )
                    for( unsigned c = 0; c < 4; ++c )
                    {
                        const auto value          = uint8_t( std::lround( levels[0].pixels[i][c] * 255 ) );
                        source->pixels[4 * i + c] = value;
                        levels[0].pixels[i][c]    = double( value ) / 255;
                    }
                source->mipPixels = { source->pixels };
            }
            else
            {
                // Oracle uses the uploaded FLOAT representation, not ideal rationals.
                for( auto& p : levels[0].pixels )
                    for( auto& c : p )
                        c = float( c );
            }
            if( srgb )
                for( auto& p : levels[0].pixels )
                    for( unsigned c = 0; c < 3; ++c )
                        p[c] = p[c] <= .04045 ? p[c] / 12.92 : std::pow( ( p[c] + .055 ) / 1.055, 2.4 );
            auto desc        = descriptor();
            desc.sRGB        = srgb;
            desc.maxMipLevel = 1;
            Backing b;
            ASSERT_NO_FATAL_FAILURE( b.open( GetParam(), source, desc ) );
            ASSERT_NO_FATAL_FAILURE( warm( b ) );
            CubicInput in;
            in.s           = .413f;
            in.t           = .537f;
            const auto out = batch( b, { in } );
            SCOPED_TRACE( ::testing::Message() << "sRGB=" << srgb << " byte=" << bytes );
            check( out[0], ref::sample( levels, in.s, in.t, 0, true ), 8, 8 );
        }
}

TEST_P( CubicBackings, GeneratedAndLimitedTails )
{
    AnisoEnvironment env( "0" );
    for( const auto& wh : cubic_expected::dimensions )
        for( bool generate : { false, true } )
        {
            auto levels          = patterns( wh[0], wh[1], 0, true );
            auto source          = sourceFor( generate ? std::vector<ref::Level>{ levels[0] } : levels );
            auto desc            = descriptor();
            desc.generateMipmaps = generate;
            desc.maxMipLevel     = std::min( 3u, unsigned( levels.size() ) );
            levels.resize( desc.maxMipLevel );
            Backing b;
            ASSERT_NO_FATAL_FAILURE( b.open( GetParam(), source, desc ) );
            ASSERT_NO_FATAL_FAILURE( warm( b ) );
            std::vector<CubicInput> inputs;
            for( float lod : { 0.f, .25f, 1.f, 5.f } )
            {
                CubicInput in;
                in.lod = lod;
                inputs.push_back( in );
            }
            const auto out = batch( b, inputs );
            for( size_t i = 0; i < out.size(); ++i )
                check( out[i], ref::sample( levels, inputs[i].s, inputs[i].t, inputs[i].lod, true ), wh[0], wh[1] );
        }
}

TEST_P( CubicBackings, SourceUploadSamplerFailuresAndOverflow )
{
    AnisoEnvironment env( "0" );
    for( unsigned fault = 0; fault < 3; ++fault )
    {
        Faults faults;
        auto   source = sourceFor( patterns( 8, 8 ) );
        if( fault == 0 )
            source->failRead = true;
        Backing b;
        ASSERT_NO_FATAL_FAILURE( b.open( GetParam(), source, descriptor() ) );
        if( fault == 1 )
            faults.state->fail( internal::HipOperation::Upload, hipErrorOutOfMemory );
        if( fault == 2 )
            faults.state->fail( internal::HipOperation::CreateSampler, hipErrorOutOfMemory, 2 );
        auto       context = b.prepare( stream_ );
        CubicInput in;
        in.key          = b.key;
        const auto cold = run( context, { in } );
        ASSERT_EQ( cold[0].decision.validity, cv::SampleValidity::Missing );
        EXPECT_NE( b.process( stream_, context ), Outcome::Success );
        context           = b.prepare( stream_ );
        const auto failed = run( context, { in } );
        EXPECT_NE( failed[0].decision.validity, cv::SampleValidity::Complete );
        EXPECT_EQ( failed[0].value.x, 0 );
        EXPECT_EQ( failed[0].ds.x, 0 );
        EXPECT_EQ( b.process( stream_, context ),
                   GetParam() ? Outcome::Success : ( fault == 0 ? Outcome::SourceFailure : Outcome::DeviceOutOfMemory ) );
    }
    if( GetParam() )
    {
        Backing b;
        ASSERT_NO_FATAL_FAILURE( b.open( true, sourceFor( patterns( 8, 8 ) ), descriptor(), anisotropy_v1::Request::legacy(), 1 ) );
        auto       context = b.prepare( stream_ );
        CubicInput in;
        in.key         = b.key;
        in.lod         = .25f;
        const auto out = run( context, { in } );
        EXPECT_EQ( out[0].decision.outcome, Outcome::RequestOverflow );
        EXPECT_EQ( out[0].value.x, 0 );
        EXPECT_EQ( b.process( stream_, context ), Outcome::RequestOverflow );
    }
    else
    {
        Backing b;
        ASSERT_NO_FATAL_FAILURE( b.open( false, sourceFor( patterns( 8, 8 ) ), descriptor(), anisotropy_v1::Request::legacy(), 1 ) );
        auto     context = b.prepare( stream_ );
        uint32_t count   = 1;
        ASSERT_EQ( hipMemcpy( context.legacy.requestCount, &count, 4, hipMemcpyHostToDevice ), hipSuccess );
        CubicInput in;
        in.key         = b.key;
        const auto out = run( context, { in } );
        EXPECT_EQ( out[0].decision.outcome, Outcome::RequestOverflow );
        EXPECT_EQ( b.process( stream_, context ), Outcome::RequestOverflow );
    }
}

TEST_P( CubicBackings, ReversedSamplerOrderSharesStorageNotSemantics )
{
    AnisoEnvironment env( "0" );
    const auto       levels = patterns( 8, 8 );
    for( bool reverse : { false, true } )
    {
        auto source = sourceFor( levels );
        auto a = descriptor( false ), c = descriptor( true );
        c.addressMode[0] = hipAddressModeMirror;
        std::array<TextureDesc, 2> desc{ a, c };
        if( reverse )
            std::swap( desc[0], desc[1] );
        std::array<cv::GpuKey, 2> keys;
        Backing                   b;
        ASSERT_NO_FATAL_FAILURE( b.open( GetParam(), source, desc[0] ) );
        keys[0] = b.key;
        if( GetParam() )
        {
            auto registered = b.whole->addSampler( desc[1] );
            ASSERT_EQ( registered.outcome, Outcome::Success );
            keys[1] = registered.key;
            ASSERT_EQ( b.whole->enableCubicV1( keys[1] ), Outcome::Success );
        }
        else
        {
            auto handle = b.legacy->createTextureV1( source, desc[1] );
            ASSERT_TRUE( handle.valid );
            auto registered = b.legacy->enableCubicV1( handle.id );
            ASSERT_EQ( registered.outcome, Outcome::Success );
            keys[1] = registered.key;
        }
        EXPECT_FALSE( keys[0] == keys[1] );
        std::vector<CubicInput> inputs( 2 );
        for( unsigned i = 0; i < 2; ++i )
        {
            inputs[i].key = keys[i];
            inputs[i].s   = -.17f;
            inputs[i].lod = .75f;
        }
        auto ctx  = b.prepare( stream_ );
        auto cold = run( ctx, inputs );
        for( const auto& out : cold )
            EXPECT_EQ( out.decision.validity, cv::SampleValidity::Missing );
        EXPECT_EQ( b.process( stream_, ctx ), Outcome::Success );
        ctx                          = b.prepare( stream_ );
        auto                     hot = run( ctx, inputs );
        std::array<cb::Entry, 2> entries;
        ASSERT_EQ( hipMemcpy( entries.data(), ctx.entries, sizeof( entries ), hipMemcpyDeviceToHost ), hipSuccess );
        std::array<hipResourceDesc, 2> resources{};
        for( unsigned i = 0; i < 2; ++i )
        {
            check( hot[i],
                   ref::sample( levels, inputs[i].s, inputs[i].t, inputs[i].lod, desc[i].mipmapFilterMode == hipFilterModeLinear,
                                referenceAddress( desc[i].addressMode[0] ), ref::Clamp ),
                   8, 8 );
            ASSERT_EQ( hipGetTextureObjectResourceDesc( &resources[i],
                                                        reinterpret_cast<hipTextureObject_t>( entries[i].texture.textureObject ) ),
                       hipSuccess );
        }
        EXPECT_EQ( resources[0].res.mipmap.mipmap, resources[1].res.mipmap.mipmap );
        EXPECT_NE( entries[0].texture.textureObject, entries[1].texture.textureObject );
        EXPECT_EQ( b.process( stream_, ctx ), Outcome::Success );
        EXPECT_EQ( source->reads, levels.size() );
    }
}

TEST_P( CubicBackings, PureOptionalOutputsAndDefaultLodZero )
{
    AnisoEnvironment env( "0" );
    const auto       levels = patterns( 8, 8 );
    Backing          b;
    ASSERT_NO_FATAL_FAILURE( b.open( GetParam(), sourceFor( levels ), descriptor() ) );
    ASSERT_NO_FATAL_FAILURE( warm( b ) );
    std::vector<CubicInput> inputs;
    for( unsigned mode = 0; mode < 3; ++mode )
        for( unsigned mask = 0; mask < 8; ++mask )
        {
            CubicInput in;
            in.gradient = mode;
            in.outputs  = mask;
            in.lod      = .75f;
            in.dx       = { .25f, 0 };
            in.dy       = { 0, .25f };
            inputs.push_back( in );
        }
    const auto out = batch( b, inputs );
    for( size_t i = 0; i < inputs.size(); ++i )
    {
        const auto&  in  = inputs[i];
        const double lod = in.gradient == 2 ? 0 : in.gradient == 1 ? 1 : .75;
        check( out[i], ref::sample( levels, in.s, in.t, lod, true ), 8, 8, in.outputs );
        EXPECT_EQ( out[i].low, unsigned( lod ) );
    }
}

TEST_P( CubicBackings, FrozenSmartValueOnlyVersusDerivativeLevel )
{
    AnisoEnvironment env( "0" );
    namespace fixed = cubic_smart_expected;
    auto   levels   = patterns( fixed::width, fixed::height );
    size_t offset   = 0;
    for( auto& l : levels )
        for( auto& p : l.pixels )
        {
            const double v = fixed::pixels[offset++];
            p              = { v, v, v, v };
        }
    ASSERT_EQ( offset, fixed::pixels.size() );
    Backing b;
    ASSERT_NO_FATAL_FAILURE( b.open( GetParam(), sourceFor( levels ), descriptor( false ) ) );
    ASSERT_NO_FATAL_FAILURE( warm( b ) );
    std::vector<CubicInput> inputs;
    for( unsigned mask = 0; mask < 8; ++mask )
    {
        CubicInput in;
        in.gradient       = 1;
        in.outputs        = mask;
        in.s              = fixed::s;
        in.t              = fixed::t;
        in.config.spatial = cb::SpatialMode::SmartBicubic;
        in.dx             = { fixed::dxS, fixed::dxT };
        in.dy             = { fixed::dyS, fixed::dyT };
        inputs.push_back( in );
    }
    const auto out = batch( b, inputs );
    for( unsigned mask = 0; mask < 8; ++mask )
    {
        EXPECT_EQ( out[mask].low, fixed::cubicLevel[mask] );
        EXPECT_EQ( out[mask].weight, fixed::weight );
        EXPECT_EQ( out[mask].decision.required.first, fixed::firstRequired );
        EXPECT_EQ( out[mask].decision.required.last, fixed::lastRequired );
        ref::Result expected;
        expected.value.fill( fixed::value[mask] );
        check( out[mask], expected, 8, 8, mask );
    }
}

TEST_P( CubicBackings, NativeDependentSmartHalfTexelStencilAndPrimitiveControl )
{
    AnisoEnvironment env( "0" );
    const auto       levels         = patterns( 8, 8 );
    double           primitiveError = 0, compositionError = 0;
    for( bool linear : { false, true } )
    {
        Backing b;
        ASSERT_NO_FATAL_FAILURE( b.open( GetParam(), sourceFor( levels ), descriptor( linear ) ) );
        ASSERT_NO_FATAL_FAILURE( warm( b ) );
        for( float rho : { 1.25f, 1.5f, 1.75f, 2.f, 4.f } )
            for( unsigned mask = 0; mask < 8; ++mask )
            {
                auto      context = b.prepare( stream_ );
                cb::Entry entry;
                ASSERT_EQ( hipMemcpy( &entry, context.entries + b.key.slot, sizeof( entry ), hipMemcpyDeviceToHost ), hipSuccess );
                CubicInput in;
                in.key               = b.key;
                in.s                 = .4375f;
                in.t                 = .34375f;
                in.gradient          = 1;
                in.outputs           = mask;
                in.config.spatial    = cb::SpatialMode::SmartBicubic;
                in.dx                = { rho / 8, 0 };
                in.dy                = { 0, rho / 8 };
                const auto footprint = ref::footprint( { in.dx.x, in.dx.y }, { in.dy.x, in.dy.y }, 8, 8, 1, false );
                const auto oracle    = ref::smart( levels, in.s, in.t, footprint, linear, ( mask & 6 ) != 0 );
                std::vector<CubicInput> nativeInputs;
                for( const auto& uv : oracle.nativeCoordinates )
                {
                    auto n = in;
                    n.s    = float( uv[0] );
                    n.t    = float( uv[1] );
                    n.dx   = { float( footprint.dx.x ), float( footprint.dx.y ) };
                    n.dy   = { float( footprint.dy.x ), float( footprint.dy.y ) };
                    nativeInputs.push_back( n );
                }
                // Primitive outputs precede the candidate and are checked against
                // the independent native contract, never fed into expected smart.
                const auto                primitive = nativeRun( entry.texture.textureObject, nativeInputs );
                std::array<ref::Pixel, 5> observed;
                for( size_t n = 0; n < primitive.size(); ++n )
                {
                    const ref::Pixel got{ primitive[n].value.x, primitive[n].value.y, primitive[n].value.z,
                                          primitive[n].value.w };
                    observed[n] = got;
                    for( unsigned k = 0; k < 4; ++k )
                    {
                        primitiveError = std::max( primitiveError, std::abs( got[k] - oracle.nativeValues[n][k] ) );
                        EXPECT_NEAR( got[k], oracle.nativeValues[n][k], ref::valueTolerance( oracle.nativeValues[n][k] ) )
                            << "NATIVE_PRIMITIVE linear=" << linear << " rho=" << rho << " stencil=" << n << " channel=" << k;
                    }
                }
                const auto actual = run( context, { in } )[0];
                EXPECT_EQ( actual.decision.validity, cv::SampleValidity::Complete );
                const ref::Pixel v{ actual.value.x, actual.value.y, actual.value.z, actual.value.w };
                const ref::Pixel s{ actual.ds.x, actual.ds.y, actual.ds.z, actual.ds.w };
                const ref::Pixel t{ actual.dt.x, actual.dt.y, actual.dt.z, actual.dt.w };
                // This secondary diagnostic proves assembly of the independently
                // measured primitive. It does NOT replace the failing contract
                // assertions above or qualify native interpolation.
                const unsigned grid = actual.low;
                const double   w = levels[grid].width, h = levels[grid].height;
                const double   x = in.s * w - .5, y = in.t * h - .5, ix = std::floor( x ), iy = std::floor( y );
                const double u = 2 * ( x - ix - ( x - ix > .5 ? .5 : 0 ) ), vv = 2 * ( y - iy - ( y - iy > .5 ? .5 : 0 ) );
                const auto cubic = ref::reconstruct( levels[grid], in.s, in.t, ref::Clamp, ref::Clamp );
                for( unsigned k = 0; k < 4; ++k )
                {
                    if( mask & 1 )
                        EXPECT_NEAR( v[k], oracle.expected.value[k], ref::valueTolerance( oracle.expected.value[k] ) );
                    else
                        EXPECT_EQ( v[k], 0 );
                    if( mask & 2 )
                        EXPECT_NEAR( s[k], oracle.expected.ds[k], oracle.dsBound[k] );
                    else
                        EXPECT_EQ( s[k], 0 );
                    if( mask & 4 )
                        EXPECT_NEAR( t[k], oracle.expected.dt[k], oracle.dtBound[k] );
                    else
                        EXPECT_EQ( t[k], 0 );
                    const double ns =
                        2 * w * ( ( 1 - vv ) * ( observed[2][k] - observed[1][k] ) + vv * ( observed[4][k] - observed[3][k] ) );
                    const double nt =
                        2 * h * ( ( 1 - u ) * ( observed[3][k] - observed[1][k] ) + u * ( observed[4][k] - observed[2][k] ) );
                    const double c         = footprint.weight;
                    const double expectedV = c * cubic.value[k] + ( 1 - c ) * observed[0][k];
                    const double expectedS = c * cubic.ds[k] + ( 1 - c ) * ns;
                    const double expectedT = c * cubic.dt[k] + ( 1 - c ) * nt;
                    if( mask & 1 )
                    {
                        EXPECT_NEAR( v[k], expectedV, ref::valueTolerance( expectedV ) ) << "NATIVE_COMPOSITION";
                        compositionError = std::max( compositionError, std::abs( v[k] - expectedV ) );
                    }
                    if( mask & 2 )
                    {
                        EXPECT_NEAR( s[k] / 8, expectedS / 8, ref::derivativeTolerance( expectedS / 8 ) )
                            << "NATIVE_COMPOSITION";
                        compositionError = std::max( compositionError, std::abs( s[k] - expectedS ) / 8 );
                    }
                    if( mask & 4 )
                    {
                        EXPECT_NEAR( t[k] / 8, expectedT / 8, ref::derivativeTolerance( expectedT / 8 ) )
                            << "NATIVE_COMPOSITION";
                        compositionError = std::max( compositionError, std::abs( t[k] - expectedT ) / 8 );
                    }
                }
                EXPECT_EQ( b.process( stream_, context ), Outcome::Success );
            }
    }
    std::cout << std::setprecision( 12 ) << "NATIVE_PRIMITIVE_MAX_ERROR=" << primitiveError
              << " NATIVE_COMPOSITION_MAX_ERROR=" << compositionError << '\n';
}

TEST_P( CubicBackings, InactiveBranchesNeverFetchPoisonedHandles )
{
    AnisoEnvironment env( "0" );
    const auto       levels = patterns( 8, 8, 0, true );
    Backing          b;
    ASSERT_NO_FATAL_FAILURE( b.open( GetParam(), sourceFor( levels ), descriptor( false ) ) );
    ASSERT_NO_FATAL_FAILURE( warm( b ) );
    for( bool native : { false, true } )
    {
        auto      ctx = b.prepare( stream_ );
        cb::Entry entry;
        ASSERT_EQ( hipMemcpy( &entry, ctx.entries + b.key.slot, sizeof( entry ), hipMemcpyDeviceToHost ), hipSuccess );
        auto          poisoned = entry;
        wm::Entry     wholeOriginal{};
        TextureObject legacyOriginal = entry.texture.textureObject;
        if( native )
            std::fill( std::begin( poisoned.points ), std::end( poisoned.points ), TextureObject{ 1 } );
        else
        {
            poisoned.texture.textureObject = 1;
            if( GetParam() )
            {
                ASSERT_EQ( hipMemcpy( &wholeOriginal, ctx.whole.entries + b.key.slot, sizeof( wholeOriginal ), hipMemcpyDeviceToHost ),
                           hipSuccess );
                auto broken                  = wholeOriginal;
                broken.texture.textureObject = 1;
                ASSERT_EQ( hipMemcpy( const_cast<wm::Entry*>( ctx.whole.entries ) + b.key.slot, &broken,
                                      sizeof( broken ), hipMemcpyHostToDevice ),
                           hipSuccess );
            }
            else
            {
                TextureObject broken = 1;
                ASSERT_EQ( hipMemcpy( ctx.legacy.textures + b.key.slot, &broken, sizeof( broken ), hipMemcpyHostToDevice ), hipSuccess );
            }
        }
        ASSERT_EQ( hipMemcpy( const_cast<cb::Entry*>( ctx.entries ) + b.key.slot, &poisoned, sizeof( poisoned ), hipMemcpyHostToDevice ),
                   hipSuccess );
        CubicInput in;
        in.key            = b.key;
        in.gradient       = 1;
        in.config.spatial = cb::SpatialMode::SmartBicubic;
        if( native )
        {
            in.dx = { .5f, 0 };
            in.dy = { 0, .5f };
        }
        const auto  out = run( ctx, { in } )[0];
        ref::Result expected;
        expected.value = { .25, -.5, 2, .75 };
        check( out, expected, 8, 8 );
        ASSERT_EQ( hipMemcpy( const_cast<cb::Entry*>( ctx.entries ) + b.key.slot, &entry, sizeof( entry ), hipMemcpyHostToDevice ),
                   hipSuccess );
        if( !native )
        {
            if( GetParam() )
                ASSERT_EQ( hipMemcpy( const_cast<wm::Entry*>( ctx.whole.entries ) + b.key.slot, &wholeOriginal,
                                      sizeof( wholeOriginal ), hipMemcpyHostToDevice ),
                           hipSuccess );
            else
                ASSERT_EQ( hipMemcpy( ctx.legacy.textures + b.key.slot, &legacyOriginal, sizeof( legacyOriginal ), hipMemcpyHostToDevice ),
                           hipSuccess );
        }
        EXPECT_EQ( b.process( stream_, ctx ), Outcome::Success );
    }
}

TEST_P( CubicBackings, PendingConsumerIsFencedBeforeViewRetirement )
{
    AnisoEnvironment env( "0" );
    const auto       levels = patterns( 17, 9 );
    Backing          b;
    ASSERT_NO_FATAL_FAILURE( b.open( GetParam(), sourceFor( levels ), descriptor() ) );
    ASSERT_NO_FATAL_FAILURE( warm( b ) );
    auto       ctx = b.prepare( stream_ );
    CubicInput in;
    in.key         = b.key;
    in.lod         = .25f;
    const auto out = run( ctx, { in }, [&] {
        if( GetParam() )
            EXPECT_EQ( b.whole->resize( 2 ), Outcome::Success );
        else
            b.legacy->unloadTexture( b.id );
    } );
    check( out[0], ref::sample( levels, in.s, in.t, in.lod, true ), 17, 9 );
    if( GetParam() )
    {
        wm::Status status;
        ASSERT_EQ( b.whole->getStatus( status ), Outcome::Success );
        EXPECT_EQ( status.mips.firstResidentMip, 2u );
        EXPECT_EQ( status.mips.resourceWidth, 4u );
        EXPECT_EQ( status.mips.resourceHeight, 2u );
        std::cout << "Suffix allocation original=17x9 firstOriginal=2 physical=0 allocation=" << status.mips.resourceWidth
                  << 'x' << status.mips.resourceHeight << '\n';
    }
    else
        EXPECT_EQ( b.legacy->getResidentTextureCount(), 0u );
}

TEST_F( CubicSampling, SuffixGrowTrimStrictUnionAndFailedGrowth )
{
    AnisoEnvironment env( "0" );
    Faults           faults;
    const auto       levels = patterns( 17, 9 );
    Backing          b;
    ASSERT_NO_FATAL_FAILURE( b.open( true, sourceFor( levels ), descriptor() ) );
    ASSERT_EQ( b.whole->resize( 2 ), Outcome::Success );
    CubicInput in;
    in.lod   = 2.25f;
    auto out = batch( b, { in } );
    check( out[0], ref::sample( levels, in.s, in.t, in.lod, true ), 17, 9 );
    ASSERT_EQ( b.whole->resize( 4 ), Outcome::Success );
    in.lod = 4;
    out    = batch( b, { in } );
    check( out[0], ref::sample( levels, in.s, in.t, in.lod, true ), 17, 9 );
    ASSERT_EQ( b.whole->resize( 2 ), Outcome::Success );
    auto ctx  = b.prepare( stream_ );
    in.key    = b.key;
    in.lod    = 1.25f;
    auto miss = run( ctx, { in } );
    EXPECT_EQ( miss[0].decision.validity, cv::SampleValidity::Missing );
    EXPECT_EQ( miss[0].decision.required.first, 1u );
    EXPECT_EQ( miss[0].decision.required.last, 2u );
    uint32_t count = 0;
    ASSERT_EQ( hipMemcpy( &count, ctx.whole.requestCount, 4, hipMemcpyDeviceToHost ), hipSuccess );
    ASSERT_EQ( count, 2u );
    std::vector<cv::RequestKey> requests( count );
    ASSERT_EQ( hipMemcpy( requests.data(), ctx.whole.requests, count * sizeof( cv::RequestKey ), hipMemcpyDeviceToHost ), hipSuccess );
    EXPECT_EQ( requests[0].originalMip, 1u );
    EXPECT_EQ( requests[1].originalMip, 2u );
    faults.state->fail( internal::HipOperation::CreateSampler, hipErrorOutOfMemory, 2 );
    EXPECT_EQ( b.process( stream_, ctx ), Outcome::DeviceOutOfMemory );
    wm::Status status;
    ASSERT_EQ( b.whole->getStatus( status ), Outcome::Success );
    EXPECT_EQ( status.mips.firstResidentMip, 2u );
    in.lod = 2.25f;
    out    = batch( b, { in } );
    check( out[0], ref::sample( levels, in.s, in.t, in.lod, true ), 17, 9 );
    in.gradient       = 1;
    in.config.spatial = cb::SpatialMode::SmartBicubic;
    in.dx             = { 1.5f / 17, 0 };
    out               = batch( b, { in } );
    EXPECT_EQ( out[0].decision.validity, cv::SampleValidity::Missing );
    EXPECT_EQ( out[0].decision.required.first, 0u );
    EXPECT_EQ( out[0].decision.required.last, 4u );
    ASSERT_EQ( b.whole->getStatus( status ), Outcome::Success );
    EXPECT_EQ( status.mips.firstResidentMip, 0u );
    ASSERT_EQ( b.whole->resize( 2 ), Outcome::Success );
    // Fail the added cubic-table copy after the native inactive table copied.
    faults.state->fail( internal::HipOperation::PublishMappings, hipErrorOutOfMemory, 2 );
    EXPECT_EQ( b.whole->resize( 0 ), Outcome::DeviceOutOfMemory );
    ASSERT_EQ( b.whole->getStatus( status ), Outcome::Success );
    EXPECT_EQ( status.mips.firstResidentMip, 2u );
    EXPECT_EQ( status.pendingBytes, 0u );
    EXPECT_EQ( status.retiringBytes, 0u );
    in     = {};
    in.lod = 2.25f;
    out    = batch( b, { in } );
    check( out[0], ref::sample( levels, in.s, in.t, in.lod, true ), 17, 9 );
}

TEST_F( CubicSampling, OverrideValidationIdentityAndQualification )
{
    for( const char* value : { static_cast<const char*>( nullptr ), "0", "", "true", "1" } )
    {
        AnisoEnvironment env( value );
        EXPECT_EQ( internal::nativeAnisotropy( 8 ), value && std::string( value ) == "1" ? 8u : 0u );
        EXPECT_EQ( internal::nativeAnisotropy( 0 ), 0u );
    }
    AnisoEnvironment env( "0" );
    for( bool suffix : { false, true } )
        for( unsigned a : cubic_expected::anisotropy )
        {
            anisotropy_v1::Request request;
            request.maxAnisotropy = a;
            Backing b;
            ASSERT_NO_FATAL_FAILURE( b.open( suffix, sourceFor( patterns( 8, 8 ) ), descriptor(), request ) );
            ASSERT_NO_FATAL_FAILURE( warm( b ) );
            anisotropy_v1::Status status;
            ASSERT_EQ( suffix ? b.whole->getAnisotropyStatusV1( b.key, status ) :
                                b.legacy->getTextureAnisotropyStatusV1( b.id, status ),
                       Outcome::Success );
            EXPECT_EQ( status.requested.maxAnisotropy, a );
            EXPECT_EQ( status.texture.submittedSampler.maxAnisotropy, 0u );
            EXPECT_EQ( status.texture.returnedSampler.maxAnisotropy, 0u );
            EXPECT_EQ( status.qualification, anisotropy_v1::Qualification::Unqualified );
        }
    DemandTextureLoader    loader( loaderOptions() );
    auto                   source = sourceFor( patterns( 8, 8 ) );
    anisotropy_v1::Request bad;
    bad.maxAnisotropy = 17;
    EXPECT_FALSE( loader.createTextureAnisotropyV1( source, descriptor(), bad ).valid );
    auto strict = anisotropy_v1::Request::parity();
    auto h      = loader.createTextureAnisotropyV1( source, descriptor(), strict );
    ASSERT_TRUE( h.valid );
    auto key = loader.enableCubicV1( h.id );
    ASSERT_EQ( key.outcome, Outcome::Success );
    loader.launchPrepare( stream_ );
    CubicInput strictInput;
    strictInput.key = key.key;
    auto ctx        = loader.getCubicContextV1();
    EXPECT_EQ( run( ctx, { strictInput } )[0].decision.validity, cv::SampleValidity::Missing );
    EXPECT_EQ( loader.processRequests( stream_, ctx.legacy ), 0u );
    anisotropy_v1::Status strictStatus;
    ASSERT_EQ( loader.getTextureAnisotropyStatusV1( h.id, strictStatus ), Outcome::Success );
    EXPECT_EQ( strictStatus.requirementRejected, 1u );
    EXPECT_EQ( strictStatus.texture.primary.outcome, Outcome::Unsupported );
    wm::Texture whole( source );
    EXPECT_EQ( whole.addSampler( descriptor(), cv::SamplingPolicy::Strict, bad ).outcome, Outcome::InvalidInput );
    EXPECT_EQ( whole.addSampler( descriptor(), cv::SamplingPolicy::Strict, strict ).outcome, Outcome::Unsupported );
}

TEST_F( CubicSampling, FailedOptOutGrowthPreservesPublishedSamplerStatus )
{
    AnisoEnvironment       env( "0" );
    Faults                 faults;
    const auto             levels = patterns( 8, 8 );
    anisotropy_v1::Request request;
    request.maxAnisotropy = 8;
    Backing b;
    ASSERT_NO_FATAL_FAILURE( b.open( true, sourceFor( levels ), descriptor(), request ) );
    ASSERT_EQ( b.whole->resize( 2 ), Outcome::Success );
    {
        AnisoEnvironment optOut( "1" );
        faults.state->fail( internal::HipOperation::CreateSampler, hipErrorNotSupported );
        EXPECT_EQ( b.whole->resize( 0 ), Outcome::Unsupported );
    }
    anisotropy_v1::Status status;
    ASSERT_EQ( b.whole->getAnisotropyStatusV1( b.key, status ), Outcome::Success );
    EXPECT_EQ( status.requested.maxAnisotropy, 8u );
    EXPECT_EQ( status.texture.firstResidentMip, 2u );
    EXPECT_EQ( status.texture.submittedSampler.maxAnisotropy, 0u );
    EXPECT_EQ( status.texture.returnedSampler.maxAnisotropy, 0u );
    EXPECT_EQ( status.texture.published, 1u );
    EXPECT_EQ( status.texture.primary.outcome, Outcome::Unsupported );
    CubicInput in;
    in.lod         = 2.25f;
    const auto out = batch( b, { in } );
    check( out[0], ref::sample( levels, in.s, in.t, in.lod, true ), 8, 8 );
}

TEST_F( CubicSampling, DeferredFilenameAppearsBeforeDemandProcessing )
{
    AnisoEnvironment env( "0" );
    const auto       directory = testFileRoot() / "cubic-deferred-appears";
    ASSERT_FALSE( std::filesystem::exists( directory ) );
    ASSERT_TRUE( std::filesystem::create_directory( directory ) );
    struct Cleanup
    {
        std::filesystem::path path;
        ~Cleanup()
        {
            std::error_code error;
            std::filesystem::remove_all( path, error );
            EXPECT_FALSE( error );
        }
    } cleanup{ directory };
    const auto          file = directory / "image.ppm";
    DemandTextureLoader loader( loaderOptions() );
    auto                desc = descriptor();
    desc.maxMipLevel         = 1;
    const auto handle        = loader.createTexture( file.string(), desc );
    ASSERT_TRUE( handle.valid );
    EXPECT_EQ( handle.width, 0 );
    const auto key = loader.enableCubicV1( handle.id );
    ASSERT_EQ( key.outcome, Outcome::Success );
    const std::vector<unsigned char> pixels( 8 * 8 * 3, 128 );
    {
        std::ofstream out( file, std::ios::binary );
        out << "P6\n8 8\n255\n";
        out.write( reinterpret_cast<const char*>( pixels.data() ), pixels.size() );
        ASSERT_TRUE( out.good() );
    }
    loader.launchPrepare( stream_ );
    auto       context = loader.getCubicContextV1();
    CubicInput invalid;
    invalid.key = key.key;
    ++invalid.key.generation;
    EXPECT_EQ( run( context, { invalid } )[0].decision.outcome, Outcome::InvalidKey );
    uint32_t requests = 99;
    ASSERT_EQ( hipMemcpy( &requests, context.legacy.requestCount, sizeof( requests ), hipMemcpyDeviceToHost ), hipSuccess );
    EXPECT_EQ( requests, 0u );
    std::vector<CubicInput> inputs( 8 );
    for( unsigned mask = 0; mask < 8; ++mask )
    {
        inputs[mask].key     = key.key;
        inputs[mask].outputs = mask;
    }
    const auto cold = run( context, inputs );
    for( const auto& out : cold )
    {
        EXPECT_EQ( out.decision.outcome, Outcome::Pending );
        EXPECT_EQ( out.decision.validity, cv::SampleValidity::Missing );
        EXPECT_EQ( out.decision.required.first, cv::InvalidSlot );
        EXPECT_EQ( out.value.x, 0 );
        EXPECT_EQ( out.ds.x, 0 );
        EXPECT_EQ( out.dt.x, 0 );
    }
    ASSERT_EQ( hipMemcpy( &requests, context.legacy.requestCount, sizeof( requests ), hipMemcpyDeviceToHost ), hipSuccess );
    EXPECT_EQ( requests, 1u );
    ASSERT_EQ( loader.processRequests( stream_, context.legacy ), 1u );
    loader.launchPrepare( stream_ );
    context = loader.getCubicContextV1();
    cb::Entry entry;
    ASSERT_EQ( hipMemcpy( &entry, context.entries + handle.id, sizeof( entry ), hipMemcpyDeviceToHost ), hipSuccess );
    EXPECT_EQ( entry.texture.mips.originalWidth, 8u );
    EXPECT_EQ( entry.texture.mips.originalHeight, 8u );
    EXPECT_EQ( entry.texture.mips.originalLevels, 1u );
    EXPECT_NE( entry.points[0], 0u );
    const auto  hot = run( context, inputs );
    ref::Result expected;
    expected.value = { 128. / 255, 128. / 255, 128. / 255, 1 };
    for( unsigned mask = 0; mask < 8; ++mask )
        check( hot[mask], expected, 8, 8, mask );
}

TEST_F( CubicSampling, DeferredFilenameReadFailureIsTerminalSourceError )
{
    AnisoEnvironment env( "0" );
    const auto       directory = testFileRoot() / "cubic-deferred-missing";
    ASSERT_FALSE( std::filesystem::exists( directory ) );
    ASSERT_TRUE( std::filesystem::create_directory( directory ) );
    struct Cleanup
    {
        std::filesystem::path path;
        ~Cleanup()
        {
            std::error_code error;
            std::filesystem::remove_all( path, error );
            EXPECT_FALSE( error );
        }
    } cleanup{ directory };
    DemandTextureLoader loader( loaderOptions() );
    auto                desc = descriptor();
    desc.maxMipLevel         = 1;
    const auto handle        = loader.createTexture( ( directory / "missing.ppm" ).string(), desc );
    ASSERT_TRUE( handle.valid );
    const auto key = loader.enableCubicV1( handle.id );
    ASSERT_EQ( key.outcome, Outcome::Success );
    CubicInput in;
    in.key = key.key;
    loader.launchPrepare( stream_ );
    auto       context = loader.getCubicContextV1();
    const auto cold    = run( context, { in } )[0];
    EXPECT_EQ( cold.decision.validity, cv::SampleValidity::Missing );
    EXPECT_EQ( cold.value.x, 0 );
    EXPECT_EQ( cold.ds.x, 0 );
    EXPECT_EQ( cold.dt.x, 0 );
    EXPECT_EQ( loader.processRequests( stream_, context.legacy ), 0u );
    capability_v1::Status status;
    ASSERT_EQ( loader.getTextureStatusV1( handle.id, status ), Outcome::Success );
    EXPECT_GT( status.attempts, 0u );
    EXPECT_EQ( status.state, capability_v1::State::Failed );
    EXPECT_EQ( status.primary.outcome, Outcome::SourceFailure );
    EXPECT_EQ( status.primary.operation, capability_v1::Operation::SourceRead );
    EXPECT_EQ( loader.getLastError(), LoaderError::ImageLoadFailed );
    loader.launchPrepare( stream_ );
    context           = loader.getCubicContextV1();
    const auto failed = run( context, { in } )[0];
    EXPECT_EQ( failed.decision.outcome, Outcome::SourceFailure );
    EXPECT_EQ( failed.decision.validity, cv::SampleValidity::Invalid );
    EXPECT_EQ( failed.value.x, 0 );
    EXPECT_EQ( failed.ds.x, 0 );
    EXPECT_EQ( failed.dt.x, 0 );
    uint32_t requests = 99;
    ASSERT_EQ( hipMemcpy( &requests, context.legacy.requestCount, sizeof( requests ), hipMemcpyDeviceToHost ), hipSuccess );
    EXPECT_EQ( requests, 0u );
}

TEST_F( CubicSampling, MalformedZeroDimensionIsNotDeferredMetadata )
{
    AnisoEnvironment env( "0" );
    Backing          b;
    ASSERT_NO_FATAL_FAILURE( b.open( false, sourceFor( patterns( 8, 8 ) ), descriptor() ) );
    ASSERT_NO_FATAL_FAILURE( warm( b ) );
    auto      context = b.prepare( stream_ );
    cb::Entry original;
    ASSERT_EQ( hipMemcpy( &original, context.entries, sizeof( original ), hipMemcpyDeviceToHost ), hipSuccess );
    CubicInput in;
    in.key = b.key;
    for( bool pending : { false, true } )
    {
        auto malformed         = original;
        malformed.texture.mips = {};
        if( pending )
        {
            malformed.texture.residency     = Outcome::Pending;
            malformed.texture.textureObject = 0;
            std::fill( std::begin( malformed.points ), std::end( malformed.points ), TextureObject{ 0 } );
        }
        ASSERT_EQ( hipMemcpy( const_cast<cb::Entry*>( context.entries ), &malformed, sizeof( malformed ), hipMemcpyHostToDevice ),
                   hipSuccess );
        const auto out = run( context, { in } )[0];
        EXPECT_EQ( out.decision.outcome, Outcome::InvalidInput );
        EXPECT_EQ( out.decision.validity, cv::SampleValidity::Invalid );
        EXPECT_EQ( out.value.x, 0 );
        EXPECT_EQ( out.ds.x, 0 );
        EXPECT_EQ( out.dt.x, 0 );
    }
    uint32_t requests = 99;
    ASSERT_EQ( hipMemcpy( &requests, context.legacy.requestCount, sizeof( requests ), hipMemcpyDeviceToHost ), hipSuccess );
    EXPECT_EQ( requests, 0u );
    ASSERT_EQ( hipMemcpy( const_cast<cb::Entry*>( context.entries ), &original, sizeof( original ), hipMemcpyHostToDevice ), hipSuccess );
    EXPECT_EQ( b.process( stream_, context ), Outcome::Success );
}

TEST_F( CubicSampling, AllPublicRegistrationRoutesUseOverride )
{
    AnisoEnvironment env( nullptr );
    const auto       file = testFileRoot() / "cubic-native-override-fixture.ppm";
    ASSERT_FALSE( std::filesystem::exists( file ) );
    struct Cleanup
    {
        std::filesystem::path path;
        ~Cleanup()
        {
            std::error_code error;
            std::filesystem::remove( path, error );
            EXPECT_FALSE( error );
        }
    } cleanup{ file };
    const std::vector<unsigned char> pixels( 8 * 8 * 3, 128 );
    {
        std::ofstream out( file, std::ios::binary );
        out << "P6\n8 8\n255\n";
        out.write( reinterpret_cast<const char*>( pixels.data() ), pixels.size() );
        ASSERT_TRUE( out.good() );
    }
    auto source        = std::make_shared<TypedImageSource>();
    source->info       = {};
    source->info.width = source->info.height = 8;
    source->info.numChannels                 = 3;
    source->info.numMipLevels                = 1;
    source->info.format                      = HIP_AD_FORMAT_UNSIGNED_INT8;
    source->info.isValid                     = true;
    source->pixels                           = pixels;
    DemandTextureLoader loader( loaderOptions() );
    auto                desc = descriptor();
    desc.maxMipLevel         = 1;
    anisotropy_v1::Request request;
    request.maxAnisotropy = 8;
    std::array<TextureHandle, 3> handles{ loader.createTextureAnisotropyV1( file.string(), desc, request ),
                                          loader.createTextureAnisotropyV1( source, desc, request ),
                                          loader.createTextureFromMemoryAnisotropyV1( pixels.data(), 8, 8, 3, desc, request ) };
    std::vector<CubicInput>      inputs;
    for( auto h : handles )
    {
        ASSERT_TRUE( h.valid );
        auto key = loader.enableCubicV1( h.id );
        ASSERT_EQ( key.outcome, Outcome::Success );
        CubicInput in;
        in.key = key.key;
        inputs.push_back( in );
    }
    loader.launchPrepare( stream_ );
    auto ctx = loader.getCubicContextV1();
    run( ctx, inputs );
    EXPECT_EQ( loader.processRequests( stream_, ctx.legacy ), 3u );
    loader.launchPrepare( stream_ );
    ctx            = loader.getCubicContextV1();
    const auto out = run( ctx, inputs );
    for( unsigned i = 0; i < 3; ++i )
    {
        anisotropy_v1::Status status;
        ASSERT_EQ( loader.getTextureAnisotropyStatusV1( handles[i].id, status ), Outcome::Success );
        EXPECT_EQ( status.requested.maxAnisotropy, 8u );
        EXPECT_EQ( status.texture.submittedSampler.maxAnisotropy, 0u );
        ref::Result expected;
        expected.value = { 128. / 255, 128. / 255, 128. / 255, 1 };
        check( out[i], expected, 8, 8 );
    }
}

class CubicNativeOptOut : public CubicSampling, public ::testing::WithParamInterface<std::tuple<bool, unsigned, bool>>
{
};
TEST_P( CubicNativeOptOut, RequestedAnisotropyRemainsNativeQualification )
{
    AnisoEnvironment       env( "1" );
    const bool             suffix    = std::get<0>( GetParam() );
    const unsigned         a         = std::get<1>( GetParam() );
    const bool             singleton = std::get<2>( GetParam() );
    anisotropy_v1::Request request;
    request.maxAnisotropy = a;
    Backing b;
    ASSERT_NO_FATAL_FAILURE( b.open( suffix, sourceFor( patterns( singleton ? 1 : 8, singleton ? 1 : 8, 0, true ) ),
                                     descriptor(), request ) );
    CubicInput in;
    in.gradient             = 1;
    in.config.spatial       = cb::SpatialMode::SmartBicubic;
    in.config.maxAnisotropy = float( a );
    in.dx                   = { .5f, 0 };
    in.dy                   = { 0, .5f };
    in.key                  = b.key;
    auto ctx                = b.prepare( stream_ );
    auto cold               = run( ctx, { in } );
    ASSERT_EQ( cold[0].decision.validity, cv::SampleValidity::Missing );
    // A native 801 is intentionally an ordinary failed test, not skipped.
    EXPECT_EQ( b.process( stream_, ctx ), Outcome::Success );
    anisotropy_v1::Status status;
    ASSERT_EQ( suffix ? b.whole->getAnisotropyStatusV1( b.key, status ) : b.legacy->getTextureAnisotropyStatusV1( b.id, status ),
               Outcome::Success );
    EXPECT_EQ( status.requested.maxAnisotropy, a );
    if( status.texture.submitted )
        EXPECT_EQ( status.texture.submittedSampler.maxAnisotropy, a );
    if( status.texture.returned )
        EXPECT_EQ( status.texture.returnedSampler.maxAnisotropy, a );
    std::cout << "NATIVE_REQUEST requested=" << a << " submitted_present=" << status.texture.submitted
              << " submitted=" << status.texture.submittedSampler.maxAnisotropy
              << " returned_present=" << status.texture.returned << " returned=" << status.texture.returnedSampler.maxAnisotropy
              << " native_error=" << status.texture.primary.rawHipError << '\n';
    auto        hot = batch( b, { in } );
    ref::Result expected;
    expected.value = { .25, -.5, 2, .75 };
    check( hot[0], expected, 8, 8 );
}
INSTANTIATE_TEST_SUITE_P( Both, CubicBackings, ::testing::Values( false, true ) );
INSTANTIATE_TEST_SUITE_P( AllRequired,
                          CubicNativeOptOut,
                          ::testing::Combine( ::testing::Values( false, true ),
                                              ::testing::Values( 1u, 2u, 4u, 8u, 16u ),
                                              ::testing::Values( false, true ) ) );
}  // namespace
}  // namespace test
}  // namespace hip_demand
