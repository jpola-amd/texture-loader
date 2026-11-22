// SPDX-License-Identifier: MIT
// Tests for the VMM sampler (VmmTextureSampling.h): its results against a native HIP texture of the same content,
// the requests it records for missing resources, and UDIM grids.

#include "VmmLoaderTestUtils.h"
#include "VmmSamplerTestTypes.h"
#include <cmath>
#include <filesystem>
#include <string>

#if defined( _WIN32 )
#define NOMINMAX
#include <Windows.h>
#endif

namespace hip_demand::vmm::test {

namespace {

std::filesystem::path executableDirectory()
{
#if defined( _WIN32 )
    std::vector<wchar_t> path( 32768 );
    const DWORD          length = GetModuleFileNameW( nullptr, path.data(), static_cast<DWORD>( path.size() ) );
    return std::filesystem::path( path.data(), path.data() + length ).parent_path();
#else
    return std::filesystem::read_symlink( "/proc/self/exe" ).parent_path();
#endif
}

// A native mipmapped RGBA8 texture with the content of a 4-channel TestImage, sampled with normalized floats.
class NativeTexture
{
  public:
    NativeTexture( TestImage& image, hipTextureAddressMode addressMode )
    {
        const TextureInfo& info = image.getInfo();
        EXPECT_EQ( info.numChannels, 4u );
        const hipChannelFormatDesc channelDesc = hipCreateChannelDesc<uchar4>();
        EXPECT_EQ( hipMallocMipmappedArray( &array_, &channelDesc, make_hipExtent( info.width, info.height, 0 ), info.numMipLevels, 0 ), hipSuccess );
        for( uint32_t level = 0; level < info.numMipLevels; ++level )
        {
            const uint32_t       width  = std::max( info.width >> level, 1u );
            const uint32_t       height = std::max( info.height >> level, 1u );
            std::vector<uint8_t> texels( static_cast<size_t>( width ) * height * 4 );
            EXPECT_TRUE( image.readMipLevel( reinterpret_cast<char*>( texels.data() ), level, width, height, nullptr ) );
            hipArray_t levelArray = nullptr;
            EXPECT_EQ( hipGetMipmappedArrayLevel( &levelArray, array_, level ), hipSuccess );
            EXPECT_EQ( hipMemcpy2DToArray( levelArray, 0, 0, texels.data(), width * 4, width * 4, height, hipMemcpyHostToDevice ), hipSuccess );
        }

        hipResourceDesc resourceDesc{};
        resourceDesc.resType           = hipResourceTypeMipmappedArray;
        resourceDesc.res.mipmap.mipmap = array_;
        hipTextureDesc textureDesc{};
        textureDesc.addressMode[0]      = addressMode;
        textureDesc.addressMode[1]      = addressMode;
        textureDesc.filterMode          = hipFilterModeLinear;
        textureDesc.mipmapFilterMode    = hipFilterModeLinear;
        textureDesc.readMode            = hipReadModeNormalizedFloat;
        textureDesc.normalizedCoords    = 1;
        textureDesc.minMipmapLevelClamp = 0.0f;
        textureDesc.maxMipmapLevelClamp = static_cast<float>( info.numMipLevels - 1 );
        EXPECT_EQ( hipCreateTextureObject( &texture_, &resourceDesc, &textureDesc, nullptr ), hipSuccess );
    }

    ~NativeTexture()
    {
        if( texture_ )
            hipDestroyTextureObject( texture_ );
        if( array_ )
            hipFreeMipmappedArray( array_ );
    }

    hipTextureObject_t get() const { return texture_; }

  private:
    hipMipmappedArray_t array_   = nullptr;
    hipTextureObject_t  texture_ = 0;
};

float maxChannelDifference( const float4& a, const float4& b )
{
    return std::max( std::max( std::fabs( a.x - b.x ), std::fabs( a.y - b.y ) ), std::max( std::fabs( a.z - b.z ), std::fabs( a.w - b.w ) ) );
}

// A grid of points over [low, high)^2 and the given lods, with isotropic gradients of 2^lod texels.
std::vector<SamplePoint> gridPoints( uint32_t perAxis, float low, float high, const std::vector<float>& lods, uint32_t textureSize )
{
    std::vector<SamplePoint> points;
    for( float lod : lods )
    {
        const float texels = std::exp2( lod ) / static_cast<float>( textureSize );
        for( uint32_t j = 0; j < perAxis; ++j )
        {
            for( uint32_t i = 0; i < perAxis; ++i )
            {
                SamplePoint point;
                point.x   = low + ( high - low ) * ( static_cast<float>( i ) + 0.37f ) / static_cast<float>( perAxis );
                point.y   = low + ( high - low ) * ( static_cast<float>( j ) + 0.61f ) / static_cast<float>( perAxis );
                point.lod = lod;
                point.ddx = { texels, 0.0f };
                point.ddy = { 0.0f, texels };
                points.push_back( point );
            }
        }
    }
    return points;
}

class VmmSamplerTest : public VmmLoaderTest
{
  protected:
    void SetUp() override
    {
        VmmLoaderTest::SetUp();
        if( IsSkipped() )
            return;
        const std::filesystem::path path = executableDirectory() / "vmm_sampler_test_kernel.co";
        if( !std::filesystem::exists( path ) )
            GTEST_SKIP() << "Sampler test kernel not found: " << path;
        ASSERT_EQ( hipModuleLoad( &module_, path.string().c_str() ), hipSuccess );
        ASSERT_EQ( hipModuleGetFunction( &vmmLod_, module_, "sampleVmmLod" ), hipSuccess );
        ASSERT_EQ( hipModuleGetFunction( &vmmGrad_, module_, "sampleVmmGrad" ), hipSuccess );
        ASSERT_EQ( hipModuleGetFunction( &vmmUdim_, module_, "sampleVmmUdim" ), hipSuccess );
        ASSERT_EQ( hipModuleGetFunction( &nativeLod_, module_, "sampleNativeLod" ), hipSuccess );
        ASSERT_EQ( hipModuleGetFunction( &nativeGrad_, module_, "sampleNativeGrad" ), hipSuccess );
    }

    void TearDown() override
    {
        VmmLoaderTest::TearDown();
        if( module_ )
            hipModuleUnload( module_ );
    }

    // Device copies of the points and results for one launch.
    struct Launch
    {
        SamplePoint*  points  = nullptr;
        SampleResult* results = nullptr;
        uint32_t      count   = 0;

        ~Launch()
        {
            hipFree( points );
            hipFree( results );
        }
    };

    void upload( Launch& launch, const std::vector<SamplePoint>& points )
    {
        launch.count = static_cast<uint32_t>( points.size() );
        ASSERT_EQ( hipMalloc( reinterpret_cast<void**>( &launch.points ), points.size() * sizeof( SamplePoint ) ), hipSuccess );
        ASSERT_EQ( hipMalloc( reinterpret_cast<void**>( &launch.results ), points.size() * sizeof( SampleResult ) ), hipSuccess );
        ASSERT_EQ( hipMemcpy( launch.points, points.data(), points.size() * sizeof( SamplePoint ), hipMemcpyHostToDevice ), hipSuccess );
        ASSERT_EQ( hipMemset( launch.results, 0, points.size() * sizeof( SampleResult ) ), hipSuccess );
    }

    std::vector<SampleResult> download( const Launch& launch )
    {
        std::vector<SampleResult> results( launch.count );
        EXPECT_EQ( hipMemcpy( results.data(), launch.results, results.size() * sizeof( SampleResult ), hipMemcpyDeviceToHost ), hipSuccess );
        return results;
    }

    void launchKernel( hipFunction_t kernel, void** args, uint32_t count )
    {
        const uint32_t blockSize = 64;
        ASSERT_EQ( hipModuleLaunchKernel( kernel, ( count + blockSize - 1 ) / blockSize, 1, 1, blockSize, 1, 1, 0, stream_, args, nullptr ), hipSuccess );
    }

    // One launch of a VMM sampling kernel on the current context, without a request pass.
    std::vector<SampleResult> sampleVmmOnce( hipFunction_t kernel, uint32_t textureId, const std::vector<SamplePoint>& points, uint32_t channel = 0 )
    {
        Launch launch;
        upload( launch, points );
        void* lodArgs[]  = { &context_, &textureId, &launch.points, &launch.count, &launch.results };
        void* udimArgs[] = { &context_, &textureId, &channel, &launch.points, &launch.count, &launch.results };
        launchKernel( kernel, kernel == vmmUdim_ ? udimArgs : lodArgs, launch.count );
        EXPECT_EQ( hipStreamSynchronize( stream_ ), hipSuccess );
        return download( launch );
    }

    // Samples and processes the requests the sampler records, pass after pass, until every point is resident or
    // maxPasses. Returns the results of the last pass and the number of passes in passes.
    std::vector<SampleResult> sampleVmm( hipFunction_t kernel, uint32_t textureId, const std::vector<SamplePoint>& points, uint32_t& passes, uint32_t channel = 0, uint32_t maxPasses = 16 )
    {
        Launch launch;
        upload( launch, points );
        void* lodArgs[]  = { &context_, &textureId, &launch.points, &launch.count, &launch.results };
        void* udimArgs[] = { &context_, &textureId, &channel, &launch.points, &launch.count, &launch.results };
        std::vector<SampleResult> results;
        for( passes = 1; passes <= maxPasses; ++passes )
        {
            loader_->launchPrepare( stream_, context_ );
            launchKernel( kernel, kernel == vmmUdim_ ? udimArgs : lodArgs, launch.count );
            loader_->processRequests( stream_, context_ ).wait();
            EXPECT_EQ( hipStreamSynchronize( stream_ ), hipSuccess );
            results = download( launch );
            if( std::all_of( results.begin(), results.end(), []( const SampleResult& r ) { return r.resident != 0; } ) )
                return results;
        }
        return results;
    }

    std::vector<SampleResult> sampleNative( hipFunction_t kernel, hipTextureObject_t texture, const std::vector<SamplePoint>& points )
    {
        Launch launch;
        upload( launch, points );
        void* args[] = { &texture, &launch.points, &launch.count, &launch.results };
        launchKernel( kernel, args, launch.count );
        EXPECT_EQ( hipStreamSynchronize( stream_ ), hipSuccess );
        return download( launch );
    }

    // Compares the VMM results with the native ones, reporting the worst point, and returns the largest difference.
    float expectMatch( const std::vector<SamplePoint>& points, const std::vector<SampleResult>& vmm, const std::vector<SampleResult>& native, float tolerance )
    {
        EXPECT_EQ( vmm.size(), native.size() );
        float  worst      = 0.0f;
        size_t worstIndex = 0;
        for( size_t i = 0; i < vmm.size(); ++i )
        {
            EXPECT_EQ( vmm[i].resident, 1u ) << "point " << i;
            const float difference = maxChannelDifference( vmm[i].value, native[i].value );
            if( difference > worst )
            {
                worst      = difference;
                worstIndex = i;
            }
        }
        EXPECT_LE( worst, tolerance ) << "point " << worstIndex << " at (" << points[worstIndex].x << ", " << points[worstIndex].y
                                      << ") lod " << points[worstIndex].lod;
        return worst;
    }

    bool isReferenced( uint32_t resourceId )
    {
        uint32_t word = 0;
        EXPECT_EQ( hipMemcpy( &word, context_.referenceBits.ptr + resourceId / 32, sizeof( word ), hipMemcpyDeviceToHost ), hipSuccess );
        return ( word >> ( resourceId % 32 ) ) & 1u;
    }

    std::shared_ptr<TestImage> uniformImage( uint8_t value ) { return std::make_shared<TestImage>( 256, 256, 1, true, value, 4 ); }

    static SamplePoint pointAt( float x, float y, float texels = 1.0f / 256.0f )
    {
        SamplePoint point;
        point.x   = x;
        point.y   = y;
        point.ddx = { texels, 0.0f };
        point.ddy = { 0.0f, texels };
        return point;
    }

    hipModule_t   module_     = nullptr;
    hipFunction_t vmmLod_     = nullptr;
    hipFunction_t vmmGrad_    = nullptr;
    hipFunction_t vmmUdim_    = nullptr;
    hipFunction_t nativeLod_  = nullptr;
    hipFunction_t nativeGrad_ = nullptr;
};

// Hardware filtering weights have 8 fractional bits; a level blend compounds two of them.
constexpr float filterTolerance = 2.5f / 255.0f;

}  // namespace

TEST_F( VmmSamplerTest, Tex2DLodMatchesTheNativeTexture )
{
    createLoader( defaultOptions() );
    const auto image = std::make_shared<TestImage>( 1024, 1024, 11, TestImage::Pattern::Noise, 0, 4 );
    NativeTexture native( *image, hipAddressModeClamp );

    TextureDescriptor descriptor{};
    descriptor.addressMode[0] = hipAddressModeClamp;
    descriptor.addressMode[1] = hipAddressModeClamp;
    const DemandTexture& texture = loader_->createTexture( image, descriptor );

    // Inside the texture, and outside it, where the clamp decides; lods on and between levels, in the tail, and past the chain.
    std::vector<SamplePoint>       points  = gridPoints( 24, 0.0f, 1.0f, { 0.0f, 0.5f, 1.0f, 1.75f, 3.0f, 3.4f, 6.5f, 9.0f, 10.0f, 12.0f }, 1024 );
    const std::vector<SamplePoint> outside = gridPoints( 6, -0.4f, 1.4f, { 0.0f, 2.3f }, 1024 );
    points.insert( points.end(), outside.begin(), outside.end() );

    uint32_t                        passes = 0;
    const std::vector<SampleResult> vmm    = sampleVmm( vmmLod_, texture.getId(), points, passes );
    EXPECT_LE( passes, 3u ) << "metadata, then the pages of every level";
    const std::vector<SampleResult> expected = sampleNative( nativeLod_, native.get(), points );
    expectMatch( points, vmm, expected, filterTolerance );
}

TEST_F( VmmSamplerTest, Tex2DGradMatchesTheNativeTexture )
{
    createLoader( defaultOptions() );
    const auto image = std::make_shared<TestImage>( 1024, 1024, 11, TestImage::Pattern::Noise, 0, 4 );
    NativeTexture native( *image, hipAddressModeClamp );

    TextureDescriptor descriptor{};
    descriptor.addressMode[0] = hipAddressModeClamp;
    descriptor.addressMode[1] = hipAddressModeClamp;
    const DemandTexture& texture = loader_->createTexture( image, descriptor );

    // The lod comes from the gradients: 2^lod texels per pixel.
    const std::vector<SamplePoint> points = gridPoints( 24, 0.0f, 1.0f, { 0.0f, 0.5f, 1.3f, 2.0f, 4.6f, 7.0f }, 1024 );

    uint32_t                        passes = 0;
    const std::vector<SampleResult> vmm    = sampleVmm( vmmGrad_, texture.getId(), points, passes );
    EXPECT_LE( passes, 3u );
    const std::vector<SampleResult> expected = sampleNative( nativeGrad_, native.get(), points );
    expectMatch( points, vmm, expected, filterTolerance );
}

TEST_F( VmmSamplerTest, WrapAddressModeMatchesTheNativeTexture )
{
    createLoader( defaultOptions() );
    const auto image = std::make_shared<TestImage>( 512, 512, 10, TestImage::Pattern::Noise, 0, 4 );
    NativeTexture native( *image, hipAddressModeWrap );
    const DemandTexture& texture = loader_->createTexture( image, TextureDescriptor{} );  // Wraps by default.

    const std::vector<SamplePoint> points = gridPoints( 16, -1.3f, 2.3f, { 0.0f, 1.5f, 5.0f }, 512 );
    uint32_t                        passes = 0;
    const std::vector<SampleResult> vmm    = sampleVmm( vmmLod_, texture.getId(), points, passes );
    const std::vector<SampleResult> expected = sampleNative( nativeLod_, native.get(), points );
    expectMatch( points, vmm, expected, filterTolerance );
}

TEST_F( VmmSamplerTest, MissingResourcesAreRequestedUntilResident )
{
    createLoader( defaultOptions() );
    const auto           image   = std::make_shared<TestImage>( 1024, 1024, 11, TestImage::Pattern::Noise, 0, 4 );
    const DemandTexture& texture = loader_->createTexture( image, TextureDescriptor{} );
    const std::vector<SamplePoint> points{ pointAt( 0.1f, 0.1f ) };

    // Before the metadata is loaded: not resident, the metadata requested, no page touched.
    loader_->launchPrepare( stream_, context_ );
    std::vector<SampleResult> results = sampleVmmOnce( vmmLod_, texture.getId(), points );
    EXPECT_EQ( results[0].resident, 0u );
    EXPECT_TRUE( isReferenced( textureInfoResourceId( texture.getId() ) ) );
    EXPECT_FALSE( isReferenced( pageResourceId( 0 ) ) );
    EXPECT_EQ( image->getNumTilesRead(), 0u );

    // With the metadata: the tile under the point is requested. 1024x1024 rgba8 has 128x128 tiles.
    const DeviceTextureInfo info = loadTextureInfo( texture );
    loader_->launchPrepare( stream_, context_ );
    results = sampleVmmOnce( vmmLod_, texture.getId(), points );
    EXPECT_EQ( results[0].resident, 0u );
    EXPECT_TRUE( isReferenced( pageResourceId( tilePageId( info, 0, 0, 0 ) ) ) );
    EXPECT_FALSE( isReferenced( pageResourceId( tilePageId( info, 0, 3, 3 ) ) ) );

    // The request passes bring it in.
    uint32_t passes = 0;
    results         = sampleVmm( vmmLod_, texture.getId(), points, passes );
    EXPECT_EQ( results[0].resident, 1u );
    EXPECT_LE( passes, 2u );
    EXPECT_TRUE( isResident( tilePageId( info, 0, 0, 0 ) ) );
    EXPECT_LE( image->getNumTilesRead(), 4u ) << "only the tiles under the point";

    // A texture id past the loader's textures is reported resident, as zero, so that nothing waits for it.
    results = sampleVmmOnce( vmmLod_, texture.getId() + 1, points );
    EXPECT_EQ( results[0].resident, 1u );
    EXPECT_EQ( maxChannelDifference( results[0].value, make_float4( 0.0f, 0.0f, 0.0f, 0.0f ) ), 0.0f );
}

TEST_F( VmmSamplerTest, CascadeTextureIsExtendedBySampling )
{
    Options options        = defaultOptions();
    options.enableCascades = true;
    createLoader( options );
    const auto image = std::make_shared<TestImage>( 1024, 1024, 11, TestImage::Pattern::Noise, 0, 4 );
    NativeTexture native( *image, hipAddressModeClamp );
    TextureDescriptor descriptor{};
    descriptor.addressMode[0] = hipAddressModeClamp;
    descriptor.addressMode[1] = hipAddressModeClamp;
    const DemandTexture& texture = loader_->createTexture( image, descriptor );

    // Seen from afar first: the tail serves, and the finer levels have no pages.
    const std::vector<SamplePoint> far = gridPoints( 8, 0.0f, 1.0f, { 7.0f }, 1024 );
    uint32_t                       passes = 0;
    std::vector<SampleResult>      vmm    = sampleVmm( vmmLod_, texture.getId(), far, passes );
    expectMatch( far, vmm, sampleNative( nativeLod_, native.get(), far ), filterTolerance );
    const uint32_t cascadeLevel = readTextureInfo( texture.getId() ).cascadeLevel;
    EXPECT_EQ( cascadeLevel, 4u );

    // Then close up: one more pass for the cascade, then the tiles.
    const std::vector<SamplePoint> near = gridPoints( 8, 0.0f, 1.0f, { 0.0f, 1.5f }, 1024 );
    vmm                                 = sampleVmm( vmmLod_, texture.getId(), near, passes );
    EXPECT_LE( passes, 3u );
    expectMatch( near, vmm, sampleNative( nativeLod_, native.get(), near ), filterTolerance );
    EXPECT_EQ( readTextureInfo( texture.getId() ).cascadeLevel, 0u );
}

TEST_F( VmmSamplerTest, UdimCellsSampleTheirTextures )
{
    createLoader( defaultOptions() );
    // A 2x2 grid: white, black, grey and an empty cell.
    const std::vector<std::shared_ptr<ImageSource>> cells{ uniformImage( 255 ), uniformImage( 0 ), uniformImage( 128 ), nullptr };
    const DemandTexture& entry = loader_->createUdimTexture( cells, std::vector<TextureDescriptor>( 4 ), 2, 2 );

    // Grid coordinates: cell (u, v) spans [u, u + 1) x [v, v + 1); the entry wraps the grid by default.
    const std::vector<SamplePoint> points{ pointAt( 0.5f, 0.5f ), pointAt( 1.5f, 0.5f ), pointAt( 0.5f, 1.5f ),
                                           pointAt( 1.5f, 1.5f ), pointAt( 2.5f, 0.5f ), pointAt( -0.5f, 1.5f ) };
    const float expected[] = { 1.0f, 0.0f, 128.0f / 255.0f, 0.0f, 1.0f, 0.0f };

    uint32_t                        passes = 0;
    const std::vector<SampleResult> vmm    = sampleVmm( vmmUdim_, entry.getId(), points, passes );
    EXPECT_LE( passes, 4u ) << "the entry, the cells' metadata, then their pages";
    for( size_t i = 0; i < points.size(); ++i )
    {
        EXPECT_EQ( vmm[i].resident, 1u ) << "point " << i;
        EXPECT_NEAR( vmm[i].value.x, expected[i], 1.0f / 255.0f ) << "point " << i;
        EXPECT_NEAR( vmm[i].value.w, expected[i], 1.0f / 255.0f ) << "point " << i;
    }
    EXPECT_EQ( loader_->getStatistics().numTextures, 3u );
}

TEST_F( VmmSamplerTest, UdimChannelSelectsTheTextureOfTheCell )
{
    createLoader( defaultOptions() );
    // A 2x1 grid of 2 channels: id = entry + cell * 2 + channel.
    const std::vector<std::shared_ptr<ImageSource>> cells{ uniformImage( 255 ), uniformImage( 64 ), uniformImage( 0 ), uniformImage( 192 ) };
    const DemandTexture& entry = loader_->createUdimTexture( cells, std::vector<TextureDescriptor>( 4 ), 2, 1, 2 );

    const std::vector<SamplePoint> points{ pointAt( 0.5f, 0.5f ), pointAt( 1.5f, 0.5f ) };
    uint32_t                       passes = 0;
    std::vector<SampleResult>      vmm    = sampleVmm( vmmUdim_, entry.getId(), points, passes, 0 );
    EXPECT_NEAR( vmm[0].value.x, 1.0f, 1.0f / 255.0f );
    EXPECT_NEAR( vmm[1].value.x, 0.0f, 1.0f / 255.0f );
    vmm = sampleVmm( vmmUdim_, entry.getId(), points, passes, 1 );
    EXPECT_NEAR( vmm[0].value.x, 64.0f / 255.0f, 1.0f / 255.0f );
    EXPECT_NEAR( vmm[1].value.x, 192.0f / 255.0f, 1.0f / 255.0f );

    // A channel the grid does not have samples as zero, resident.
    vmm = sampleVmm( vmmUdim_, entry.getId(), points, passes, 2 );
    EXPECT_EQ( passes, 1u );
    EXPECT_EQ( vmm[0].resident, 1u );
    EXPECT_EQ( vmm[0].value.x, 0.0f );
}

TEST_F( VmmSamplerTest, UdimSamplerReadsAPlainTextureDirectly )
{
    createLoader( defaultOptions() );
    const DemandTexture&           texture = loader_->createTexture( uniformImage( 200 ), TextureDescriptor{} );
    const std::vector<SamplePoint> points{ pointAt( 0.25f, 0.75f ) };
    uint32_t                       passes = 0;
    const std::vector<SampleResult> udim = sampleVmm( vmmUdim_, texture.getId(), points, passes, 3 );
    const std::vector<SampleResult> grad = sampleVmm( vmmGrad_, texture.getId(), points, passes );
    EXPECT_EQ( udim[0].resident, 1u );
    EXPECT_NEAR( udim[0].value.x, 200.0f / 255.0f, 1.0f / 255.0f );
    EXPECT_EQ( maxChannelDifference( udim[0].value, grad[0].value ), 0.0f );
}

}  // namespace hip_demand::vmm::test
