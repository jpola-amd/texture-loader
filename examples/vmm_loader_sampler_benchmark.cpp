#include <DemandLoading/VmmDemandTextureLoader.h>

#include "vmm_loader_common.h"

#include <array>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <utility>

#ifndef TEST_IMAGES_DIR
#error "TEST_IMAGES_DIR must be defined by the build target"
#endif

namespace {

using namespace vmm_loader_example;

constexpr uint32_t outputWidth     = 3840;
constexpr uint32_t outputHeight    = 2160;
constexpr uint32_t blockWidth      = 16;
constexpr uint32_t blockHeight     = 16;
constexpr uint32_t outputChannels  = 4;
constexpr uint32_t maxWarmupPasses = 256;
constexpr uint32_t callCount       = 1000;
constexpr uint32_t batchSize       = 10;
constexpr float    mipLevel        = 0.0f;

struct BenchmarkTimes
{
    float vmmMilliseconds    = 0.0f;
    float nativeMilliseconds = 0.0f;
};

enum class ResultStatus
{
    completed,
    skipped,
    failed
};

struct ImageResult
{
    std::string    name;
    std::string    fileBytes   = "-";
    std::string    dimensions  = "-";
    std::string    pixelFormat = "-";
    BenchmarkTimes times{};
    ResultStatus   status = ResultStatus::failed;
    std::string    note;
};

std::shared_ptr<hip_demand::ImageSource> readBenchmarkImage( const fs::path& path )
{
    auto                    image = std::make_shared<hip_demand::OIIOReader>( path.string() );
    hip_demand::TextureInfo info{};
    image->open( &info );
    if( !info.isValid || info.width == 0 || info.height == 0 )
        throw std::runtime_error( "Invalid image: " + path.string() );
    return image;
}

std::string pixelFormatName( const hip_demand::TextureInfo& info )
{
    const char* channels[] = { "", "r", "rg", "rgb", "rgba" };
    if( info.numChannels < 1 || info.numChannels > 4 )
        return "unsupported";

    const char* format = nullptr;
    switch( info.format )
    {
        case HIP_AD_FORMAT_UNSIGNED_INT8:
            format = "8";
            break;
        case HIP_AD_FORMAT_SIGNED_INT8:
            format = "8s";
            break;
        case HIP_AD_FORMAT_UNSIGNED_INT16:
            format = "16u";
            break;
        case HIP_AD_FORMAT_SIGNED_INT16:
            format = "16s";
            break;
        case HIP_AD_FORMAT_UNSIGNED_INT32:
            format = "32u";
            break;
        case HIP_AD_FORMAT_SIGNED_INT32:
            format = "32s";
            break;
        case HIP_AD_FORMAT_HALF:
            format = "16f";
            break;
        case HIP_AD_FORMAT_FLOAT:
            format = "32f";
            break;
        default:
            return "unsupported";
    }
    return std::string( channels[info.numChannels] ) + format;
}

class KernelModule
{
  public:
    explicit KernelModule( const fs::path& path )
    {
        HIP_CHECK( hipModuleLoad( &module_, path.string().c_str() ) );
        HIP_CHECK( hipModuleGetFunction( &vmmTex2DLodKernel_, module_, "renderVmmTex2DLodBenchmark" ) );
        HIP_CHECK( hipModuleGetFunction( &hipTex2DLodKernel_, module_, "renderHipTex2DLodBenchmark" ) );
    }

    ~KernelModule()
    {
        if( module_ != nullptr )
            HIP_WARN( hipModuleUnload( module_ ) );
    }

    KernelModule( const KernelModule& )            = delete;
    KernelModule& operator=( const KernelModule& ) = delete;

    hipFunction_t vmmTex2DLodKernel() const { return vmmTex2DLodKernel_; }
    hipFunction_t hipTex2DLodKernel() const { return hipTex2DLodKernel_; }

  private:
    hipModule_t   module_            = nullptr;
    hipFunction_t vmmTex2DLodKernel_ = nullptr;
    hipFunction_t hipTex2DLodKernel_ = nullptr;
};

class GpuTimer
{
  public:
    GpuTimer()
    {
        HIP_CHECK( hipEventCreate( &start_ ) );
        HIP_CHECK( hipEventCreate( &stop_ ) );
    }

    ~GpuTimer()
    {
        HIP_WARN( hipEventDestroy( start_ ) );
        HIP_WARN( hipEventDestroy( stop_ ) );
    }

    GpuTimer( const GpuTimer& )            = delete;
    GpuTimer& operator=( const GpuTimer& ) = delete;

    void start( hipStream_t stream ) { HIP_CHECK( hipEventRecord( start_, stream ) ); }

    float stop( hipStream_t stream )
    {
        HIP_CHECK( hipEventRecord( stop_, stream ) );
        HIP_CHECK( hipEventSynchronize( stop_ ) );
        float milliseconds = 0.0f;
        HIP_CHECK( hipEventElapsedTime( &milliseconds, start_, stop_ ) );
        return milliseconds;
    }

  private:
    hipEvent_t start_ = nullptr;
    hipEvent_t stop_  = nullptr;
};

class CompletionEvent
{
  public:
    CompletionEvent() { HIP_CHECK( hipEventCreateWithFlags( &event_, hipEventDisableTiming ) ); }
    ~CompletionEvent() { HIP_WARN( hipEventDestroy( event_ ) ); }

    CompletionEvent( const CompletionEvent& )            = delete;
    CompletionEvent& operator=( const CompletionEvent& ) = delete;

    void record( hipStream_t stream ) { HIP_CHECK( hipEventRecord( event_, stream ) ); }
    void wait() { HIP_CHECK( hipEventSynchronize( event_ ) ); }

  private:
    hipEvent_t event_ = nullptr;
};

struct NativeTextureFormat
{
    hipChannelFormatDesc channelDesc{};
    hipTextureReadMode   readMode     = hipReadModeNormalizedFloat;
    bool                 convertInt32 = false;
};

NativeTextureFormat makeNativeTextureFormat( hipArray_Format sourceFormat )
{
    const bool convertInt32 = sourceFormat == HIP_AD_FORMAT_UNSIGNED_INT32 || sourceFormat == HIP_AD_FORMAT_SIGNED_INT32;
    const bool floating = convertInt32 || sourceFormat == HIP_AD_FORMAT_HALF || sourceFormat == HIP_AD_FORMAT_FLOAT;
    const bool signedInteger = sourceFormat == HIP_AD_FORMAT_SIGNED_INT8 || sourceFormat == HIP_AD_FORMAT_SIGNED_INT16;
    const hipChannelFormatKind kind =
        floating ? hipChannelFormatKindFloat : ( signedInteger ? hipChannelFormatKindSigned : hipChannelFormatKindUnsigned );
    const unsigned int bits = 8 * hip_demand::getBytesPerChannel( sourceFormat );
    if( bits == 0 )
        throw std::invalid_argument( "Unsupported image channel format" );

    return { hipCreateChannelDesc( bits, bits, bits, bits, kind ),
             floating ? hipReadModeElementType : hipReadModeNormalizedFloat, convertInt32 };
}

float normalizedInt32Channel( const uint8_t* channel, hipArray_Format format )
{
    // Hardware linear filtering uses floating-point channels for 32-bit integer sources.
    if( format == HIP_AD_FORMAT_UNSIGNED_INT32 )
    {
        uint32_t value = 0;
        std::memcpy( &value, channel, sizeof( value ) );
        return static_cast<float>( value ) * ( 1.0f / 4294967295.0f );
    }

    int32_t value = 0;
    std::memcpy( &value, channel, sizeof( value ) );
    return std::max( -1.0f, static_cast<float>( value ) * ( 1.0f / 2147483647.0f ) );
}

class HipMipmappedTexture
{
  public:
    HipMipmappedTexture( hip_demand::ImageSource& image, const hip_demand::TextureInfo& info )
    {
        try
        {
            const NativeTextureFormat format = makeNativeTextureFormat( info.format );
            HIP_CHECK( hipMallocMipmappedArray( &mipmappedArray_, &format.channelDesc,
                                                make_hipExtent( info.width, info.height, 0 ), info.numMipLevels ) );
            uploadMipLevels( image, info, format.convertInt32 );
            createTextureObject( info.numMipLevels, format.readMode );
        }
        catch( ... )
        {
            reset();
            throw;
        }
    }

    ~HipMipmappedTexture() { reset(); }

    HipMipmappedTexture( const HipMipmappedTexture& )            = delete;
    HipMipmappedTexture& operator=( const HipMipmappedTexture& ) = delete;

    hipTextureObject_t get() const { return texture_; }

  private:
    static std::vector<uint8_t> readRgbaMipLevel( hip_demand::ImageSource&       image,
                                                  const hip_demand::TextureInfo& info,
                                                  uint32_t                       level,
                                                  uint32_t                       width,
                                                  uint32_t                       height,
                                                  bool                           convertInt32 )
    {
        const size_t         channelBytes = hip_demand::getBytesPerChannel( info.format );
        const size_t         pixelCount   = static_cast<size_t>( width ) * height;
        std::vector<uint8_t> source( pixelCount * info.numChannels * channelBytes );
        if( !image.readMipLevel( reinterpret_cast<char*>( source.data() ), level, width, height ) )
            throw std::runtime_error( "Could not read mip level " + std::to_string( level ) );

        // HIP arrays have 1, 2 or 4 channels. Pad RGB and smaller formats to RGBA.
        std::vector<uint8_t> rgba( pixelCount * 4 * channelBytes, 0 );
        for( size_t pixel = 0; pixel < pixelCount; ++pixel )
        {
            const uint8_t* src = source.data() + pixel * info.numChannels * channelBytes;
            uint8_t*       dst = rgba.data() + pixel * 4 * channelBytes;
            if( convertInt32 )
            {
                for( uint32_t channel = 0; channel < info.numChannels; ++channel )
                {
                    const float value = normalizedInt32Channel( src + channel * channelBytes, info.format );
                    std::memcpy( dst + channel * channelBytes, &value, sizeof( value ) );
                }
            }
            else
                std::memcpy( dst, src, info.numChannels * channelBytes );
        }
        return rgba;
    }

    void uploadMipLevels( hip_demand::ImageSource& image, const hip_demand::TextureInfo& info, bool convertInt32 )
    {
        const size_t channelBytes = hip_demand::getBytesPerChannel( info.format );
        uint32_t     width        = info.width;
        uint32_t     height       = info.height;
        for( uint32_t level = 0; level < info.numMipLevels; ++level )
        {
            const std::vector<uint8_t> rgba       = readRgbaMipLevel( image, info, level, width, height, convertInt32 );
            hipArray_t                 levelArray = nullptr;
            HIP_CHECK( hipGetMipmappedArrayLevel( &levelArray, mipmappedArray_, level ) );
            const size_t rowBytes = static_cast<size_t>( width ) * 4 * channelBytes;
            HIP_CHECK( hipMemcpy2DToArray( levelArray, 0, 0, rgba.data(), rowBytes, rowBytes, height, hipMemcpyHostToDevice ) );
            width  = std::max( 1u, width / 2 );
            height = std::max( 1u, height / 2 );
        }
    }

    void createTextureObject( uint32_t mipLevelCount, hipTextureReadMode readMode )
    {
        hipResourceDesc resourceDesc{};
        resourceDesc.resType           = hipResourceTypeMipmappedArray;
        resourceDesc.res.mipmap.mipmap = mipmappedArray_;

        hipTextureDesc textureDesc{};
        textureDesc.addressMode[0]      = hipAddressModeClamp;
        textureDesc.addressMode[1]      = hipAddressModeClamp;
        textureDesc.filterMode          = hipFilterModeLinear;
        textureDesc.readMode            = readMode;
        textureDesc.normalizedCoords    = 1;
        textureDesc.mipmapFilterMode    = hipFilterModePoint;
        textureDesc.minMipmapLevelClamp = 0.0f;
        textureDesc.maxMipmapLevelClamp = static_cast<float>( mipLevelCount - 1 );
        HIP_CHECK( hipCreateTextureObject( &texture_, &resourceDesc, &textureDesc, nullptr ) );
    }

    void reset()
    {
        if( texture_ != 0 )
            HIP_WARN( hipDestroyTextureObject( texture_ ) );
        if( mipmappedArray_ != nullptr )
            HIP_WARN( hipFreeMipmappedArray( mipmappedArray_ ) );
        texture_        = 0;
        mipmappedArray_ = nullptr;
    }

    hipMipmappedArray_t mipmappedArray_ = nullptr;
    hipTextureObject_t  texture_        = 0;
};

template <class Launch>
void makeVmmTextureResident( hip_demand::vmm::DemandTextureLoader& loader,
                             hipStream_t                           stream,
                             hip_demand::vmm::DeviceContext&       context,
                             uint32_t                              maxPasses,
                             Launch&&                              launch )
{
    using namespace hip_demand::vmm;

    for( uint32_t pass = 1; pass <= maxPasses; ++pass )
    {
        loader.launchPrepare( stream, context );
        launch( context );

        Ticket ticket = loader.processRequests( stream, context );
        ticket.wait();
        HIP_CHECK( hipStreamSynchronize( stream ) );
        if( ticket.numTasksTotal() == 0 )
            return;
    }
    throw std::runtime_error( "VMM benchmark texture did not become resident" );
}

template <class Launch>
float measureQueuedRenderCalls( hipStream_t stream, uint32_t callCount, uint32_t batchSize, Launch&& launch )
{
    if( batchSize == 0 )
        throw std::invalid_argument( "Batch size must be greater than zero" );

    constexpr uint32_t                              bufferedBatchCount = 2;
    std::array<CompletionEvent, bufferedBatchCount> batchFinished;
    GpuTimer                                        timer;
    uint32_t                                        submittedCalls = 0;
    uint32_t                                        batchIndex     = 0;

    timer.start( stream );
    while( submittedCalls < callCount )
    {
        // Keep one batch queued while the CPU waits to reuse the other batch's event.
        const uint32_t eventIndex = batchIndex % bufferedBatchCount;
        if( batchIndex >= bufferedBatchCount )
            batchFinished[eventIndex].wait();

        const uint32_t callsInBatch = std::min( batchSize, callCount - submittedCalls );
        for( uint32_t call = 0; call < callsInBatch; ++call )
            launch();

        batchFinished[eventIndex].record( stream );
        submittedCalls += callsInBatch;
        ++batchIndex;
    }

    return timer.stop( stream );
}

BenchmarkTimes benchmarkTex2DLod( const fs::path& executableDir, const std::shared_ptr<hip_demand::ImageSource>& image )
{
    using namespace hip_demand::vmm;

    uint32_t renderWidth      = outputWidth;
    uint32_t renderHeight     = outputHeight;
    float    selectedMipLevel = mipLevel;

    Options options{};
    options.maxTextures      = 1;
    options.maxVirtualPages  = 64 * 1024;
    options.maxPhysicalPages = 64 * 1024;
    options.maxRequests      = 8192;
    options.enableEviction   = false;
    options.maxEvictedPages  = 0;

    const fs::path kernelPath = executableDir / "vmm_loader_sampler_benchmark_kernel.co";
    if( !fs::exists( kernelPath ) )
        throw std::runtime_error( "HIP module not found: " + kernelPath.string() );

    const hip_demand::TextureInfo& info = image->getInfo();
    KernelModule                   module( kernelPath );
    const size_t                   byteCount = static_cast<size_t>( outputWidth ) * outputHeight * outputChannels;
    GpuResources                   gpu( byteCount );
    const uint32_t                 gridWidth  = ( outputWidth + blockWidth - 1 ) / blockWidth;
    const uint32_t                 gridHeight = ( outputHeight + blockHeight - 1 ) / blockHeight;

    BenchmarkTimes times{};
    {
        auto              loader = createDemandTextureLoader( options );
        TextureDescriptor descriptor{};
        descriptor.addressMode[0]   = hipAddressModeClamp;
        descriptor.addressMode[1]   = hipAddressModeClamp;
        descriptor.filterMode       = hipFilterModeLinear;
        descriptor.mipmapFilterMode = hipFilterModePoint;
        descriptor.normalizedCoords = true;
        loader->createTexture( image, descriptor );

        DeviceContext context{};
        const auto    launchVmmKernel = [&]( const DeviceContext& deviceContext ) {
            DeviceContext mutableContext = deviceContext;
            void* arguments[] = { &mutableContext, &gpu.output, &renderWidth, &renderHeight, &selectedMipLevel };
            HIP_CHECK( hipModuleLaunchKernel( module.vmmTex2DLodKernel(), gridWidth, gridHeight, 1, blockWidth,
                                              blockHeight, 1, 0, gpu.stream, arguments, nullptr ) );
        };
        makeVmmTextureResident( *loader, gpu.stream, context, maxWarmupPasses, launchVmmKernel );

        const auto launchVmm  = [&] { launchVmmKernel( context ); };
        times.vmmMilliseconds = measureQueuedRenderCalls( gpu.stream, callCount, batchSize, launchVmm );
    }

    {
        HipMipmappedTexture texture( *image, info );
        hipTextureObject_t  textureObject = texture.get();
        const auto          launchHip     = [&] {
            void* arguments[] = { &textureObject, &gpu.output, &renderWidth, &renderHeight, &selectedMipLevel };
            HIP_CHECK( hipModuleLaunchKernel( module.hipTex2DLodKernel(), gridWidth, gridHeight, 1, blockWidth,
                                              blockHeight, 1, 0, gpu.stream, arguments, nullptr ) );
        };

        launchHip();
        HIP_CHECK( hipStreamSynchronize( gpu.stream ) );
        times.nativeMilliseconds = measureQueuedRenderCalls( gpu.stream, callCount, batchSize, launchHip );
    }
    return times;
}

ImageResult benchmarkImage( const fs::path& imagePath,
                            const fs::path& imageDir,
                            const fs::path& executableDir,
                            uint32_t        maxWidth,
                            uint32_t        maxHeight,
                            size_t          index,
                            size_t          total )
{
    ImageResult result{};
    result.name = imagePath.lexically_relative( imageDir ).string();
    std::cout << "Rendering [" << index + 1 << '/' << total << "] " << result.name << " ... " << std::flush;

    try
    {
        result.fileBytes   = std::to_string( fs::file_size( imagePath ) );
        const auto  image  = readBenchmarkImage( imagePath );
        const auto& info   = image->getInfo();
        result.dimensions  = std::to_string( info.width ) + 'x' + std::to_string( info.height );
        result.pixelFormat = pixelFormatName( info );

        if( info.width > maxWidth || info.height > maxHeight )
        {
            result.status = ResultStatus::skipped;
            result.note = "exceeds native mipmapped texture limit " + std::to_string( maxWidth ) + 'x' + std::to_string( maxHeight );
        }
        else
        {
            result.times  = benchmarkTex2DLod( executableDir, image );
            result.status = ResultStatus::completed;
        }
    }
    catch( const std::exception& error )
    {
        result.note = error.what();
    }

    switch( result.status )
    {
        case ResultStatus::completed:
            std::cout << "done\n";
            break;
        case ResultStatus::skipped:
            std::cout << "SKIPPED\n";
            break;
        case ResultStatus::failed:
            std::cout << "FAILED\n";
            break;
    }
    return result;
}

std::string formatNumber( float value, const char* suffix = "" )
{
    std::ostringstream out;
    out << std::fixed << std::setprecision( 3 ) << value << suffix;
    return out.str();
}

using TableRow = std::array<std::string, 8>;

const char* statusName( ResultStatus status )
{
    switch( status )
    {
        case ResultStatus::completed:
            return "OK";
        case ResultStatus::skipped:
            return "SKIPPED";
        case ResultStatus::failed:
            return "FAILED";
    }
    return "FAILED";
}

TableRow makeTableRow( const ImageResult& result )
{
    const char* status = statusName( result.status );
    if( result.status != ResultStatus::completed )
        return { result.name, result.fileBytes, result.dimensions, result.pixelFormat, "-", "-", "-", status };

    return { result.name,
             result.fileBytes,
             result.dimensions,
             result.pixelFormat,
             formatNumber( result.times.vmmMilliseconds ),
             formatNumber( result.times.nativeMilliseconds ),
             formatNumber( result.times.vmmMilliseconds / result.times.nativeMilliseconds, "x" ),
             status };
}

void printResultsTable( const std::vector<ImageResult>& results )
{
    size_t fileWidth = std::string( "File" ).size();
    for( const ImageResult& result : results )
        fileWidth = std::max( fileWidth, result.name.size() );
    const std::array<size_t, 8> widths{ fileWidth, 12, 11, 8, 10, 10, 11, 7 };
    const auto                  printSeparator = [&] {
        std::cout << '+';
        for( size_t width : widths )
            std::cout << std::string( width + 2, '-' ) << '+';
        std::cout << '\n';
    };
    const auto printRow = [&]( const TableRow& row ) {
        std::cout << '|';
        for( size_t column = 0; column < row.size(); ++column )
            std::cout << ' ' << ( column == 1 || ( column >= 4 && column <= 6 ) ? std::right : std::left )
                      << std::setw( static_cast<int>( widths[column] ) ) << row[column] << " |";
        std::cout << '\n';
    };

    std::cout << '\n';
    printSeparator();
    printRow( TableRow{ "File", "Bytes", "Dimensions", "Format", "VMM ms", "Native ms", "VMM/native", "Status" } );
    printSeparator();
    for( const ImageResult& result : results )
        printRow( makeTableRow( result ) );
    printSeparator();

    for( const ImageResult& result : results )
    {
        if( !result.note.empty() )
            std::cout << result.name << ": " << result.note << '\n';
    }
}

}  // namespace

int main( int argc, char** argv )
{
    try
    {
        const fs::path executableDir = fs::absolute( fs::path{ argv[0] } ).parent_path();
        fs::path       imageDir      = fs::path{ TEST_IMAGES_DIR };
        imageDir.make_preferred();

        HIP_CHECK( hipSetDevice( 0 ) );
        hipDeviceProp_t deviceProperties{};
        HIP_CHECK( hipGetDeviceProperties( &deviceProperties, 0 ) );
        const uint32_t maxWidth =
            static_cast<uint32_t>( deviceProperties.maxTexture2DMipmap[0] > 0 ? deviceProperties.maxTexture2DMipmap[0] :
                                                                                deviceProperties.maxTexture2D[0] );
        const uint32_t maxHeight =
            static_cast<uint32_t>( deviceProperties.maxTexture2DMipmap[1] > 0 ? deviceProperties.maxTexture2DMipmap[1] :
                                                                                deviceProperties.maxTexture2D[1] );

        const std::vector<fs::path> imagePaths = findImages( imageDir, true, true );
        std::cout << "VMM loader sampler benchmark\n"
                  << "  Input: " << imageDir.string() << '\n'
                  << "  Files: " << imagePaths.size() << '\n'
                  << "  Rendering: " << outputWidth << 'x' << outputHeight << ", mip " << mipLevel << ", " << callCount
                  << " calls per sampler\n"
                  << "  Times are total GPU times after warm-up\n";

        std::vector<ImageResult> results;
        results.reserve( imagePaths.size() );
        size_t completed = 0;
        size_t skipped   = 0;
        size_t failed    = 0;
        for( size_t imageIndex = 0; imageIndex < imagePaths.size(); ++imageIndex )
        {
            ImageResult result = benchmarkImage( imagePaths[imageIndex], imageDir, executableDir, maxWidth, maxHeight,
                                                 imageIndex, imagePaths.size() );
            switch( result.status )
            {
                case ResultStatus::completed:
                    ++completed;
                    break;
                case ResultStatus::skipped:
                    ++skipped;
                    break;
                case ResultStatus::failed:
                    ++failed;
                    break;
            }
            results.push_back( std::move( result ) );
        }

        printResultsTable( results );

        std::cout << "\nResult: " << ( failed == 0 && completed != 0 ? "PASS" : "FAIL" ) << " (completed " << completed
                  << ", skipped " << skipped << ", failed " << failed << ")\n";
        return failed == 0 && completed != 0 ? 0 : 1;
    }
    catch( const std::exception& error )
    {
        std::cerr << "\nResult: FAIL - " << error.what() << '\n';
        return 1;
    }
}
