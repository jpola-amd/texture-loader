#include <DemandLoading/VmmDemandTextureLoader.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vmm_loader_common.h"

#include <mempulse/mempulse.h>

#include <chrono>
#include <cmath>

#ifndef TEST_IMAGES_DIR
#error "TEST_IMAGES_DIR must be defined by the build target"
#endif

namespace {

using namespace vmm_loader_example;

class KernelModule
{
  public:
    explicit KernelModule( const fs::path& path )
    {
        HIP_CHECK( hipModuleLoad( &module_, path.string().c_str() ) );
        HIP_CHECK( hipModuleGetFunction( &regionKernel_, module_, "renderVmmTextureRegion" ) );
    }

    ~KernelModule()
    {
        if( module_ != nullptr )
            HIP_WARN( hipModuleUnload( module_ ) );
    }

    KernelModule( const KernelModule& )            = delete;
    KernelModule& operator=( const KernelModule& ) = delete;

    hipFunction_t regionKernel() const { return regionKernel_; }

  private:
    hipModule_t   module_       = nullptr;
    hipFunction_t regionKernel_ = nullptr;
};

class GpuMemoryMonitor
{
  public:
    explicit GpuMemoryMonitor( int device )
        : device_( device )
    {
#ifdef _WIN32
        constexpr MempulseBackend backend = MEMPULSE_BACKEND_D3DKMT;
#else
        constexpr MempulseBackend backend = MEMPULSE_BACKEND_ANY;
#endif
        if( MempulseInitialize( &context_, backend ) != MEMPULSE_SUCCESS )
            throw std::runtime_error( "Could not initialize MemPulse" );

        int deviceCount = 0;
        if( MempulseGetAvailabeDeviceCount( context_, &deviceCount ) != MEMPULSE_SUCCESS || device_ >= deviceCount )
        {
            MempulseShutdown( context_ );
            context_ = nullptr;
            throw std::runtime_error( "MemPulse could not find the active HIP device" );
        }
    }

    ~GpuMemoryMonitor()
    {
        if( context_ != nullptr )
            MempulseShutdown( context_ );
    }

    GpuMemoryMonitor( const GpuMemoryMonitor& )            = delete;
    GpuMemoryMonitor& operator=( const GpuMemoryMonitor& ) = delete;

    MempulseDeviceMemoryInfo memoryInfo() const
    {
        MempulseDeviceMemoryInfo info{};
        if( MempulseGetDeviceMemoryInfo( context_, device_, &info ) != MEMPULSE_SUCCESS )
            throw std::runtime_error( "MemPulse could not read GPU memory usage" );
        return info;
    }

  private:
    MempulseContext context_ = nullptr;
    int             device_  = 0;
};

struct TextureGrid
{
    uint32_t outputWidth;
    uint32_t outputHeight;
    uint32_t textureCount;
    uint32_t columnCount;
    uint32_t rowCount;
};

struct OutputTile
{
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
};

struct TileRenderResult
{
    uint32_t passes;
    uint32_t evictedPages;
};

class EvictionCounter
{
  public:
    explicit EvictionCounter( bool enabled )
        : enabled_( enabled )
    {
    }

    uint32_t synchronizeAndCount( const hip_demand::vmm::DeviceContext& context, hipStream_t stream )
    {
        if( !enabled_ )
        {
            HIP_CHECK( hipStreamSynchronize( stream ) );
            return 0;
        }

        if( currentResidence_.empty() )
        {
            currentResidence_.resize( context.residenceBits.len );
            previousResidence_.resize( context.residenceBits.len, 0 );
        }

        HIP_CHECK( hipMemcpyAsync( currentResidence_.data(), context.residenceBits.ptr,
                                   context.residenceBits.sizeInBytes(), hipMemcpyDeviceToHost, stream ) );
        HIP_CHECK( hipStreamSynchronize( stream ) );

        uint32_t evictions = 0;
        for( uint32_t resourceId = context.resourceTable.textureTiles.start;
             resourceId < context.resourceTable.textureTiles.end(); ++resourceId )
        {
            const uint32_t mask = 1u << ( resourceId & 31u );
            const uint32_t word = resourceId >> 5;
            if( ( previousResidence_[word] & mask ) && !( currentResidence_[word] & mask ) )
                ++evictions;
        }

        previousResidence_.swap( currentResidence_ );
        total_ += evictions;
        return evictions;
    }

    uint32_t total() const { return total_; }

  private:
    bool                  enabled_ = false;
    std::vector<uint32_t> previousResidence_;
    std::vector<uint32_t> currentResidence_;
    uint32_t              total_ = 0;
};

std::unique_ptr<hip_demand::vmm::DemandTextureLoader> createLoaderForImages( const hip_demand::vmm::Options& options,
                                                                             const std::vector<fs::path>& imagePaths )
{
    using namespace hip_demand::vmm;

    auto              loader = createDemandTextureLoader( options );
    TextureDescriptor descriptor{};
    descriptor.addressMode[0]   = hipAddressModeClamp;
    descriptor.addressMode[1]   = hipAddressModeClamp;
    descriptor.filterMode       = hipFilterModeLinear;
    descriptor.mipmapFilterMode = hipFilterModePoint;
    descriptor.normalizedCoords = true;

    for( const fs::path& imagePath : imagePaths )
        loader->createTexture( readImage( imagePath ), descriptor );
    return loader;
}

TileRenderResult renderTileUntilResident( hip_demand::vmm::DemandTextureLoader& loader,
                                          hipFunction_t                         kernel,
                                          GpuResources&                         gpu,
                                          TextureGrid                           textureGrid,
                                          OutputTile                            tile,
                                          dim3                                  block,
                                          uint32_t                              maxPasses,
                                          EvictionCounter&                      evictionCounter )
{
    using namespace hip_demand::vmm;

    const uint32_t gridWidth  = ( tile.width + block.x - 1 ) / block.x;
    const uint32_t gridHeight = ( tile.height + block.y - 1 ) / block.y;
    uint32_t       evictions  = 0;

    for( uint32_t pass = 1; pass <= maxPasses; ++pass )
    {
        DeviceContext context{};
        loader.launchPrepare( gpu.stream, context );

        DeviceContext mutableContext = context;
        void*         arguments[]    = { &mutableContext,
                                         &gpu.output,
                                         &textureGrid.outputWidth,
                                         &textureGrid.outputHeight,
                                         &textureGrid.textureCount,
                                         &textureGrid.columnCount,
                                         &textureGrid.rowCount,
                                         &tile.x,
                                         &tile.y,
                                         &tile.width,
                                         &tile.height };
        HIP_CHECK( hipModuleLaunchKernel( kernel, gridWidth, gridHeight, 1, block.x, block.y, 1, 0, gpu.stream, arguments, nullptr ) );

        Ticket ticket = loader.processRequests( gpu.stream, context );
        ticket.wait();
        evictions += evictionCounter.synchronizeAndCount( context, gpu.stream );
        if( ticket.numTasksTotal() == 0 )
            return { pass, evictions };
    }

    throw std::runtime_error( "Output tile (" + std::to_string( tile.x ) + ", " + std::to_string( tile.y )
                              + ") did not become resident" );
}

void renderTextureGrid( const fs::path& executableDir, const std::vector<fs::path>& imagePaths, bool enableEviction )
{
    using namespace hip_demand::vmm;

    constexpr uint32_t outputWidth      = 3840;
    constexpr uint32_t outputHeight     = 2160;
    constexpr uint32_t channels         = 4;
    constexpr uint32_t outputTileSize   = 512;
    constexpr uint32_t maxPassesPerTile = 64;
    constexpr double   bytesPerMiB      = 1024.0 * 1024.0;
    const dim3         block{ 16, 16 };

    Options options{};
    options.maxVirtualPages  = 64 * 1024;
    options.maxPhysicalPages = 64 * 1024;
    options.maxRequests      = 1024;
    options.maxEvictedPages  = 1024;
    options.enableEviction   = enableEviction;

    const fs::path outputPath = enableEviction ? "vmm_loader_vram_usage_eviction.png" : "vmm_loader_vram_usage_no_eviction.png";
    const fs::path kernelPath = executableDir / "vmm_loader_vram_usage_kernel.co";
    if( !fs::exists( kernelPath ) )
        throw std::runtime_error( "HIP module not found: " + kernelPath.string() );

    std::cout << "\nTiled render (eviction " << ( enableEviction ? "enabled" : "disabled" ) << ")\n"
              << "  Output size: " << outputWidth << 'x' << outputHeight << " pixels\n"
              << "  Output tile size: " << outputTileSize << 'x' << outputTileSize << " pixels\n";

    HIP_CHECK( hipSetDevice( 0 ) );
    // MemPulse resets the HIP device on shutdown, so keep it alive until the render resources are destroyed.
    GpuMemoryMonitor memoryMonitor( 0 );
    KernelModule     module( kernelPath );
    auto             loader = createLoaderForImages( options, imagePaths );

    const uint32_t textureCount = static_cast<uint32_t>( imagePaths.size() );
    const uint32_t columnCount = static_cast<uint32_t>( std::ceil( std::sqrt( static_cast<double>( textureCount ) ) ) );
    const uint32_t rowCount    = ( textureCount + columnCount - 1 ) / columnCount;
    const TextureGrid textureGrid{ outputWidth, outputHeight, textureCount, columnCount, rowCount };

    const size_t byteCount = static_cast<size_t>( outputWidth ) * outputHeight * channels;
    GpuResources gpu( byteCount );

    EvictionCounter evictionCounter( enableEviction );
    uint32_t        completedTiles = 0;
    const uint32_t  tilesPerRow    = ( outputWidth + outputTileSize - 1 ) / outputTileSize;
    const uint32_t  tileRows       = ( outputHeight + outputTileSize - 1 ) / outputTileSize;
    const uint32_t  tileCount      = tilesPerRow * tileRows;
    const auto      renderStart    = std::chrono::steady_clock::now();

    for( uint32_t tileY = 0; tileY < outputHeight; tileY += outputTileSize )
    {
        for( uint32_t tileX = 0; tileX < outputWidth; tileX += outputTileSize )
        {
            const OutputTile       tile{ tileX, tileY, std::min( outputTileSize, outputWidth - tileX ),
                                         std::min( outputTileSize, outputHeight - tileY ) };
            const TileRenderResult result = renderTileUntilResident( *loader, module.regionKernel(), gpu, textureGrid,
                                                                     tile, block, maxPassesPerTile, evictionCounter );
            ++completedTiles;

            std::cout << "  Tile " << completedTiles << '/' << tileCount << " at (" << tile.x << ", " << tile.y
                      << ") | passes: " << result.passes;
            if( enableEviction )
                std::cout << " | evictions: +" << result.evictedPages << " (total " << evictionCounter.total() << ')';
            const MempulseDeviceMemoryInfo memory = memoryMonitor.memoryInfo();
            std::cout << " | GPU VRAM: " << memory.dedicatedUsed / bytesPerMiB << " / "
                      << memory.dedicatedTotal / bytesPerMiB << " MiB\n";
        }
    }

    const auto renderEnd  = std::chrono::steady_clock::now();
    const auto renderTime = std::chrono::duration_cast<std::chrono::milliseconds>( renderEnd - renderStart );
    std::cout << "  Render complete: " << completedTiles << '/' << tileCount << " tiles in " << renderTime.count() << " ms";
    if( enableEviction )
        std::cout << "; VMM page evictions: " << evictionCounter.total();
    std::cout << '\n';

    if( enableEviction && evictionCounter.total() == 0 )
        throw std::runtime_error( "Eviction test loaded every output tile without evicting a VMM page" );

    saveGpuOutput( outputPath, gpu.stream, gpu.output, outputWidth, outputHeight, channels );
}

}  // namespace

int main( int argc, char** argv )
{
    try
    {
        constexpr bool includeOver4kImages = true;
        const fs::path executableDir       = fs::absolute( fs::path{ argv[0] } ).parent_path();
        fs::path       imageDirectory      = fs::path{ TEST_IMAGES_DIR } / "png";
        imageDirectory.make_preferred();

        std::cout << "VMM loader VRAM usage\n"
                  << "  Image directory: " << imageDirectory.string() << '\n'
                  << "  Size filter: " << ( includeOver4kImages ? "all images" : "up to 4096 pixels per side" )
                  << "\n  GPU VRAM logging: enabled\n";

        const auto imagePaths = findImages( imageDirectory, includeOver4kImages );
        std::cout << "  Selected textures (" << imagePaths.size() << "):\n";
        for( const fs::path& imagePath : imagePaths )
        {
            const auto  image = readImage( imagePath );
            const auto& info  = image->getInfo();
            std::cout << "    " << imagePath.filename().string() << " (" << info.width << 'x' << info.height << ")\n";
        }
        std::cout << "  GPU VRAM readings cover the whole device, including other processes.\n";

        renderTextureGrid( executableDir, imagePaths, false );
        renderTextureGrid( executableDir, imagePaths, true );
        std::cout << "\nResult: PASS (VRAM usage renders completed)\n";
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr << "\nResult: FAIL - " << error.what() << '\n';
        return 1;
    }
}
