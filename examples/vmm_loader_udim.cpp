// Renders a UDIM texture assembled from the test images with the VMM demand loader.
//
// The images fill a 3x3 grid with the center cell left empty, repeating when there are fewer images than cells. The
// grid has two channels: the images as they are, and the same images with inverted colors.
// Four views are rendered until every pixel is final and saved as PNG: the grid, a distant view of it, which samples
// the subtextures' mip tails, the inverted channel, and a grid whose cells are all empty, which renders black in one
// pass. The passes printed for each view show the loading steps: entry-point metadata, subtexture metadata, tiles.

#include <DemandLoading/VmmDemandTextureLoader.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vmm_loader_common.h"

#include <ImageSource/WrappedImageSource.h>

#include <array>
#include <cstring>

#ifndef TEST_IMAGES_DIR
#error "TEST_IMAGES_DIR must be defined by the build target"
#endif

namespace {

using namespace vmm_loader_example;

constexpr uint32_t outputWidth    = 1536;
constexpr uint32_t outputHeight   = 1536;
constexpr uint32_t outputChannels = 4;
constexpr uint32_t blockSize      = 16;
constexpr uint32_t maxPasses      = 16;
constexpr uint32_t udim           = 3;
constexpr uint32_t vdim           = 3;
constexpr uint32_t numChannels    = 2;  // The images, and the images with inverted colors.
constexpr uint32_t emptyCell      = 4;  // The center of the grid.

// Inverts the colors of an 8-bit image source, leaving a fourth (alpha) channel alone.
class InvertedImage : public hip_demand::WrappedImageSource
{
  public:
    using WrappedImageSource::WrappedImageSource;

    void open( hip_demand::TextureInfo* info ) override
    {
        WrappedImageSource::open( info );
        if( getInfo().format != HIP_AD_FORMAT_UNSIGNED_INT8 )
            throw std::runtime_error( "InvertedImage needs an 8-bit image" );
    }

    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, hipStream_t stream ) override
    {
        if( !WrappedImageSource::readMipLevel( dest, mipLevel, width, height, stream ) )
            return false;
        invert( dest, static_cast<size_t>( width ) * height );
        return true;
    }

    bool readTile( char* dest, unsigned int mipLevel, const hip_demand::Tile& tile, hipStream_t stream ) override
    {
        if( !WrappedImageSource::readTile( dest, mipLevel, tile, stream ) )
            return false;
        invert( dest, static_cast<size_t>( tile.width ) * tile.height );
        return true;
    }

    bool readMipTail( char* dest, unsigned int mipTailFirstLevel, unsigned int numMipLevels, const uint2* mipLevelDims, hipStream_t stream ) override
    {
        if( !WrappedImageSource::readMipTail( dest, mipTailFirstLevel, numMipLevels, mipLevelDims, stream ) )
            return false;
        size_t texelCount = 0;
        for( unsigned int level = mipTailFirstLevel; level < numMipLevels; ++level )
            texelCount += static_cast<size_t>( mipLevelDims[level].x ) * mipLevelDims[level].y;
        invert( dest, texelCount );
        return true;
    }

    bool readBaseColor( float4& color ) override
    {
        if( !WrappedImageSource::readBaseColor( color ) )
            return false;
        color = make_float4( 1.0f - color.x, 1.0f - color.y, 1.0f - color.z, color.w );
        return true;
    }

  private:
    void invert( char* texels, size_t texelCount ) const
    {
        const unsigned int channels      = getInfo().numChannels;
        const unsigned int colorChannels = std::min( channels, 3u );
        for( size_t texel = 0; texel < texelCount; ++texel )
        {
            unsigned char* bytes = reinterpret_cast<unsigned char*>( texels ) + texel * channels;
            for( unsigned int channel = 0; channel < colorChannels; ++channel )
                bytes[channel] = static_cast<unsigned char>( 255 - bytes[channel] );
        }
    }
};

std::shared_ptr<hip_demand::ImageSource> readChannelImage( const fs::path& path, uint32_t channel )
{
    std::shared_ptr<hip_demand::ImageSource> image = readImage( path );
    return channel == 0 ? image : std::make_shared<InvertedImage>( std::move( image ) );
}

class KernelModule
{
  public:
    explicit KernelModule( const fs::path& path )
    {
        HIP_CHECK( hipModuleLoad( &module_, path.string().c_str() ) );
        HIP_CHECK( hipModuleGetFunction( &kernel_, module_, "renderUdimTexture" ) );
    }

    ~KernelModule()
    {
        if( module_ != nullptr )
            HIP_WARN( hipModuleUnload( module_ ) );
    }

    KernelModule( const KernelModule& )            = delete;
    KernelModule& operator=( const KernelModule& ) = delete;

    hipFunction_t kernel() const { return kernel_; }

  private:
    hipModule_t   module_ = nullptr;
    hipFunction_t kernel_ = nullptr;
};

struct View
{
    const char* name;
    float       footprintScale;
    uint32_t    channel;
    bool        allCellsEmpty = false;  // No images at all: every cell of every channel is empty.
};

hip_demand::vmm::TextureDescriptor makeDescriptor()
{
    hip_demand::vmm::TextureDescriptor descriptor{};
    descriptor.addressMode[0]   = hipAddressModeClamp;
    descriptor.addressMode[1]   = hipAddressModeClamp;
    descriptor.filterMode       = hipFilterModeLinear;
    descriptor.mipmapFilterMode = hipFilterModeLinear;
    descriptor.normalizedCoords = true;
    descriptor.maxAnisotropy    = 1;
    return descriptor;
}

uint32_t countPixels( const std::vector<uint8_t>& pixels, uint8_t alpha )
{
    uint32_t count = 0;
    for( size_t i = 3; i < pixels.size(); i += outputChannels )
        count += pixels[i] == alpha ? 1 : 0;
    return count;
}

void renderView( const fs::path& kernelPath, const std::vector<fs::path>& cellImages, const View& view, const fs::path& outputPath )
{
    using namespace hip_demand::vmm;

    Options options{};
    options.maxTextures      = 64;
    options.maxVirtualPages  = 64 * 1024;
    options.maxPhysicalPages = 64 * 1024;
    options.maxRequests      = 8192;
    options.enableEviction   = false;

    KernelModule module( kernelPath );
    GpuResources gpu( static_cast<size_t>( outputWidth ) * outputHeight * outputChannels );
    auto         loader = createDemandTextureLoader( options );

    // The channels of a cell are listed together.
    std::vector<std::shared_ptr<hip_demand::ImageSource>> images( udim * vdim * numChannels );
    std::vector<TextureDescriptor>                        descriptors( images.size(), makeDescriptor() );
    for( uint32_t cell = 0; cell < udim * vdim && !view.allCellsEmpty; ++cell )
    {
        if( cell == emptyCell )
            continue;
        const fs::path& cellImage = cellImages[cell % cellImages.size()];
        for( uint32_t channel = 0; channel < numChannels; ++channel )
            images[cell * numChannels + channel] = readChannelImage( cellImage, channel );
    }

    uint32_t textureId = loader->createUdimTexture( images, descriptors, udim, vdim, numChannels ).getId();
    uint32_t channel   = view.channel;
    uint32_t renderWidth = outputWidth, renderHeight = outputHeight, gridU = udim, gridV = vdim;
    float    footprintScale = view.footprintScale;
    const uint32_t gridWidth  = ( outputWidth + blockSize - 1 ) / blockSize;
    const uint32_t gridHeight = ( outputHeight + blockSize - 1 ) / blockSize;

    std::vector<uint8_t> pixels( static_cast<size_t>( outputWidth ) * outputHeight * outputChannels );
    const auto           readPixels = [&] {
        HIP_CHECK( hipMemcpyAsync( pixels.data(), gpu.output, pixels.size(), hipMemcpyDeviceToHost, gpu.stream ) );
        HIP_CHECK( hipStreamSynchronize( gpu.stream ) );
    };

    for( uint32_t pass = 1; pass <= maxPasses; ++pass )
    {
        DeviceContext context{};
        loader->launchPrepare( gpu.stream, context );
        void* arguments[] = { &context, &gpu.output, &renderWidth, &renderHeight, &textureId, &channel, &gridU, &gridV, &footprintScale };
        HIP_CHECK( hipModuleLaunchKernel( module.kernel(), gridWidth, gridHeight, 1, blockSize, blockSize, 1, 0,
                                          gpu.stream, arguments, nullptr ) );

        Ticket ticket = loader->processRequests( gpu.stream, context );
        ticket.wait();
        readPixels();

        // Pixels whose sample was not final after this pass.
        const uint32_t provisionalPixels = countPixels( pixels, 0 );
        std::cout << "  Pass " << pass << ": " << ticket.numTasksTotal() << " requests, " << provisionalPixels
                  << " provisional pixels\n";
        if( ticket.numTasksTotal() == 0 && provisionalPixels == 0 )
            break;
    }

    // Alpha holds residency; write the image opaque.
    for( size_t i = 3; i < pixels.size(); i += outputChannels )
        pixels[i] = 255;
    if( stbi_write_png( outputPath.string().c_str(), static_cast<int>( outputWidth ), static_cast<int>( outputHeight ),
                        static_cast<int>( outputChannels ), pixels.data(), static_cast<int>( outputWidth * outputChannels ) )
        == 0 )
        throw std::runtime_error( "Failed to save output PNG: " + outputPath.string() );
    std::cout << "  Image saved: " << fs::absolute( outputPath ).string() << '\n';
}

}  // namespace

int main( int argc, char** argv )
{
    try
    {
        const fs::path executableDir = fs::absolute( fs::path{ argv[0] } ).parent_path();
        const fs::path kernelPath    = executableDir / "vmm_loader_udim_kernel.co";
        if( !fs::exists( kernelPath ) )
            throw std::runtime_error( "HIP module not found: " + kernelPath.string() );

        fs::path imageDirectory = fs::path{ TEST_IMAGES_DIR } / "png";
        imageDirectory.make_preferred();

        std::cout << "VMM loader UDIM texture\n  Image directory: " << imageDirectory.string() << '\n';
        const std::vector<fs::path> imagePaths = findImages( imageDirectory, false );
        std::cout << "  Grid: " << udim << 'x' << vdim << ", cell " << emptyCell << " empty\n";
        for( size_t i = 0; i < imagePaths.size(); ++i )
            std::cout << "  Cell image " << i << ": " << imagePaths[i].filename().string() << '\n';

        const std::array<View, 4> views{ { { "grid", 1.0f, 0 }, { "far", 64.0f, 0 }, { "inverted", 1.0f, 1 }, { "all_empty", 1.0f, 0, true } } };
        for( const View& view : views )
        {
            std::cout << "\nView " << view.name << " (footprint x" << view.footprintScale << ", channel " << view.channel << ")\n";
            const fs::path outputPath = fs::path( "vmm_loader_udim_" ) += fs::path( view.name ).concat( ".png" );
            renderView( kernelPath, imagePaths, view, outputPath );
        }

        std::cout << "\nResult: PASS\n";
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr << "\nResult: FAIL - " << error.what() << '\n';
        return 1;
    }
}
