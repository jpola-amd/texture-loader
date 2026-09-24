#pragma once

#include <ImageSource/OIIOReader.h>
#include <ImageSource/TextureInfo.h>

#include "hip_check.h"
#include "stb_image_write.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace vmm_loader_example {

struct GpuResources
{
    explicit GpuResources( size_t outputBytes )
    {
        HIP_CHECK( hipStreamCreate( &stream ) );
        HIP_CHECK( hipMalloc( reinterpret_cast<void**>( &output ), outputBytes ) );
    }

    ~GpuResources()
    {
        HIP_WARN( hipStreamSynchronize( stream ) );
        HIP_WARN( hipFree( output ) );
        HIP_WARN( hipStreamDestroy( stream ) );
    }

    GpuResources( const GpuResources& )            = delete;
    GpuResources& operator=( const GpuResources& ) = delete;

    hipStream_t stream = nullptr;
    uint8_t*    output = nullptr;
};

inline void saveGpuOutput( const fs::path& outputPath, hipStream_t stream, const uint8_t* deviceOutput,
                           uint32_t width, uint32_t height, uint32_t channels )
{
    const size_t         byteCount = static_cast<size_t>( width ) * height * channels;
    std::vector<uint8_t> hostOutput( byteCount );
    HIP_CHECK( hipMemcpyAsync( hostOutput.data(), deviceOutput, byteCount, hipMemcpyDeviceToHost, stream ) );
    HIP_CHECK( hipStreamSynchronize( stream ) );

    if( stbi_write_png( outputPath.string().c_str(), static_cast<int>( width ), static_cast<int>( height ),
                        static_cast<int>( channels ), hostOutput.data(), static_cast<int>( width * channels ) )
        == 0 )
        throw std::runtime_error( "Failed to save output PNG: " + outputPath.string() );

    std::cout << "  Image saved: " << fs::absolute( outputPath ).string() << '\n';
}

inline std::shared_ptr<hip_demand::ImageSource> readImage( const fs::path& path )
{
    auto image = std::make_shared<hip_demand::OIIOReader>( path.string(), HIP_AD_FORMAT_UNSIGNED_INT8 );

    hip_demand::TextureInfo info{};
    image->open( &info );
    if( !info.isValid || info.width == 0 || info.height == 0 )
        throw std::runtime_error( "Invalid image: " + path.string() );

    return image;
}

inline std::vector<fs::path> findImages( const fs::path& directory, bool includeOver4k, bool recursive = false )
{
    constexpr uint32_t max4kDimension = 4096;

    if( !fs::is_directory( directory ) )
        throw std::runtime_error( "Image directory not found: " + directory.string() );

    std::vector<fs::path> imagePaths;
    const auto addImage = [&]( const fs::directory_entry& entry ) {
        if( !entry.is_regular_file() )
            return;

        std::string extension = entry.path().extension().string();
        std::transform( extension.begin(), extension.end(), extension.begin(),
                        []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );
        if( extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp"
            || extension == ".tga" || extension == ".tif" || extension == ".tiff" || extension == ".exr"
            || extension == ".hdr" )
            imagePaths.push_back( entry.path() );
    };
    if( recursive )
    {
        for( const fs::directory_entry& entry : fs::recursive_directory_iterator( directory ) )
            addImage( entry );
    }
    else
    {
        for( const fs::directory_entry& entry : fs::directory_iterator( directory ) )
            addImage( entry );
    }
    std::sort( imagePaths.begin(), imagePaths.end() );
    if( imagePaths.empty() )
        throw std::runtime_error( "No supported images found in: " + directory.string() );

    if( !includeOver4k )
    {
        std::vector<fs::path> selectedPaths;
        selectedPaths.reserve( imagePaths.size() );
        for( const fs::path& path : imagePaths )
        {
            const auto  image = readImage( path );
            const auto& info  = image->getInfo();
            if( info.width > max4kDimension || info.height > max4kDimension )
            {
                std::cout << "  Skipped (>4K): " << path.filename().string() << " (" << info.width << 'x'
                          << info.height << ")\n";
                continue;
            }
            selectedPaths.push_back( path );
        }
        imagePaths.swap( selectedPaths );
    }

    if( imagePaths.empty() )
        throw std::runtime_error( "No images match the configured size filter in: " + directory.string() );

    return imagePaths;
}

}  // namespace vmm_loader_example
