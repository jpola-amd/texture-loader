#include <hip/hip_runtime.h>
#include "ImageSource/OIIOReader.h"

#include <OpenImageIO/imageio.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace hip_demand {

namespace {

inline hipArray_Format getHipFormat( OIIO::TypeDesc::BASETYPE format )
{
    switch( format )
    {
        case OIIO::TypeDesc::UINT8:
            return HIP_AD_FORMAT_UNSIGNED_INT8;
        case OIIO::TypeDesc::INT8:
            return HIP_AD_FORMAT_SIGNED_INT8;
        case OIIO::TypeDesc::UINT16:
            return HIP_AD_FORMAT_UNSIGNED_INT16;
        case OIIO::TypeDesc::INT16:
            return HIP_AD_FORMAT_SIGNED_INT16;
        case OIIO::TypeDesc::UINT32:
            return HIP_AD_FORMAT_UNSIGNED_INT32;
        case OIIO::TypeDesc::INT32:
            return HIP_AD_FORMAT_SIGNED_INT32;
        case OIIO::TypeDesc::HALF:
            return HIP_AD_FORMAT_HALF;
        case OIIO::TypeDesc::FLOAT:
            return HIP_AD_FORMAT_FLOAT;
        default:
            return HIP_AD_FORMAT_FLOAT;
    }
}

inline OIIO::TypeDesc getOiioFormat( hipArray_Format format )
{
    switch( format )
    {
        case HIP_AD_FORMAT_UNSIGNED_INT8:
            return OIIO::TypeDesc::UINT8;
        case HIP_AD_FORMAT_SIGNED_INT8:
            return OIIO::TypeDesc::INT8;
        case HIP_AD_FORMAT_UNSIGNED_INT16:
            return OIIO::TypeDesc::UINT16;
        case HIP_AD_FORMAT_SIGNED_INT16:
            return OIIO::TypeDesc::INT16;
        case HIP_AD_FORMAT_UNSIGNED_INT32:
            return OIIO::TypeDesc::UINT32;
        case HIP_AD_FORMAT_SIGNED_INT32:
            return OIIO::TypeDesc::INT32;
        case HIP_AD_FORMAT_HALF:
            return OIIO::TypeDesc::HALF;
        case HIP_AD_FORMAT_FLOAT:
            return OIIO::TypeDesc::FLOAT;
        default:
            return OIIO::TypeDesc::UNKNOWN;
    }
}

class Stopwatch
{
  public:
    double elapsed() const { return std::chrono::duration<double>( std::chrono::steady_clock::now() - start_ ).count(); }

  private:
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
};

// Selects the OpenEXR Core backend, which lets threads read tiles of one EXR file in parallel. The setting is global
// and is read when a file is opened, so it is made once up front. Core needs OpenEXR 3.1.10; an older one is reported.
void preferExrCoreReader()
{
    static std::once_flag once;
    std::call_once( once, [] {
        OIIO::attribute( "openexr:core", 1 );

        std::string libs = OIIO::get_string_attribute( "library_list" );
        std::transform( libs.begin(), libs.end(), libs.begin(), []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );

        // Entries look like "openexr:openexr 3.3.2" and are separated by ';'.
        const std::string key     = "openexr:";
        const size_t      pos     = libs.find( key );
        std::string       version = "unknown";
        if( pos != std::string::npos )
        {
            const size_t      start = pos + key.size();
            const size_t      end   = libs.find( ';', start );
            const std::string entry = libs.substr( start, end == std::string::npos ? std::string::npos : end - start );
            const size_t      digit = entry.find_first_of( "0123456789" );
            if( digit != std::string::npos )
            {
                version = entry.substr( digit );
                int major = 0, minor = 0, patch = 0;
                if( std::sscanf( version.c_str(), "%d.%d.%d", &major, &minor, &patch ) >= 2
                    && ( major > 3 || ( major == 3 && ( minor > 1 || ( minor == 1 && patch >= 10 ) ) ) ) )
                    return;
            }
        }
        std::cerr << "OIIOReader: OpenEXR " << version << " has no Core backend (needs 3.1.10); EXR tile reads will serialize.\n";
    } );
}

}  // namespace

struct OIIOReader::File
{
    std::unique_ptr<OIIO::ImageInput> input;
    OIIO::TypeDesc                    pixelFormat;  // Format of the pixels in the file, the same on every level.
    OIIO::TypeDesc                    readFormat;   // Format OIIO converts to while reading; pixelFormat unless unsupported.
};

OIIOReader::OIIOReader( const std::string& filename, bool readBaseColor )
    : filename_( filename )
    , readBaseColor_( readBaseColor )
{
}

OIIOReader::~OIIOReader()
{
    close();
}

void OIIOReader::open( TextureInfo* info )
{
    preferExrCoreReader();

    std::lock_guard<std::mutex> lock( mutex_ );

    if( !file_ )
    {
        auto file   = std::make_shared<File>();
        file->input = OIIO::ImageInput::open( filename_ );
        if( !file->input )
            throw std::runtime_error( "Failed to open image " + filename_ + ": " + OIIO::geterror() );
        OIIO::ImageInput& input = *file->input;

        const OIIO::ImageSpec spec = input.spec_dimensions( 0, 0 );
        if( spec.width <= 0 || spec.height <= 0 || spec.depth != 1 || spec.nchannels < 1 || spec.nchannels > 4 )
            throw std::runtime_error( "Unsupported image dimensions or channel count: " + filename_ );

        TextureInfo textureInfo{};
        textureInfo.width       = static_cast<uint32_t>( spec.width );
        textureInfo.height      = static_cast<uint32_t>( spec.height );
        textureInfo.numChannels = static_cast<uint32_t>( spec.nchannels );
        textureInfo.format      = getHipFormat( static_cast<OIIO::TypeDesc::BASETYPE>( spec.format.basetype ) );
        textureInfo.isTiled     = spec.tile_width > 0 && spec.tile_height > 0;
        textureInfo.isValid     = true;

        file->pixelFormat = spec.format;
        file->readFormat  = getOiioFormat( textureInfo.format );
        bytesPerPixel_    = static_cast<uint32_t>( file->readFormat.size() ) * textureInfo.numChannels;
        if( file->readFormat == OIIO::TypeDesc::UNKNOWN || bytesPerPixel_ == 0 )
            throw std::runtime_error( "Unsupported pixel format " + std::string( spec.format.c_str() ) + ": " + filename_ );

        std::vector<uint2> levelSizes, tileSizes;
        for( int level = 0;; ++level )
        {
            const OIIO::ImageSpec levelSpec = level == 0 ? spec : input.spec_dimensions( 0, level );
            if( levelSpec.format == OIIO::TypeDesc::UNKNOWN || levelSpec.width <= 0 || levelSpec.height <= 0 )
                break;
            // HIP halves level dimensions rounding down. A file may round up instead (EXR allows it), and its levels
            // from the first mismatch on would be sampled with the wrong size, so only the matching prefix is kept.
            if( levelSpec.width != std::max( 1, spec.width >> level ) || levelSpec.height != std::max( 1, spec.height >> level ) )
                break;
            if( levelSpec.format != file->pixelFormat || levelSpec.nchannels != spec.nchannels || levelSpec.depth != 1 )
                throw std::runtime_error( "Mip levels differ in pixel format, channel count or depth: " + filename_ );

            levelSizes.push_back( { static_cast<uint32_t>( levelSpec.width ), static_cast<uint32_t>( levelSpec.height ) } );
            tileSizes.push_back( { static_cast<uint32_t>( std::max( 0, levelSpec.tile_width ) ),
                                   static_cast<uint32_t>( std::max( 0, levelSpec.tile_height ) ) } );
        }
        textureInfo.numMipLevels = static_cast<uint32_t>( levelSizes.size() );

        file_         = std::move( file );
        info_         = textureInfo;
        levelSizes_   = std::move( levelSizes );
        tileSizes_    = std::move( tileSizes );

        // The base color is the 1x1 level, which exists when the mip chain is complete or the image itself is 1x1.
        // OIIO converts the pixel to float while reading, normalizing integer formats.
        const uint32_t lastLevel = info_.numMipLevels - 1;
        if( readBaseColor_ && levelSizes_[lastLevel].x == 1 && levelSizes_[lastLevel].y == 1 )
        {
            float channels[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
            if( !input.read_image( 0, static_cast<int>( lastLevel ), 0, spec.nchannels, OIIO::TypeDesc::FLOAT, channels ) )
                throw std::runtime_error( "Failed to read the base color of " + filename_ + ": " + input.geterror() );
            baseColor_        = float4{ channels[0], channels[1], channels[2], channels[3] };
            baseColorWasRead_ = true;
        }
    }

    if( info )
        *info = info_;
}

void OIIOReader::close()
{
    // Only the reference is dropped; the OIIO input closes the file in its destructor, which runs once the last
    // in-flight read releases its copy.
    std::lock_guard<std::mutex> lock( mutex_ );
    file_.reset();
    baseColorWasRead_ = false;
}

bool OIIOReader::isOpen() const
{
    std::lock_guard<std::mutex> lock( mutex_ );
    return static_cast<bool>( file_ );
}

const TextureInfo& OIIOReader::getInfo() const
{
    return info_;
}

std::shared_ptr<OIIOReader::File> OIIOReader::acquireFile() const
{
    std::lock_guard<std::mutex> lock( mutex_ );
    return file_;
}

void OIIOReader::addStats( unsigned long long tiles, unsigned long long bytes, double seconds )
{
    std::lock_guard<std::mutex> lock( statsMutex_ );
    numTilesRead_ += tiles;
    numBytesRead_ += bytes;
    totalReadTime_ += seconds;
}

bool OIIOReader::readTile( char* dest, uint32_t mipLevel, const Tile& tile, hipStream_t /*stream*/ )
{
    if( !dest || tile.width == 0 || tile.height == 0 )
        return false;

    const std::shared_ptr<File> file = acquireFile();
    if( !file || mipLevel >= info_.numMipLevels )
        return false;
    OIIO::ImageInput& input = *file->input;

    const Stopwatch stopwatch;

    const uint32_t levelWidth  = levelSizes_[mipLevel].x;
    const uint32_t levelHeight = levelSizes_[mipLevel].y;
    const uint32_t fileTileW   = tileSizes_[mipLevel].x;
    const uint32_t fileTileH   = tileSizes_[mipLevel].y;
    const uint32_t rowPitch    = tile.width * bytesPerPixel_;

    const uint32_t x0 = tile.x * tile.width;
    const uint32_t y0 = tile.y * tile.height;
    if( x0 >= levelWidth || y0 >= levelHeight )
        return false;
    const uint32_t x1 = std::min( x0 + tile.width, levelWidth );
    const uint32_t y1 = std::min( y0 + tile.height, levelHeight );

    // Pixels outside the level are zero. The reads fill only the in-bounds rectangle, so the tile is cleared first
    // when it crosses the right or bottom edge.
    if( x1 - x0 < tile.width || y1 - y0 < tile.height )
        std::memset( dest, 0, static_cast<size_t>( tile.height ) * rowPitch );

    unsigned long long bytesRead = 0;
    if( fileTileW != 0 && fileTileH != 0 )
    {
        // The requested tile must be a whole multiple of the file's tile so that one read_tiles() call covers it.
        if( fileTileW > tile.width || tile.width % fileTileW != 0 || fileTileH > tile.height || tile.height % fileTileH != 0 )
        {
            std::ostringstream message;
            message << "Unsupported file tile size " << fileTileW << 'x' << fileTileH << " for a " << tile.width << 'x'
                    << tile.height << " tile: the file tile must divide the requested tile (" << filename_ << ')';
            throw std::runtime_error( message.str() );
        }

        // The region is clamped to the level so that no nonexistent edge tiles are requested; OIIO places each file
        // tile at its destination offset through the strides.
        if( !input.read_tiles( 0, static_cast<int>( mipLevel ), static_cast<int>( x0 ), static_cast<int>( x1 ),
                               static_cast<int>( y0 ), static_cast<int>( y1 ), 0, 1, 0, static_cast<int>( info_.numChannels ),
                               file->readFormat, dest, bytesPerPixel_, rowPitch ) )
        {
            throw std::runtime_error( "Failed to read tile from " + filename_ + ": " + input.geterror() );
        }
        const unsigned long long fileTiles = static_cast<unsigned long long>( ( x1 - x0 + fileTileW - 1 ) / fileTileW )
                                             * ( ( y1 - y0 + fileTileH - 1 ) / fileTileH );
        bytesRead = fileTiles * fileTileW * fileTileH * file->pixelFormat.size() * info_.numChannels;
    }
    else
    {
        // Scanline file: the tile's rows are read one at a time and the tile's columns copied out of each.
        const size_t      filePixelBytes = file->pixelFormat.size() * info_.numChannels;
        std::vector<char> row( static_cast<size_t>( levelWidth ) * bytesPerPixel_ );
        for( uint32_t y = y0; y < y1; ++y )
        {
            if( !input.read_scanlines( 0, static_cast<int>( mipLevel ), static_cast<int>( y ), static_cast<int>( y + 1 ), 0,
                                       0, static_cast<int>( info_.numChannels ), file->readFormat, row.data() ) )
            {
                throw std::runtime_error( "Failed to read scanline from " + filename_ + ": " + input.geterror() );
            }
            std::memcpy( dest + static_cast<size_t>( y - y0 ) * rowPitch, row.data() + static_cast<size_t>( x0 ) * bytesPerPixel_,
                         static_cast<size_t>( x1 - x0 ) * bytesPerPixel_ );
        }
        bytesRead = static_cast<unsigned long long>( y1 - y0 ) * levelWidth * filePixelBytes;
    }

    addStats( 1, bytesRead, stopwatch.elapsed() );
    return true;
}

bool OIIOReader::readMipLevel( char* dest, uint32_t mipLevel, uint32_t expectedWidth, uint32_t expectedHeight, hipStream_t /*stream*/ )
{
    if( !dest )
        return false;

    const std::shared_ptr<File> file = acquireFile();
    if( !file || mipLevel >= info_.numMipLevels )
        return false;
    OIIO::ImageInput& input = *file->input;

    const uint32_t levelWidth  = levelSizes_[mipLevel].x;
    const uint32_t levelHeight = levelSizes_[mipLevel].y;
    if( levelWidth != expectedWidth || levelHeight != expectedHeight )
        return false;

    const Stopwatch stopwatch;
    if( !input.read_image( 0, static_cast<int>( mipLevel ), 0, static_cast<int>( info_.numChannels ), file->readFormat, dest ) )
        throw std::runtime_error( "Failed to read mip level " + std::to_string( mipLevel ) + " from " + filename_ + ": " + input.geterror() );

    const unsigned long long bytesRead = static_cast<unsigned long long>( levelWidth ) * levelHeight * file->pixelFormat.size() * info_.numChannels;
    addStats( 1, bytesRead, stopwatch.elapsed() );
    return true;
}

bool OIIOReader::readBaseColor( float4& dest )
{
    std::lock_guard<std::mutex> lock( mutex_ );
    dest = baseColor_;
    return baseColorWasRead_;
}

unsigned long long OIIOReader::getNumBytesRead() const
{
    std::lock_guard<std::mutex> lock( statsMutex_ );
    return numBytesRead_;
}

double OIIOReader::getTotalReadTime() const
{
    std::lock_guard<std::mutex> lock( statsMutex_ );
    return totalReadTime_;
}

unsigned long long OIIOReader::getNumTilesRead() const
{
    std::lock_guard<std::mutex> lock( statsMutex_ );
    return numTilesRead_;
}

unsigned long long OIIOReader::getHash( hipStream_t /*stream*/ ) const
{
    return static_cast<unsigned long long>( std::hash<std::string>{}( filename_ ) );
}

std::unique_ptr<ImageSource> createImageSource( const std::string& filename )
{
    return std::make_unique<OIIOReader>( filename );
}

}  // namespace hip_demand
