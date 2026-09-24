#include <hip/hip_runtime.h>
#include "ImageSource/OIIOReader.h"
#include "../DemandLoading/Internal/ImageData.h"
#include <OpenImageIO/half.h>
#include <OpenImageIO/imageio.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace hip_demand {

namespace {

hipArray_Format getHipFormat( OIIO::TypeDesc::BASETYPE format )
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
            // HIP textures do not support OIIO's 64-bit integer and double
            // formats, so retain their dynamic range by converting to float.
            return HIP_AD_FORMAT_FLOAT;
    }
}

OIIO::TypeDesc getOiioFormat( hipArray_Format format )
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

template <typename T>
T loadSample( const unsigned char* data )
{
    T value{};
    std::memcpy( &value, data, sizeof( value ) );
    return value;
}

template <>
Imath::half loadSample<Imath::half>( const unsigned char* data )
{
    uint16_t bits{};
    std::memcpy( &bits, data, sizeof( bits ) );
    return Imath::half( Imath::half::FromBits, bits );
}

template <typename T>
void storeSample( unsigned char* data, T value )
{
    std::memcpy( data, &value, sizeof( value ) );
}

template <>
void storeSample( unsigned char* data, Imath::half value )
{
    const uint16_t bits = value.bits();
    std::memcpy( data, &bits, sizeof( bits ) );
}

template <typename T>
double sampleToDouble( T value )
{
    return static_cast<double>( value );
}

template <>
double sampleToDouble( Imath::half value )
{
    return static_cast<float>( value );
}

template <typename T>
T sampleFromDouble( double value )
{
    return static_cast<T>( value );
}

template <>
Imath::half sampleFromDouble( double value )
{
    return Imath::half( static_cast<float>( value ) );
}

template <typename T>
void generateMipLevelTyped( const unsigned char* srcData, int srcWidth, int srcHeight, unsigned char* dstData,
                            int dstWidth, int dstHeight, int channels )
{
    constexpr size_t SAMPLE_BYTES = sizeof( T );

    for( int y = 0; y < dstHeight; ++y )
    {
        for( int x = 0; x < dstWidth; ++x )
        {
            const int srcX = x * 2;
            const int srcY = y * 2;

            for( int channel = 0; channel < channels; ++channel )
            {
                double sum   = 0.0;
                int    count = 0;

                for( int dy = 0; dy < 2 && srcY + dy < srcHeight; ++dy )
                {
                    for( int dx = 0; dx < 2 && srcX + dx < srcWidth; ++dx )
                    {
                        const size_t srcSample =
                            ( ( static_cast<size_t>( srcY + dy ) * srcWidth + srcX + dx ) * channels + channel )
                            * SAMPLE_BYTES;
                        sum += sampleToDouble( loadSample<T>( srcData + srcSample ) );
                        ++count;
                    }
                }

                const size_t dstSample =
                    ( ( static_cast<size_t>( y ) * dstWidth + x ) * channels + channel ) * SAMPLE_BYTES;
                storeSample( dstData + dstSample, sampleFromDouble<T>( sum / count ) );
            }
        }
    }
}

float decodeChannel( const unsigned char* data, hipArray_Format format )
{
    switch( format )
    {
        case HIP_AD_FORMAT_UNSIGNED_INT8:
            return loadSample<uint8_t>( data ) / 255.0f;
        case HIP_AD_FORMAT_SIGNED_INT8:
            return std::max( -1.0f, loadSample<int8_t>( data ) / 127.0f );
        case HIP_AD_FORMAT_UNSIGNED_INT16:
            return loadSample<uint16_t>( data ) / 65535.0f;
        case HIP_AD_FORMAT_SIGNED_INT16:
            return std::max( -1.0f, loadSample<int16_t>( data ) / 32767.0f );
        case HIP_AD_FORMAT_UNSIGNED_INT32:
            return static_cast<float>( loadSample<uint32_t>( data ) ) / 4294967295.0f;
        case HIP_AD_FORMAT_SIGNED_INT32:
            return std::max( -1.0f, static_cast<float>( loadSample<int32_t>( data ) ) / 2147483647.0f );
        case HIP_AD_FORMAT_HALF:
            return static_cast<float>( loadSample<Imath::half>( data ) );
        case HIP_AD_FORMAT_FLOAT:
            return loadSample<float>( data );
        default:
            return 0.0f;
    }
}

}  // namespace

OIIOReader::OIIOReader( const std::string& filename, std::optional<hipArray_Format> outputFormat )
    : filename_( filename )
    , outputFormat_( outputFormat )
{
}

OIIOReader::~OIIOReader()
{
    close();
}

void OIIOReader::open( TextureInfo* info )
{
    std::unique_lock lock( mutex_ );

    if( isOpen_ )
    {
        if( info )
            *info = info_;
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Open image with OIIO
    auto inp = OIIO::ImageInput::open( filename_ );
    if( !inp )
    {
        throw std::runtime_error( "Failed to open image: " + filename_ );
    }

    const OIIO::ImageSpec& spec = inp->spec();
    if( spec.width <= 0 || spec.height <= 0 || spec.depth != 1 || spec.nchannels < 1 || spec.nchannels > 4 )
    {
        inp->close();
        throw std::runtime_error( "Unsupported OIIO texture dimensions or channels: " + filename_ );
    }

    // Fill texture info
    info_.width        = spec.width;
    info_.height       = spec.height;
    info_.numChannels  = spec.nchannels;
    info_.numMipLevels = calculateNumMipLevels( spec.width, spec.height );
    info_.isValid      = true;
    info_.isTiled      = spec.tile_width > 0;

    const hipArray_Format sourceFormat =
        getHipFormat( static_cast<OIIO::TypeDesc::BASETYPE>( spec.format.basetype ) );
    info_.format = outputFormat_.value_or( sourceFormat );
    if( getOiioFormat( info_.format ) == OIIO::TypeDesc::UNKNOWN )
    {
        inp->close();
        throw std::invalid_argument( "Unsupported requested HIP array format: "
                                     + std::to_string( static_cast<int>( info_.format ) ) );
    }

    inp->close();

    isOpen_ = true;

    if( info )
        *info = info_;

    auto end = std::chrono::high_resolution_clock::now();
    totalReadTime_ += std::chrono::duration<double>( end - start ).count();
}

void OIIOReader::close()
{
    std::unique_lock lock( mutex_ );

    if( !isOpen_ )
        return;

    mipLevels_.clear();
    isOpen_ = false;
}

bool OIIOReader::isOpen() const
{
    std::shared_lock lock( mutex_ );
    return isOpen_;
}

const TextureInfo& OIIOReader::getInfo() const
{
    return info_;
}

bool OIIOReader::loadImage()
{
    if( !isOpen_ )
        return false;

    auto start = std::chrono::high_resolution_clock::now();

    // Open image
    auto inp = OIIO::ImageInput::open( filename_ );
    if( !inp )
        return false;

    const OIIO::ImageSpec& spec = inp->spec();

    // Read base level (level 0)
    const size_t bytesPerChannel = getBytesPerChannel( info_.format );
    const size_t baseSize = internal::imageByteSize( spec.width, spec.height, info_.numChannels, bytesPerChannel );
    std::vector<unsigned char> baseLevel( baseSize );

    const OIIO::TypeDesc outputFormat = getOiioFormat( info_.format );
    if( outputFormat == OIIO::TypeDesc::UNKNOWN )
    {
        inp->close();
        return false;
    }

    const bool success = inp->read_image( 0, 0, 0, spec.nchannels, outputFormat, baseLevel.data() );

    if( !success )
    {
        inp->close();
        return false;
    }

    // Store base level
    mipLevels_.resize( info_.numMipLevels );
    mipLevels_[0] = std::move( baseLevel );

    // Keep mip levels supplied by the file; generate any missing levels for
    // VMM tile sampling.
    int width  = spec.width;
    int height = spec.height;
    unsigned long long bytesRead = baseSize;

    for( unsigned int level = 1; level < info_.numMipLevels; ++level )
    {
        int prevWidth  = width;
        int prevHeight = height;
        width          = std::max( 1, width / 2 );
        height         = std::max( 1, height / 2 );

        const size_t levelSize = internal::imageByteSize( width, height, info_.numChannels, bytesPerChannel );
        mipLevels_[level].resize( levelSize );

        OIIO::ImageSpec mipSpec;
        const bool hasNativeMip = inp->seek_subimage( 0, static_cast<int>( level ), mipSpec )
            && mipSpec.width == width && mipSpec.height == height && mipSpec.depth == 1
            && mipSpec.nchannels == static_cast<int>( info_.numChannels );
        if( hasNativeMip )
        {
            if( !inp->read_image( 0, static_cast<int>( level ), 0, mipSpec.nchannels, outputFormat,
                                  mipLevels_[level].data() ) )
            {
                inp->close();
                mipLevels_.clear();
                return false;
            }
            bytesRead += levelSize;
        }
        else
        {
            generateMipLevel( mipLevels_[level - 1].data(), prevWidth, prevHeight, mipLevels_[level].data(), width,
                              height, info_.numChannels );
        }
    }

    inp->close();
    bytesRead_ += bytesRead;

    auto end = std::chrono::high_resolution_clock::now();
    totalReadTime_ += std::chrono::duration<double>( end - start ).count();

    return true;
}

void OIIOReader::generateMipLevel( const unsigned char* srcData, int srcWidth, int srcHeight, unsigned char* dstData,
                                   int dstWidth, int dstHeight, int channels )
{
    switch( info_.format )
    {
        case HIP_AD_FORMAT_UNSIGNED_INT8:
            generateMipLevelTyped<uint8_t>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
            return;
        case HIP_AD_FORMAT_SIGNED_INT8:
            generateMipLevelTyped<int8_t>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
            return;
        case HIP_AD_FORMAT_UNSIGNED_INT16:
            generateMipLevelTyped<uint16_t>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
            return;
        case HIP_AD_FORMAT_SIGNED_INT16:
            generateMipLevelTyped<int16_t>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
            return;
        case HIP_AD_FORMAT_UNSIGNED_INT32:
            generateMipLevelTyped<uint32_t>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
            return;
        case HIP_AD_FORMAT_SIGNED_INT32:
            generateMipLevelTyped<int32_t>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
            return;
        case HIP_AD_FORMAT_HALF:
            generateMipLevelTyped<Imath::half>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
            return;
        case HIP_AD_FORMAT_FLOAT:
            generateMipLevelTyped<float>( srcData, srcWidth, srcHeight, dstData, dstWidth, dstHeight, channels );
            return;
        default:
            throw std::runtime_error( "Cannot generate mip level for unsupported HIP array format: "
                                      + std::to_string( static_cast<int>( info_.format ) ) );
    }
}

bool OIIOReader::readMipLevelNoLock( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight )
{
    assert( dest );

    if( mipLevel >= mipLevels_.size() )
        return false;

    unsigned int w = info_.width >> mipLevel;
    unsigned int h = info_.height >> mipLevel;
    w              = std::max( 1u, w );
    h              = std::max( 1u, h );

    if( w != expectedWidth || h != expectedHeight )
        return false;

    // Copy data
    const size_t bytesPerTexel = static_cast<size_t>( getBytesPerChannel( info_.format ) ) * info_.numChannels;
    std::memcpy( dest, mipLevels_[mipLevel].data(), bytesPerTexel * w * h );
    return true;
}

bool OIIOReader::readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, hipStream_t stream )
{
    for( ;; )
    {
        {
            std::shared_lock lock( mutex_ );
            if( !mipLevels_.empty() )
                return readMipLevelNoLock(dest, mipLevel, expectedWidth, expectedHeight);
        }

        {
            std::unique_lock lock( mutex_ );
            if( mipLevels_.empty() && !loadImage() )
                return false;
        }
    }
}

bool OIIOReader::readTileNoLock( char* dest, unsigned int mipLevel, const Tile& tile )
{
    assert( dest );

    if( mipLevel >= mipLevels_.size() )
        return false;

    const size_t mipWidth  = std::max( 1u, info_.width >> mipLevel );
    const size_t mipHeight = std::max( 1u, info_.height >> mipLevel );
    const size_t firstX    = static_cast<size_t>( tile.x ) * tile.width;
    const size_t firstY    = static_cast<size_t>( tile.y ) * tile.height;

    if( firstX >= mipWidth || firstY >= mipHeight )
        return false;

    const size_t bytesPerTexel = static_cast<size_t>( getBytesPerChannel( info_.format ) ) * info_.numChannels;
    if( bytesPerTexel == 0 )
        return false;

    const size_t copyWidth    = std::min( static_cast<size_t>( tile.width ), mipWidth - firstX );
    const size_t copyHeight   = std::min( static_cast<size_t>( tile.height ), mipHeight - firstY );
    const size_t tileRowBytes = static_cast<size_t>( tile.width ) * bytesPerTexel;

    const std::vector<unsigned char>& source = mipLevels_[mipLevel];
    if( source.size() < mipWidth * mipHeight * bytesPerTexel )
        return false;

    // Edge tiles are padded with zeroes so data left in a reused page buffer
    // cannot leak into the unused part of the tile.
    if( copyWidth < tile.width || copyHeight < tile.height )
        std::memset( dest, 0, static_cast<size_t>( tile.height ) * tileRowBytes );

    for( size_t row = 0; row < copyHeight; ++row )
    {
        const size_t sourceOffset = ( ( firstY + row ) * mipWidth + firstX ) * bytesPerTexel;
        const size_t destOffset   = row * tileRowBytes;
        std::memcpy( dest + destOffset, source.data() + sourceOffset, copyWidth * bytesPerTexel );
    }

    return true;
}

bool OIIOReader::readTile( char* dest, unsigned int mipLevel, const Tile& tile, hipStream_t stream )
{
    for( ;; )
    {
        {
            std::shared_lock lock( mutex_ );
            if( !mipLevels_.empty() )
                return readTileNoLock( dest, mipLevel, tile );
        }

        {
            std::unique_lock lock( mutex_ );
            if( mipLevels_.empty() && !loadImage() )
                return false;
        }
    }
}

bool OIIOReader::readBaseColor( float4& dest )
{
    for( ;; )
    {
        {
            std::shared_lock lock( mutex_ );
            if( !mipLevels_.empty() )
            {
                const unsigned int lastLevel       = info_.numMipLevels - 1;
                const size_t       bytesPerChannel = getBytesPerChannel( info_.format );
                const auto&        mip             = mipLevels_[lastLevel];
                if( bytesPerChannel == 0 || mip.size() < bytesPerChannel * info_.numChannels )
                    return false;

                const unsigned char* data = mip.data();
                dest.x                    = decodeChannel( data, info_.format );
                dest.y = info_.numChannels > 1 ? decodeChannel( data + bytesPerChannel, info_.format ) : dest.x;
                dest.z = info_.numChannels > 2 ? decodeChannel( data + 2 * bytesPerChannel, info_.format ) : dest.x;
                dest.w = info_.numChannels > 3 ? decodeChannel( data + 3 * bytesPerChannel, info_.format ) : 1.0f;
                return true;
            }
        }

        {
            std::unique_lock lock( mutex_ );
            if( mipLevels_.empty() && !loadImage() )
                return false;
        }
    }

    return true;
}

unsigned long long OIIOReader::getNumBytesRead() const
{
    std::shared_lock lock( mutex_ );
    return bytesRead_;
}

double OIIOReader::getTotalReadTime() const
{
    std::shared_lock lock( mutex_ );
    return totalReadTime_;
}

unsigned long long OIIOReader::getHash( hipStream_t /*stream*/ ) const
{
    // Hash the filename for content-based deduplication
    // Two ImageSource objects with the same filename should return the same hash
    return static_cast<unsigned long long>( std::hash<std::string>{}( filename_ ) );
}

std::unique_ptr<ImageSource> createImageSource( const std::string& filename )
{
    return std::make_unique<OIIOReader>( filename );
}

}  // namespace hip_demand
