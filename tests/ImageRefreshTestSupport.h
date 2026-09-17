// SPDX-License-Identifier: MIT
#pragma once
#include <cmath>
#include <hip/hip_runtime.h>

// Public ImageSource headers require HIP and math types to be declared first.
#include "ImageRefreshFixtures.h"
#include "TestPaths.h"
#include <DemandLoading/Contracts.h>
#include <ImageSource/ImageSource.h>
#include <ImageSource/TextureInfo.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iterator>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace hip_demand {
namespace test {
namespace refresh {
using Bytes = std::vector<uint8_t>;
using Pixel = std::array<double, 4>;
inline std::atomic<unsigned> liveSources{ 0 }, liveOwners{ 0 };
inline std::atomic<uint64_t> nextSource{ 1 }, nextOwner{ 1 };

inline uint64_t digest( const Bytes& bytes )
{
    uint64_t hash = 14695981039346656037ULL;
    for( uint8_t b : bytes )
    {
        hash ^= b;
        hash *= 1099511628211ULL;
    }
    return hash;
}
inline std::string hex( const Bytes& bytes )
{
    std::ostringstream out;
    out << std::hex << std::setfill( '0' );
    for( uint8_t b : bytes )
        out << std::setw( 2 ) << unsigned( b );
    return out.str();
}
inline void u32( Bytes& b, uint32_t v )
{
    for( unsigned shift = 0; shift < 32; shift += 8 )
        b.push_back( uint8_t( v >> shift ) );
}
inline uint32_t readU32( const Bytes& b, size_t& p )
{
    if( p + 4 > b.size() )
        throw std::runtime_error( "truncated refresh header" );
    uint32_t result = 0;
    for( unsigned shift = 0; shift < 32; shift += 8 )
        result |= uint32_t( b[p++] ) << shift;
    return result;
}
inline Bytes readFile( const std::filesystem::path& file )
{
    std::ifstream in( file, std::ios::binary );
    if( !in )
        throw std::runtime_error( "refresh file cannot be opened" );
    Bytes result{ std::istreambuf_iterator<char>( in ), std::istreambuf_iterator<char>() };
    if( in.bad() )
        throw std::runtime_error( "refresh file read failed" );
    return result;
}
inline void writeFile( const std::filesystem::path& file, const Bytes& bytes )
{
    std::ofstream out( file, std::ios::binary | std::ios::trunc );
    if( !out )
        throw std::runtime_error( "refresh file cannot be written" );
    out.write( reinterpret_cast<const char*>( bytes.data() ), bytes.size() );
    out.flush();
    if( !out )
        throw std::runtime_error( "refresh file flush failed" );
    out.close();
    if( !out )
        throw std::runtime_error( "refresh file close failed" );
    if( readFile( file ) != bytes )
        throw std::runtime_error( "refresh write/read bytes differ" );
    std::cout << "REFRESH_FILE path=" << file.string() << " fnv=" << digest( bytes ) << " bytes=" << bytes.size()
              << " hex=" << hex( bytes ) << '\n';
}
struct Image
{
    uint32_t           width = 8, height = 8;
    bool               floating = false;
    std::vector<Bytes> levels;
};
inline Image solid( bool green, unsigned width = 8, unsigned height = 8, bool authored = false, bool floating = false )
{
    Image          image{ width, height, floating, {} };
    const unsigned count = authored ? contract_v1::fullMipCount( width, height ) : 1;
    for( unsigned m = 0; m < count; ++m )
    {
        Bytes level;
        for( unsigned i = 0; i < contract_v1::mipDimension( width, m ) * contract_v1::mipDimension( height, m ); ++i )
        {
            if( floating )
            {
                for( float v : green ? image_refresh_data::greenFloat : image_refresh_data::redFloat )
                {
                    uint32_t bits;
                    std::memcpy( &bits, &v, 4 );
                    u32( level, bits );
                }
            }
            else
            {
                const auto& p = authored ?
                                    ( green ? image_refresh_data::greenAuthored[m] : image_refresh_data::redAuthored[m] ) :
                                    ( green ? image_refresh_data::green : image_refresh_data::red );
                level.insert( level.end(), p.begin(), p.end() );
            }
        }
        image.levels.push_back( std::move( level ) );
    }
    return image;
}
inline Image pattern( bool green )
{
    const auto& bytes = green ? image_refresh_data::greenPattern : image_refresh_data::redPattern;
    return { 4, 4, false, { Bytes( bytes.begin(), bytes.end() ) } };
}
inline Bytes tga( const Image& image )
{
    if( image.floating || image.levels.size() != 1 )
        throw std::runtime_error( "TGA fixture is RGBA8 base only" );
    Bytes result( 18, 0 );
    result[2]     = 2;
    result[12]    = uint8_t( image.width );
    result[13]    = uint8_t( image.width >> 8 );
    result[14]    = uint8_t( image.height );
    result[15]    = uint8_t( image.height >> 8 );
    result[16]    = 32;
    result[17]    = 0x28;
    const auto& b = image.levels[0];
    for( size_t p = 0; p < b.size(); p += 4 )
        result.insert( result.end(), { b[p + 2], b[p + 1], b[p], b[p + 3] } );
    return result;
}
inline Bytes snapshotFile( const Image& image )
{
    Bytes result{ 'H', 'D', 'T', 'R', 'E', 'F', '1', 0 };
    u32( result, image.width );
    u32( result, image.height );
    u32( result, image.floating ? 1 : 0 );
    u32( result, uint32_t( image.levels.size() ) );
    for( const auto& level : image.levels )
        result.insert( result.end(), level.begin(), level.end() );
    return result;
}
inline Image decodeSnapshot( const Bytes& bytes )
{
    const Bytes magic{ 'H', 'D', 'T', 'R', 'E', 'F', '1', 0 };
    if( bytes.size() < 24 || !std::equal( magic.begin(), magic.end(), bytes.begin() ) )
        throw std::runtime_error( "invalid refresh snapshot header" );
    size_t offset = 8;
    Image  image;
    image.width         = readU32( bytes, offset );
    image.height        = readU32( bytes, offset );
    const unsigned kind = readU32( bytes, offset ), count = readU32( bytes, offset );
    if( !image.width || !image.height || image.width > 4096 || image.height > 4096 || kind > 1 || !count
        || count > contract_v1::fullMipCount( image.width, image.height ) )
        throw std::runtime_error( "invalid refresh snapshot metadata" );
    image.floating = kind == 1;
    for( unsigned m = 0; m < count; ++m )
    {
        const size_t n = size_t( contract_v1::mipDimension( image.width, m ) )
                         * contract_v1::mipDimension( image.height, m ) * ( image.floating ? 16 : 4 );
        if( n > bytes.size() - offset )
            throw std::runtime_error( "truncated refresh snapshot pixels" );
        Bytes level( bytes.begin() + offset, bytes.begin() + offset + n );
        offset += n;
        if( image.floating )
        {
            size_t p = 0;
            while( p < level.size() )
            {
                const uint32_t bits = readU32( level, p );
                float          v;
                std::memcpy( &v, &bits, 4 );
                std::memcpy( level.data() + p - 4, &v, 4 );
            }
        }
        image.levels.push_back( std::move( level ) );
    }
    if( offset != bytes.size() )
        throw std::runtime_error( "trailing refresh snapshot data" );
    return image;
}

// Independent point oracle: authored bytes are retained; only missing POT tails
// use explicit 2x2 sums with the frozen unsigned-byte half-up rule.
inline std::vector<Pixel> expected( const Image& image, unsigned requested )
{
    std::vector<Pixel> result;
    unsigned           previousWidth = image.width, previousHeight = image.height;
    for( unsigned m = 0; m <= requested; ++m )
    {
        const unsigned w = contract_v1::mipDimension( image.width, m ), h = contract_v1::mipDimension( image.height, m );
        std::vector<Pixel> next( w * h );
        if( m < image.levels.size() )
        {
            for( unsigned i = 0; i < w * h; ++i )
                for( unsigned c = 0; c < 4; ++c )
                {
                    if( image.floating )
                    {
                        float v;
                        std::memcpy( &v, image.levels[m].data() + 4 * ( 4 * i + c ), 4 );
                        next[i][c] = v;
                    }
                    else
                        next[i][c] = image.levels[m][4 * i + c];
                }
        }
        else
        {
            for( unsigned y = 0; y < h; ++y )
                for( unsigned x = 0; x < w; ++x )
                    for( unsigned c = 0; c < 4; ++c )
                    {
                        double   sum     = 0;
                        unsigned samples = 0;
                        for( unsigned yy = 2 * y; yy < std::min( 2 * y + 2, previousHeight ); ++yy )
                            for( unsigned xx = 2 * x; xx < std::min( 2 * x + 2, previousWidth ); ++xx )
                            {
                                sum += result[yy * previousWidth + xx][c];
                                ++samples;
                            }
                        next[y * w + x][c] = image.floating ? sum / samples : std::floor( sum / samples + .5 );
                    }
        }
        result         = std::move( next );
        previousWidth  = w;
        previousHeight = h;
    }
    if( !image.floating )
        for( auto& p : result )
            for( auto& c : p )
                c /= 255.;
    return result;
}

class FileSnapshot final : public ImageSource
{
    std::filesystem::path file_;
    bool                  failReads_;
    mutable std::mutex    mutex_;
    std::atomic<bool>     opened_{ false };
    Image                 image_;
    Bytes                 bytes_;
    TextureInfo           info_{};
    std::atomic<unsigned> reads_{ 0 };

  public:
    const uint64_t identity = nextSource++;
    explicit FileSnapshot( std::filesystem::path file, bool failReads = false )
        : file_( std::move( file ) )
        , failReads_( failReads )
    {
        ++liveSources;
    }
    ~FileSnapshot() override { --liveSources; }
    void open( TextureInfo* info ) override
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        if( bytes_.empty() )
        {
            const auto bytes   = readFile( file_ );
            const auto image   = decodeSnapshot( bytes );
            image_             = image;
            bytes_             = bytes;
            info_.width        = image_.width;
            info_.height       = image_.height;
            info_.numChannels  = 4;
            info_.numMipLevels = unsigned( image_.levels.size() );
            info_.format       = image_.floating ? HIP_AD_FORMAT_FLOAT : HIP_AD_FORMAT_UNSIGNED_INT8;
            info_.isValid      = true;
            std::cout << "REFRESH_SOURCE identity=" << identity << " fnv=" << digest( bytes_ )
                      << " bytes=" << bytes_.size() << '\n';
        }
        opened_ = true;
        if( info )
            *info = info_;
    }
    void               close() override { opened_ = false; }
    bool               isOpen() const override { return opened_; }
    const TextureInfo& getInfo() const override { return info_; }
    bool               readMipLevel( char* dest, unsigned m, unsigned w, unsigned h, hipStream_t ) override
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        ++reads_;
        if( failReads_ || !opened_ || m >= image_.levels.size() || w != contract_v1::mipDimension( info_.width, m )
            || h != contract_v1::mipDimension( info_.height, m ) )
            return false;
        std::memcpy( dest, image_.levels[m].data(), image_.levels[m].size() );
        return true;
    }
    bool               readBaseColor( float4& ) override { return false; }
    unsigned long long getNumBytesRead() const override
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        return bytes_.size();
    }
    double             getTotalReadTime() const override { return 0; }
    unsigned long long getHash( hipStream_t = 0 ) const override { return 0; }
    uint64_t           contentIdentity() const
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        return digest( bytes_ );
    }
    unsigned reads() const { return reads_; }
};
}  // namespace refresh
}  // namespace test
}  // namespace hip_demand
