#pragma once

#include <ImageSource/TextureInfo.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <string>

namespace hip_demand::vmm {

/// A white or black tile holds one texel throughout, whose color channels are all 0 or all 1 and whose alpha, when
/// there is one, is 0 or 1: 0 and 1 of one and two channels, 000 and 111 of three, 0000, 0001, 1110 and 1111 of four,
/// 1 is the value the sampler normalizes to 1.0: the largest value of an integer format. 1.0 of half and float.
/// Looks at the width x height texels at the top left of a tile of tileWidth texels per row: the whole tile, or the
/// part of an edge tile that lies inside its level. Returns whether they are one such texel, and then its key: the
/// texel's bytes.
inline bool whiteBlackTileKey( const uint8_t*  tile,
                               uint32_t        tileWidth,
                               uint32_t        visibleTileWidth,
                               uint32_t        visibleTileHeight,
                               hipArray_Format format,
                               uint32_t        numChannels,
                               std::string&    key )
{
    const uint32_t channelBytes  = getBytesPerChannel( format );
    const uint32_t bytesPerTexel = channelBytes * numChannels;
    if( !tile || bytesPerTexel == 0 || numChannels > 4 || visibleTileWidth == 0 || visibleTileHeight == 0
        || visibleTileWidth > tileWidth )
        return false;

    // The first row holds one texel exactly when it equals itself shifted by one texel.
    const size_t rowBytes = static_cast<size_t>( visibleTileWidth ) * bytesPerTexel;
    const size_t pitch    = static_cast<size_t>( tileWidth ) * bytesPerTexel;
    if( std::memcmp( tile, tile + bytesPerTexel, rowBytes - bytesPerTexel ) != 0 )
        return false;
    for( uint32_t row = 1; row < visibleTileHeight; ++row )
    {
        if( std::memcmp( tile + row * pitch, tile, rowBytes ) != 0 )
            return false;
    }

    uint8_t one[4] = {};
    switch( format )
    {
        case HIP_AD_FORMAT_UNSIGNED_INT8: { const uint8_t  value = UINT8_MAX;  std::memcpy( one, &value, sizeof( value ) ); break; }
        case HIP_AD_FORMAT_SIGNED_INT8:   { const int8_t   value = INT8_MAX;   std::memcpy( one, &value, sizeof( value ) ); break; }
        case HIP_AD_FORMAT_UNSIGNED_INT16:{ const uint16_t value = UINT16_MAX; std::memcpy( one, &value, sizeof( value ) ); break; }
        case HIP_AD_FORMAT_SIGNED_INT16:  { const int16_t  value = INT16_MAX;  std::memcpy( one, &value, sizeof( value ) ); break; }
        case HIP_AD_FORMAT_UNSIGNED_INT32:{ const uint32_t value = UINT32_MAX; std::memcpy( one, &value, sizeof( value ) ); break; }
        case HIP_AD_FORMAT_SIGNED_INT32:  { const int32_t  value = INT32_MAX;  std::memcpy( one, &value, sizeof( value ) ); break; }
        case HIP_AD_FORMAT_HALF:          { const uint16_t value = 0x3C00;     std::memcpy( one, &value, sizeof( value ) ); break; }  // 1.0h
        case HIP_AD_FORMAT_FLOAT:         { const float    value = 1.0f;       std::memcpy( one, &value, sizeof( value ) ); break; }
        default:
            return false;
    }

    // 0 or 1 for a channel of the texel, else -1.
    const auto channelValue = [&]( uint32_t channel ) {
        const uint8_t* bytes = tile + channel * channelBytes;
        const uint8_t  zero[4] = {};
        if( std::memcmp( bytes, zero, channelBytes ) == 0 )
            return 0;
        if( std::memcmp( bytes, one, channelBytes ) == 0 )
            return 1;
        return -1;
    };

    const int color = channelValue( 0 );
    if( color < 0 )
        return false;
    for( uint32_t channel = 1; channel < std::min( numChannels, 3u ); ++channel )
    {
        if( channelValue( channel ) != color )
            return false;
    }
    if( numChannels == 4 && channelValue( 3 ) < 0 )
        return false;

    key.assign( reinterpret_cast<const char*>( tile ), bytesPerTexel );
    return true;
}

}  // namespace hip_demand::vmm
