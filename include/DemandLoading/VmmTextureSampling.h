#pragma once

#include "DeviceContext.h"
#include "VmmDeviceUtils.h"
#include <algorithm>
#include <cmath>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <type_traits>


namespace hip_demand::vmm {

HIP_DEMAND_INLINE void recordRequest( const DeviceContext& context, uint32_t resourceId )
{
// #if defined( HIP_ENABLE_WARP_SYNC_BUILTINS )
//     const uint64_t activeLanes   = __activemask();
//     const uint64_t matchingLanes = __match_any_sync( activeLanes, resourceId );
//     const uint32_t leaderLane    = static_cast<uint32_t>( __ffsll( static_cast<long long>( matchingLanes ) ) - 1 );
//     if( static_cast<uint32_t>( __lane_id() ) != leaderLane )
//         return;
// #endif

    atomicSetBit( context.referenceBits, resourceId );
}
HIP_DEMAND_INLINE bool isResourceResident( const DeviceContext& context, uint32_t resourceId ) { return checkBitSet(context.residenceBits, resourceId); }

HIP_DEMAND_INLINE int wrapCoordinate( int coordinate, int extent )
{
    coordinate %= extent;
    return coordinate < 0 ? coordinate + extent : coordinate;
}

HIP_DEMAND_INLINE int mirrorCoordinate( int coordinate, int extent )
{
    const int period = extent * 2;
    coordinate %= period;
    if( coordinate < 0 )
        coordinate += period;
    return coordinate < extent ? coordinate : period - coordinate - 1;
}

HIP_DEMAND_INLINE int applyAddressMode( int coordinate, int extent, uint32_t mode )
{
    switch( static_cast<hipTextureAddressMode>( mode ) )
    {
        case hipAddressModeWrap:
            return wrapCoordinate( coordinate, extent );
        case hipAddressModeMirror:
            return mirrorCoordinate( coordinate, extent );
        case hipAddressModeClamp:
        default:
            return std::clamp( coordinate, 0, extent - 1 );
    }
}

HIP_DEMAND_INLINE int2 applyAddressModePair( int coordinate0, int coordinate1, int extent, uint32_t mode )
{
    switch( static_cast<hipTextureAddressMode>( mode ) )
    {
        case hipAddressModeWrap:
            return make_int2( wrapCoordinate( coordinate0, extent ), wrapCoordinate( coordinate1, extent ) );
        case hipAddressModeMirror:
            return make_int2( mirrorCoordinate( coordinate0, extent ), mirrorCoordinate( coordinate1, extent ) );
        case hipAddressModeClamp:
        default:
            return make_int2( std::clamp( coordinate0, 0, extent - 1 ), std::clamp( coordinate1, 0, extent - 1 ) );
    }
}

HIP_DEMAND_INLINE float decodeChannelValue( uint8_t value )
{
    return value * ( 1.0f / 255.0f );
}
HIP_DEMAND_INLINE float decodeChannelValue( int8_t value )
{
    return fmaxf( -1.0f, value * ( 1.0f / 127.0f ) );
}
HIP_DEMAND_INLINE float decodeChannelValue( uint16_t value )
{
    return value * ( 1.0f / 65535.0f );
}
HIP_DEMAND_INLINE float decodeChannelValue( int16_t value )
{
    return fmaxf( -1.0f, value * ( 1.0f / 32767.0f ) );
}
HIP_DEMAND_INLINE float decodeChannelValue( uint32_t value )
{
    return static_cast<float>( value ) * ( 1.0f / 4294967295.0f );
}
HIP_DEMAND_INLINE float decodeChannelValue( int32_t value )
{
    return fmaxf( -1.0f, static_cast<float>( value ) * ( 1.0f / 2147483647.0f ) );
}
HIP_DEMAND_INLINE float decodeChannelValue( __half value )
{
    return __half2float( value );
}
HIP_DEMAND_INLINE float decodeChannelValue( float value )
{
    return value;
}

template <class Sample, uint32_t NumChannels, class Vector>
HIP_DEMAND_INLINE Sample decodeVectorTexel( const Vector& value )
{
    const float x = decodeChannelValue( value.x );
    if constexpr( std::is_same<Sample, float>::value )
        return x;
    else
    {
        float y = 0.0f;
        if constexpr( NumChannels > 1 )
            y = decodeChannelValue( value.y );
        if constexpr( std::is_same<Sample, float2>::value )
            return make_float2( x, y );
        else
        {
            float z = 0.0f;
            if constexpr( NumChannels > 2 )
                z = decodeChannelValue( value.z );
            if constexpr( std::is_same<Sample, float3>::value )
                return make_float3( x, y, z );
            else if constexpr( std::is_same<Sample, float4>::value )
            {
                float w = 0.0f;
                if constexpr( NumChannels > 3 )
                    w = decodeChannelValue( value.w );
                return make_float4( x, y, z, w );
            }
            else
                static_assert( std::is_same<Sample, void>::value, "Unsupported texture sample type" );
        }
    }
}

template <class Sample, uint32_t NumChannels, class Channel>
HIP_DEMAND_INLINE Sample decodePackedChannels( const uint8_t* texel )
{
    using PackedChannels       = HIP_vector_type<Channel, NumChannels>;
    const PackedChannels value = *reinterpret_cast<const PackedChannels*>( texel );
    return decodeVectorTexel<Sample, NumChannels>( value );
}

template <class Sample, uint32_t NumChannels>
HIP_DEMAND_INLINE Sample decodeHalfChannels( const uint8_t* texel )
{
    const __half* channels = reinterpret_cast<const __half*>( texel );
    const float   x        = decodeChannelValue( channels[0] );
    if constexpr( std::is_same<Sample, float>::value )
        return x;
    else
    {
        float y = 0.0f;
        if constexpr( NumChannels > 1 )
            y = decodeChannelValue( channels[1] );
        if constexpr( std::is_same<Sample, float2>::value )
            return make_float2( x, y );
        else
        {
            float z = 0.0f;
            if constexpr( NumChannels > 2 )
                z = decodeChannelValue( channels[2] );
            if constexpr( std::is_same<Sample, float3>::value )
                return make_float3( x, y, z );
            else if constexpr( std::is_same<Sample, float4>::value )
            {
                float w = 0.0f;
                if constexpr( NumChannels > 3 )
                    w = decodeChannelValue( channels[3] );
                return make_float4( x, y, z, w );
            }
            else
                static_assert( std::is_same<Sample, void>::value, "Unsupported texture sample type" );
        }
    }
}

template <class Sample, uint32_t NumChannels>
HIP_DEMAND_INLINE Sample decodeTexelChannels( const uint8_t* texel, hipArray_Format format )
{
    switch( format )
    {
        case HIP_AD_FORMAT_UNSIGNED_INT8:
            return decodePackedChannels<Sample, NumChannels, uint8_t>( texel );
        case HIP_AD_FORMAT_SIGNED_INT8:
            return decodePackedChannels<Sample, NumChannels, int8_t>( texel );
        case HIP_AD_FORMAT_UNSIGNED_INT16:
            return decodePackedChannels<Sample, NumChannels, uint16_t>( texel );
        case HIP_AD_FORMAT_SIGNED_INT16:
            return decodePackedChannels<Sample, NumChannels, int16_t>( texel );
        case HIP_AD_FORMAT_UNSIGNED_INT32:
            return decodePackedChannels<Sample, NumChannels, uint32_t>( texel );
        case HIP_AD_FORMAT_SIGNED_INT32:
            return decodePackedChannels<Sample, NumChannels, int32_t>( texel );
        case HIP_AD_FORMAT_HALF:
            return decodeHalfChannels<Sample, NumChannels>( texel );
        case HIP_AD_FORMAT_FLOAT:
            return decodePackedChannels<Sample, NumChannels, float>( texel );
        default:
            return Sample{};
    }
}

template <class Sample>
HIP_DEMAND_INLINE Sample decodeTexel( const uint8_t* texel, hipArray_Format format, uint32_t numChannels )
{
    switch( numChannels )
    {
        case 1:
            return decodeTexelChannels<Sample, 1>( texel, format );
        case 2:
            return decodeTexelChannels<Sample, 2>( texel, format );
        case 3:
            return decodeTexelChannels<Sample, 3>( texel, format );
        case 4:
            return decodeTexelChannels<Sample, 4>( texel, format );
        default:
            return Sample{};
    }
}

// Decode and blend one channel before moving to the next, avoiding four decoded Sample values.
template <class Channel>
HIP_DEMAND_INLINE float filterBilinearChannel( Channel value00, Channel value10, Channel value01, Channel value11,
                                               float weight00, float weight10, float weight01, float weight11 )
{
    if constexpr( std::is_same<Channel, uint8_t>::value || std::is_same<Channel, uint16_t>::value
                  || std::is_same<Channel, uint32_t>::value )
    {
        constexpr float normalization = std::is_same<Channel, uint8_t>::value ? ( 1.0f / 255.0f )
                                        : std::is_same<Channel, uint16_t>::value ? ( 1.0f / 65535.0f )
                                                                                 : ( 1.0f / 4294967295.0f );
        const float blended = weight00 * static_cast<float>( value00 ) + weight10 * static_cast<float>( value10 )
                              + weight01 * static_cast<float>( value01 ) + weight11 * static_cast<float>( value11 );
        return blended * normalization;
    }
    else
    {
        return weight00 * decodeChannelValue( value00 ) + weight10 * decodeChannelValue( value10 )
               + weight01 * decodeChannelValue( value01 ) + weight11 * decodeChannelValue( value11 );
    }
}

template <class Sample, uint32_t NumChannels, class Channel>
HIP_DEMAND_INLINE Sample filterBilinearPackedTexels( const uint8_t* texel00, const uint8_t* texel10,
                                                     const uint8_t* texel01, const uint8_t* texel11,
                                                     float weight00, float weight10, float weight01, float weight11 )
{
    using PackedChannels = HIP_vector_type<Channel, NumChannels>;
    const PackedChannels value00 = *reinterpret_cast<const PackedChannels*>( texel00 );
    const PackedChannels value10 = *reinterpret_cast<const PackedChannels*>( texel10 );
    const PackedChannels value01 = *reinterpret_cast<const PackedChannels*>( texel01 );
    const PackedChannels value11 = *reinterpret_cast<const PackedChannels*>( texel11 );

    const float x = filterBilinearChannel( value00.x, value10.x, value01.x, value11.x,
                                          weight00, weight10, weight01, weight11 );
    if constexpr( std::is_same<Sample, float>::value )
        return x;
    else
    {
        float y = 0.0f;
        if constexpr( NumChannels > 1 )
            y = filterBilinearChannel( value00.y, value10.y, value01.y, value11.y,
                                       weight00, weight10, weight01, weight11 );
        if constexpr( std::is_same<Sample, float2>::value )
            return make_float2( x, y );
        else
        {
            float z = 0.0f;
            if constexpr( NumChannels > 2 )
                z = filterBilinearChannel( value00.z, value10.z, value01.z, value11.z,
                                           weight00, weight10, weight01, weight11 );
            if constexpr( std::is_same<Sample, float3>::value )
                return make_float3( x, y, z );
            else if constexpr( std::is_same<Sample, float4>::value )
            {
                float w = 0.0f;
                if constexpr( NumChannels > 3 )
                    w = filterBilinearChannel( value00.w, value10.w, value01.w, value11.w,
                                               weight00, weight10, weight01, weight11 );
                return make_float4( x, y, z, w );
            }
            else
                static_assert( std::is_same<Sample, void>::value, "Unsupported texture sample type" );
        }
    }
}

template <class Sample, uint32_t NumChannels>
HIP_DEMAND_INLINE Sample filterBilinearHalfTexels( const uint8_t* texel00, const uint8_t* texel10,
                                                   const uint8_t* texel01, const uint8_t* texel11,
                                                   float weight00, float weight10, float weight01, float weight11 )
{
    const __half* value00 = reinterpret_cast<const __half*>( texel00 );
    const __half* value10 = reinterpret_cast<const __half*>( texel10 );
    const __half* value01 = reinterpret_cast<const __half*>( texel01 );
    const __half* value11 = reinterpret_cast<const __half*>( texel11 );

    const float x = filterBilinearChannel( value00[0], value10[0], value01[0], value11[0],
                                          weight00, weight10, weight01, weight11 );
    if constexpr( std::is_same<Sample, float>::value )
        return x;
    else
    {
        float y = 0.0f;
        if constexpr( NumChannels > 1 )
            y = filterBilinearChannel( value00[1], value10[1], value01[1], value11[1],
                                       weight00, weight10, weight01, weight11 );
        if constexpr( std::is_same<Sample, float2>::value )
            return make_float2( x, y );
        else
        {
            float z = 0.0f;
            if constexpr( NumChannels > 2 )
                z = filterBilinearChannel( value00[2], value10[2], value01[2], value11[2],
                                           weight00, weight10, weight01, weight11 );
            if constexpr( std::is_same<Sample, float3>::value )
                return make_float3( x, y, z );
            else if constexpr( std::is_same<Sample, float4>::value )
            {
                float w = 0.0f;
                if constexpr( NumChannels > 3 )
                    w = filterBilinearChannel( value00[3], value10[3], value01[3], value11[3],
                                               weight00, weight10, weight01, weight11 );
                return make_float4( x, y, z, w );
            }
            else
                static_assert( std::is_same<Sample, void>::value, "Unsupported texture sample type" );
        }
    }
}

template <class Sample, uint32_t NumChannels>
HIP_DEMAND_INLINE Sample filterBilinearTexelsChannels( hipArray_Format format,
                                                       const uint8_t* texel00, const uint8_t* texel10,
                                                       const uint8_t* texel01, const uint8_t* texel11,
                                                       float weight00, float weight10, float weight01, float weight11 )
{
    switch( format )
    {
        case HIP_AD_FORMAT_UNSIGNED_INT8:
            return filterBilinearPackedTexels<Sample, NumChannels, uint8_t>( texel00, texel10, texel01, texel11,
                                                                              weight00, weight10, weight01, weight11 );
        case HIP_AD_FORMAT_SIGNED_INT8:
            return filterBilinearPackedTexels<Sample, NumChannels, int8_t>( texel00, texel10, texel01, texel11,
                                                                             weight00, weight10, weight01, weight11 );
        case HIP_AD_FORMAT_UNSIGNED_INT16:
            return filterBilinearPackedTexels<Sample, NumChannels, uint16_t>( texel00, texel10, texel01, texel11,
                                                                               weight00, weight10, weight01, weight11 );
        case HIP_AD_FORMAT_SIGNED_INT16:
            return filterBilinearPackedTexels<Sample, NumChannels, int16_t>( texel00, texel10, texel01, texel11,
                                                                              weight00, weight10, weight01, weight11 );
        case HIP_AD_FORMAT_UNSIGNED_INT32:
            return filterBilinearPackedTexels<Sample, NumChannels, uint32_t>( texel00, texel10, texel01, texel11,
                                                                               weight00, weight10, weight01, weight11 );
        case HIP_AD_FORMAT_SIGNED_INT32:
            return filterBilinearPackedTexels<Sample, NumChannels, int32_t>( texel00, texel10, texel01, texel11,
                                                                              weight00, weight10, weight01, weight11 );
        case HIP_AD_FORMAT_HALF:
            return filterBilinearHalfTexels<Sample, NumChannels>( texel00, texel10, texel01, texel11,
                                                                   weight00, weight10, weight01, weight11 );
        case HIP_AD_FORMAT_FLOAT:
            return filterBilinearPackedTexels<Sample, NumChannels, float>( texel00, texel10, texel01, texel11,
                                                                            weight00, weight10, weight01, weight11 );
        default:
            return Sample{};
    }
}

template <class Sample>
HIP_DEMAND_INLINE Sample filterBilinearTexels( const DeviceTextureInfo& texture,
                                               const uint8_t* texel00, const uint8_t* texel10,
                                               const uint8_t* texel01, const uint8_t* texel11,
                                               float weight00, float weight10, float weight01, float weight11 )
{
    switch( texture.numChannels )
    {
        case 1:
            return filterBilinearTexelsChannels<Sample, 1>( texture.format, texel00, texel10, texel01, texel11,
                                                             weight00, weight10, weight01, weight11 );
        case 2:
            return filterBilinearTexelsChannels<Sample, 2>( texture.format, texel00, texel10, texel01, texel11,
                                                             weight00, weight10, weight01, weight11 );
        case 3:
            return filterBilinearTexelsChannels<Sample, 3>( texture.format, texel00, texel10, texel01, texel11,
                                                             weight00, weight10, weight01, weight11 );
        case 4:
            return filterBilinearTexelsChannels<Sample, 4>( texture.format, texel00, texel10, texel01, texel11,
                                                             weight00, weight10, weight01, weight11 );
        default:
            return Sample{};
    }
}

template <class Sample>
HIP_DEMAND_INLINE Sample
fetchTexel( const DeviceContext& context, const DeviceTextureInfo& texture, const DeviceMipLevel& mip, int x, int y, bool& resident )
{
    resident = false;

    x = applyAddressMode( x, static_cast<int>( mip.width ), texture.addressMode[0] );
    y = applyAddressMode( y, static_cast<int>( mip.height ), texture.addressMode[1] );

    uint32_t pageId         = INVALID_PAGE;
    uint32_t pageByteOffset = 0;
    if( mip.mipTail )
    {
        pageId = texture.mipTailPage;
        pageByteOffset =
            mip.mipTailOffset + ( static_cast<uint32_t>( y ) * mip.width + static_cast<uint32_t>( x ) ) * texture.bytesPerTexel;
    }
    else
    {
        const uint2 tile  = texture.getTileCoords( static_cast<uint32_t>( x ), static_cast<uint32_t>( y ) );
        const uint2 local = texture.getLocalCoords( static_cast<uint32_t>( x ), static_cast<uint32_t>( y ) );
        pageId            = mip.startPage + tile.y * mip.tilesX + tile.x;
        pageByteOffset    = ( local.y * texture.tileWidth + local.x ) * texture.bytesPerTexel;
    }

    const uint32_t resourceId = context.resourceTable.textureTiles.getResourceId( pageId );
    resident                  = isResourceResident( context, resourceId );
    if( !resident )
    {
        recordRequest( context, resourceId );
        return Sample{};
    }
    else if( context.requestIfResident )
    {
        recordRequest( context, resourceId );
    }

    const size_t byteOffset = static_cast<size_t>( pageId ) * context.pageSize + pageByteOffset;
    resident                = true;
    return decodeTexel<Sample>( context.pageMemory.ptr + byteOffset, texture.format, texture.numChannels );
}

HIP_DEMAND_INLINE uint8_t* resolveTexturePage( const DeviceContext& context, uint32_t pageId )
{
    const uint32_t resourceId = context.resourceTable.textureTiles.getResourceId( pageId );
    const bool     resident   = isResourceResident( context, resourceId );
    if( !resident || context.requestIfResident )
        recordRequest( context, resourceId );

    return resident ? context.pageMemory.ptr + static_cast<size_t>( pageId ) * context.pageSize : nullptr;
}

template <class Sample>
HIP_DEMAND_INLINE Sample fetchBilinearSample( const DeviceContext&     context,
                                              const DeviceTextureInfo& texture,
                                              const DeviceMipLevel&    mip,
                                              int                      x0,
                                              int                      y0,
                                              float                    weight00,
                                              float                    weight10,
                                              float                    weight01,
                                              float                    weight11,
                                              bool&                    resident )
{
    // Fast path for four bilinear taps inside one tile. In-bounds coordinates need no address-mode handling,
    // and all texel addresses can be derived from one page lookup and one base offset.
    const bool footprintInBounds = x0 >= 0 && y0 >= 0 && static_cast<uint32_t>( x0 ) + 1 < mip.width
                                   && static_cast<uint32_t>( y0 ) + 1 < mip.height;
    if( !mip.mipTail && footprintInBounds )
    {
        const uint32_t texelX = static_cast<uint32_t>( x0 );
        const uint32_t texelY = static_cast<uint32_t>( y0 );
        const uint2    local  = texture.getLocalCoords( texelX, texelY );
        // The 2x2 footprint fits in one tile only if its top-left texel is before the last column and row.
        if( local.x < texture.tileWidthMask && local.y < texture.tileHeightMask )
        {
            const uint2    tile     = texture.getTileCoords( texelX, texelY );
            const uint32_t pageId   = mip.startPage + tile.y * mip.tilesX + tile.x;
            const uint32_t rowBytes = texture.tileWidth * texture.bytesPerTexel;
            const uint32_t offset00 = local.y * rowBytes + local.x * texture.bytesPerTexel;

            const uint8_t* page = resolveTexturePage( context, pageId );
            resident            = page != nullptr;
            if( !resident )
                return Sample{};

            const uint8_t* row0 = page + offset00;
            const uint8_t* row1 = row0 + rowBytes;
            return filterBilinearTexels<Sample>( texture, row0, row0 + texture.bytesPerTexel,
                                                 row1, row1 + texture.bytesPerTexel,
                                                 weight00, weight10, weight01, weight11 );
        }
    }

    int x1;
    int y1;
    if( footprintInBounds )
    {
        x1 = x0 + 1;
        y1 = y0 + 1;
    }
    else
    {
        const int2 x = applyAddressModePair( x0, x0 + 1, static_cast<int>( mip.width ), texture.addressMode[0] );
        const int2 y = applyAddressModePair( y0, y0 + 1, static_cast<int>( mip.height ), texture.addressMode[1] );
        x0           = x.x;
        y0           = y.x;
        x1           = x.y;
        y1           = y.y;
    }

    if( mip.mipTail )
    {
        // All four taps share one page; compute each row and column address once.
        const uint8_t* page = resolveTexturePage( context, texture.mipTailPage );
        resident            = page != nullptr;
        if( !resident )
            return Sample{};

        const uint32_t rowBytes  = mip.width * texture.bytesPerTexel;
        const uint8_t* row0     = page + mip.mipTailOffset + static_cast<uint32_t>( y0 ) * rowBytes;
        const uint8_t* row1     = page + mip.mipTailOffset + static_cast<uint32_t>( y1 ) * rowBytes;
        const uint32_t xOffset0 = static_cast<uint32_t>( x0 ) * texture.bytesPerTexel;
        const uint32_t xOffset1 = static_cast<uint32_t>( x1 ) * texture.bytesPerTexel;
        return filterBilinearTexels<Sample>( texture, row0 + xOffset0, row0 + xOffset1,
                                             row1 + xOffset0, row1 + xOffset1,
                                             weight00, weight10, weight01, weight11 );
    }
    const uint2    tile0    = texture.getTileCoords( static_cast<uint32_t>( x0 ), static_cast<uint32_t>( y0 ) );
    const uint2    tile1    = texture.getTileCoords( static_cast<uint32_t>( x1 ), static_cast<uint32_t>( y1 ) );
    const uint2    local0   = texture.getLocalCoords( static_cast<uint32_t>( x0 ), static_cast<uint32_t>( y0 ) );
    const uint2    local1   = texture.getLocalCoords( static_cast<uint32_t>( x1 ), static_cast<uint32_t>( y1 ) );
    const uint32_t rowBytes = texture.tileWidth * texture.bytesPerTexel;
    const uint32_t offset00 = local0.y * rowBytes + local0.x * texture.bytesPerTexel;
    const uint32_t offset10 = local0.y * rowBytes + local1.x * texture.bytesPerTexel;
    const uint32_t offset01 = local1.y * rowBytes + local0.x * texture.bytesPerTexel;
    const uint32_t offset11 = local1.y * rowBytes + local1.x * texture.bytesPerTexel;

    const uint32_t pageRow0 = mip.startPage + tile0.y * mip.tilesX;
    const uint32_t pageRow1 = mip.startPage + tile1.y * mip.tilesX;
    const bool     sameTileX = tile0.x == tile1.x;
    const bool     sameTileY = tile0.y == tile1.y;
    const uint8_t* page00   = resolveTexturePage( context, pageRow0 + tile0.x );
    const uint8_t* page10   = sameTileX ? page00 : resolveTexturePage( context, pageRow0 + tile1.x );
    const uint8_t* page01   = sameTileY ? page00 : resolveTexturePage( context, pageRow1 + tile0.x );
    const uint8_t* page11   = sameTileY ? page10
                                     : ( sameTileX ? page01 : resolveTexturePage( context, pageRow1 + tile1.x ) );

    resident = page00 && page10 && page01 && page11;
    if( !resident )
        return Sample{};

    return filterBilinearTexels<Sample>( texture, page00 + offset00, page10 + offset10,
                                         page01 + offset01, page11 + offset11,
                                         weight00, weight10, weight01, weight11 );
}

template <class Sample>
HIP_DEMAND_INLINE Sample
sampleMipLevel( const DeviceContext& context, const DeviceTextureInfo& texture, uint32_t mipLevel, float x, float y, bool& resident )
{
    resident                 = false;
    const DeviceMipLevel mip = texture.getMipLevel( mipLevel );
    if( texture.normalizedCoords )
    {
        x = x * static_cast<float>( mip.width );
        y = y * static_cast<float>( mip.height );
    }

    if( texture.filterMode == hipFilterModePoint )
    {
        x = std::floorf( x );
        y = std::floorf( y );
        return fetchTexel<Sample>( context, texture, mip, static_cast<int>( x ), static_cast<int>( y ), resident );
    }

    x = x - 0.5f;
    y = y - 0.5f;

    const int   x0 = static_cast<int>( std::floorf( x ) );
    const int   y0 = static_cast<int>( std::floorf( y ) );
    const float a  = x - static_cast<float>( x0 );
    const float b  = y - static_cast<float>( y0 );

    const float weight00 = ( 1.0f - a ) * ( 1.0f - b );
    const float weight10 = a * ( 1.0f - b );
    const float weight01 = ( 1.0f - a ) * b;
    const float weight11 = a * b;
    return fetchBilinearSample<Sample>( context, texture, mip, x0, y0,
                                        weight00, weight10, weight01, weight11, resident );
}

template <class Sample>
HIP_DEMAND_INLINE Sample tex2DLod( const DeviceContext& context, uint32_t textureId, float x, float y, float lod, bool& resident )
{
    resident = false;
    if( textureId >= context.textureInfos.len )
    {
        resident = true;
        return Sample{};
    }

    // no need to check isResourceResident if we could just try to load texture info and check if it's null
    // const uint32_t resourceId = context.resourceTable.textureInfos.getResourceId( textureId );
    // resident                  = isResourceResident( context, resourceId );

    const DeviceTextureInfo* texture = context.textureInfos.ptr[textureId];
    if( !texture )
    {
        recordRequest( context, context.resourceTable.textureInfos.getResourceId( textureId ) );
        return Sample{};
    }
    else if( context.requestIfResident )
    {
        recordRequest( context, context.resourceTable.textureInfos.getResourceId( textureId ) );
    }

    resident = true;

    lod = std::clamp( lod, 0.0f, static_cast<float>( texture->mipCount - 1 ) );
    if( texture->mipmapFilterMode == hipFilterModePoint )
    {
        const uint32_t mipLevel = static_cast<uint32_t>( std::floorf( lod + 0.5f ) );
        return sampleMipLevel<Sample>( context, *texture, mipLevel, x, y, resident );
    }

    const uint32_t mipLevel0 = static_cast<uint32_t>( std::floorf( lod ) );
    const uint32_t mipLevel1 = mipLevel0 + 1 < texture->mipCount ? mipLevel0 + 1 : mipLevel0;
    const float    t         = lod - static_cast<float>( mipLevel0 );

    bool         resident0 = false;
    const Sample sample0   = sampleMipLevel<Sample>( context, *texture, mipLevel0, x, y, resident0 );
    // The next mip level cannot affect an exact-level sample or its residency.
    if( mipLevel0 == mipLevel1 || t == 0.0f )
    {
        resident = resident0;
        return sample0;
    }

    bool         resident1 = false;
    const Sample sample1   = sampleMipLevel<Sample>( context, *texture, mipLevel1, x, y, resident1 );

    resident = resident0 && resident1;
    return ( 1.0f - t ) * sample0 + t * sample1;
}

// TODO_BS: this implementation is generated by chatGPT. I'll need to do more research about it
template <class Sample>
HIP_DEMAND_INLINE Sample tex2DGrad( const DeviceContext& context, uint32_t textureId, float x, float y, float2 ddx, float2 ddy, bool& resident )
{
    resident = false;
    if( textureId >= context.textureInfos.len )
    {
        resident = true;
        return Sample{};
    }

    const uint32_t resourceId = context.resourceTable.textureInfos.getResourceId( textureId );
    resident = isResourceResident( context, resourceId );
    if( !resident )
    {
        recordRequest( context, resourceId );
        return Sample{};
    }
    else if (context.requestIfResident)
    {
        recordRequest( context, resourceId );
    }

    const DeviceTextureInfo& texture = *context.textureInfos.ptr[textureId];
    if( texture.normalizedCoords )
    {
        const DeviceMipLevel baseMip = texture.getMipLevel( 0 );
        ddx.x *= static_cast<float>( baseMip.width );
        ddx.y *= static_cast<float>( baseMip.height );
        ddy.x *= static_cast<float>( baseMip.width );
        ddy.y *= static_cast<float>( baseMip.height );
    }

    const float footprintXSquared = ddx.x * ddx.x + ddx.y * ddx.y;
    const float footprintYSquared = ddy.x * ddy.x + ddy.y * ddy.y;
    const float footprintSquared  = fmaxf( footprintXSquared, footprintYSquared );
    const float lod               = 0.5f * log2f( fmaxf( footprintSquared, 1.0f ) );

    return tex2DLod<Sample>( context, textureId, x, y, lod, resident );
}

template <class Sample>
HIP_DEMAND_INLINE Sample tex2D( const DeviceContext& context, uint32_t textureId, float x, float y, bool& resident )
{
    return tex2DLod<Sample>( context, textureId, x, y, 0.0f, resident );
}

}  // namespace hip_demand::vmm
