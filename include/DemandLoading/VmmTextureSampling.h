#pragma once

#include "DeviceContext.h"
#include "VmmDeviceUtils.h"
#include <algorithm>
#include <cmath>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <limits>
#include <type_traits>


namespace hip_demand::vmm {

HIP_DEMAND_INLINE void recordRequest( const DeviceContext& context, uint32_t resourceId )
{
    const uint64_t activeLanes   = __activemask();
    const uint64_t matchingLanes = __match_any_sync( activeLanes, resourceId );
    const uint32_t leaderLane    = static_cast<uint32_t>( __ffsll( static_cast<long long>( matchingLanes ) ) - 1 );
    if( static_cast<uint32_t>( __lane_id() ) != leaderLane )
        return;

    atomicSetBit( context.referenceBits, resourceId );
}

HIP_DEMAND_INLINE bool referenceResource( const DeviceContext& context, uint32_t resourceId, bool resident )
{
    if( !resident || context.requestIfResident )
        recordRequest( context, resourceId );
    return resident;
}

HIP_DEMAND_INLINE uint32_t loadPageResidenceWord( const DeviceContext& context, uint32_t pageId )
{
    return context.residenceBits.ptr[context.resourceTable.textureTiles.getResourceId( pageId ) >> 5];
}

HIP_DEMAND_INLINE bool referencePage( const DeviceContext& context, uint32_t pageId, uint32_t residenceWord )
{
    const uint32_t resourceId = context.resourceTable.textureTiles.getResourceId( pageId );
    return referenceResource( context, resourceId, ( residenceWord >> ( resourceId & 31u ) ) & 1u );
}

HIP_DEMAND_INLINE bool referencePage( const DeviceContext& context, uint32_t pageId )
{
    return referencePage( context, pageId, loadPageResidenceWord( context, pageId ) );
}

// The page size is a power of two, so page offsets are computed with shifts.
HIP_DEMAND_INLINE uint32_t pageSizeShift( const DeviceContext& context )
{
    return static_cast<uint32_t>( __builtin_ctzll( context.pageSize ) );
}

HIP_DEMAND_INLINE void referenceTextureInfo( const DeviceContext& context, uint32_t textureId )
{
    if( context.requestIfResident )
        recordRequest( context, context.resourceTable.textureInfos.getResourceId( textureId ) );
}

HIP_DEMAND_INLINE void requestCascade( const DeviceContext& context, uint32_t textureId, uint32_t level )
{
    recordRequest( context, context.resourceTable.cascades.getResourceId( textureId * MAX_TILED_MIP_LEVELS + level ) );
}

// Reads a value from device global memory through a global address space pointer. A generic pointer loaded from memory
// can only be read with flat loads, which the compiler never turns into scalar loads, even for wave-uniform addresses.
template <class T>
HIP_DEMAND_INLINE T loadGlobal( const T* pointer )
{
    static_assert( std::is_trivially_copyable<T>::value && sizeof( T ) % sizeof( uint32_t ) == 0,
                   "loadGlobal copies whole 32-bit words" );
    using GlobalWord = const __attribute__( ( address_space( 1 ) ) ) uint32_t;

    constexpr uint32_t wordCount = sizeof( T ) / sizeof( uint32_t );
    GlobalWord*        source    = (GlobalWord*)pointer;
    uint32_t           words[wordCount];
    for( uint32_t word = 0; word < wordCount; ++word )
        words[word] = source[word];

    T value;
    __builtin_memcpy( &value, words, sizeof( T ) );
    return value;
}

// Texture metadata as read by the sampler: a local copy of the descriptor plus the location of the per-level page
// descriptors, which are loaded only for the sampled levels.
//
// Metadata from the loader lives in global memory and is not written while kernels run. The sampler reads all of it
// before referencing any resource: the compiler cannot prove that memory read after an atomic is unchanged, and would
// then fall back from scalar to vector loads for wave-uniform textures.
struct TextureMetadata
{
    DeviceTextureInfo info;
    const uint32_t*   mips = nullptr;
    uint32_t          id   = INVALID_TEXTURE;

    HIP_DEMAND_INLINE DeviceMipLevel mipLevel( uint32_t level ) const
    {
        // info.mips must not be indexed.
        // The page descriptor is read from global memory rather than from the local copy in info.mips. That copy
        // lives in registers, and a register array cannot be indexed by a runtime value: the compiler would store the
        // whole array to scratch memory and read the entry back through it, on every sample. A direct load of the one
        // entry is a single scalar load for a wave-uniform level. The copy of info.mips is never read, so the compiler
        // drops those loads from loadGlobal() and nothing is loaded twice.
        // A tail level has no entry: its page is info.mipTailPage, and its index may exceed the array.
        const uint32_t startPage = level < info.mipTailFirstLevel() ? loadGlobal( mips + level ) : INVALID_PAGE;
        return info.getMipLevel( level, startPage );
    }
};

// Reads a texture's metadata and reports whether it is resident, requesting it otherwise. The metadata array is
// always allocated, so the descriptor is read first; it is resident once the loader has copied it, which is visible
// as the loaded bit because the copy and every kernel that can read it are ordered on the same stream.
HIP_DEMAND_INLINE TextureMetadata loadTextureMetadata( const DeviceContext& context, uint32_t textureId, bool& resident )
{
    const DeviceTextureInfo* texture  = context.textureInfos.ptr + textureId;
    const TextureMetadata    metadata = { loadGlobal( texture ), texture->mips, textureId };

    resident = metadata.info.isLoaded();
    if( !resident )
        recordRequest( context, context.resourceTable.textureInfos.getResourceId( textureId ) );
    return metadata;
}

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

// ---------------------------------------------------------------------------------------------------------------------
// Footprints
//
// A footprint holds the page and in-page byte offset of every texel a filter reads from one mip level. Footprints are
// computed for all mip levels of a sample before any memory access, so that residence checks and texel loads of
// different levels are issued together instead of as a chain of dependent loads.
// ---------------------------------------------------------------------------------------------------------------------

template <uint32_t NumTaps>
struct Footprint
{
    static constexpr uint32_t numTaps = NumTaps;

    uint32_t pageId[NumTaps];
    uint32_t texelOffset[NumTaps];  // Low 32 bits of the byte offset from the level's first page; see texelOffsetInLevel().
    bool  singlePage = true;  // All taps are in pageId[0].
    float fracX      = 0.0f;  // Bilinear weights of the right column and the bottom row.
    float fracY      = 0.0f;
};

using PointFootprint    = Footprint<1>;
using BilinearFootprint = Footprint<4>;

// Texel addresses are kept as 32-bit offsets from the first page of their mip level, so that loads can use a
// wave-uniform base address with a 32-bit per-lane offset instead of 64-bit address arithmetic per tap. Levels have
// pages of their own because a texture may get them level by level (see DeviceTextureInfo::cascadeLevel). For a level
// whose pages span more than 4 GiB (DeviceTextureInfo::wideOffsets) the upper 32 bits are recovered from the page id
// when filtering.
HIP_DEMAND_INLINE uint32_t texelOffsetInLevel( const DeviceMipLevel& mip, uint32_t pageShift, uint32_t pageId, uint32_t byteOffset )
{
    return ( ( pageId - mip.startPage ) << pageShift ) + byteOffset;
}

// Texel address for the low offset bits in a footprint. The in-page offset never carries into the page bits, so the
// upper 32 bits of the full offset are simply the upper bits of the page offset.
template <bool Wide>
HIP_DEMAND_INLINE const uint8_t* texelAddress( const uint8_t*        levelMemory,
                                               const DeviceMipLevel& mip,
                                               uint32_t              pageShift,
                                               uint32_t              pageId,
                                               uint32_t              texelOffset )
{
    if constexpr( Wide )
    {
        const size_t high = static_cast<size_t>( ( pageId - mip.startPage ) >> ( 32 - pageShift ) );
        return levelMemory + ( ( high << 32 ) | texelOffset );
    }
    else
        return levelMemory + texelOffset;
}

HIP_DEMAND_INLINE float2 texelCoordinates( const DeviceTextureInfo& texture, const DeviceMipLevel& mip, float x, float y )
{
    if( texture.normalizedCoords() )
        return make_float2( x * static_cast<float>( mip.width ), y * static_cast<float>( mip.height ) );
    return make_float2( x, y );
}

HIP_DEMAND_INLINE PointFootprint
makePointFootprint( const DeviceTextureInfo& texture, const DeviceMipLevel& mip, uint32_t pageShift, float x, float y )
{
    const float2   coords = texelCoordinates( texture, mip, x, y );
    const uint32_t texelX = static_cast<uint32_t>(
        applyAddressMode( static_cast<int>( floorf( coords.x ) ), static_cast<int>( mip.width ), texture.addressMode( 0 ) ) );
    const uint32_t texelY = static_cast<uint32_t>(
        applyAddressMode( static_cast<int>( floorf( coords.y ) ), static_cast<int>( mip.height ), texture.addressMode( 1 ) ) );

    uint32_t pageId = texture.mipTailPage;
    // Within the page, so it fits 32 bits: a mip tail level is at most one page.
    uint32_t byteOffset = mip.mipTailOffset + ( texelY * mip.width + texelX ) * texture.bytesPerTexel();
    if( !mip.mipTail )
    {
        const uint2 tile  = texture.getTileCoords( texelX, texelY );
        const uint2 local = texture.getLocalCoords( texelX, texelY );
        pageId            = mip.startPage + tile.y * mip.tilesX + tile.x;
        byteOffset        = ( local.y * texture.tileWidth() + local.x ) * texture.bytesPerTexel();
    }

    PointFootprint footprint{};
    footprint.pageId[0]      = pageId;
    footprint.texelOffset[0] = texelOffsetInLevel( mip, pageShift, pageId, byteOffset );
    return footprint;
}

HIP_DEMAND_INLINE BilinearFootprint
makeBilinearFootprint( const DeviceTextureInfo& texture, const DeviceMipLevel& mip, uint32_t pageShift, float x, float y )
{
    const float2 coords = texelCoordinates( texture, mip, x, y );
    const float  fx     = coords.x - 0.5f;
    const float  fy     = coords.y - 0.5f;
    const float  floorX = floorf( fx );
    const float  floorY = floorf( fy );
    int          x0     = static_cast<int>( floorX );
    int          y0     = static_cast<int>( floorY );

    BilinearFootprint footprint{};
    footprint.fracX = fx - floorX;
    footprint.fracY = fy - floorY;

    const uint32_t bytesPerTexel = texture.bytesPerTexel();
    const bool     footprintInBounds =
        x0 >= 0 && y0 >= 0 && static_cast<uint32_t>( x0 ) + 1 < mip.width && static_cast<uint32_t>( y0 ) + 1 < mip.height;

    // Fast path for four taps inside one tile: in-bounds coordinates need no address-mode handling.
    if( !mip.mipTail && footprintInBounds )
    {
        const uint2 local = texture.getLocalCoords( static_cast<uint32_t>( x0 ), static_cast<uint32_t>( y0 ) );
        // The 2x2 footprint fits in one tile only if its top-left texel is before the last column and row.
        if( local.x < texture.tileWidthMask() && local.y < texture.tileHeightMask() )
        {
            const uint2    tile     = texture.getTileCoords( static_cast<uint32_t>( x0 ), static_cast<uint32_t>( y0 ) );
            const uint32_t pageId   = mip.startPage + tile.y * mip.tilesX + tile.x;
            const uint32_t rowBytes = texture.tileWidth() * bytesPerTexel;
            const uint32_t offset   = local.y * rowBytes + local.x * bytesPerTexel;

            const uint32_t texel00 = texelOffsetInLevel( mip, pageShift, pageId, offset );
            footprint.pageId[0] = footprint.pageId[1] = footprint.pageId[2] = footprint.pageId[3] = pageId;
            footprint.texelOffset[0]                                                              = texel00;
            footprint.texelOffset[1] = texel00 + bytesPerTexel;
            footprint.texelOffset[2] = texel00 + rowBytes;
            footprint.texelOffset[3] = texel00 + rowBytes + bytesPerTexel;
            return footprint;
        }
    }

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    if( !footprintInBounds )
    {
        const int2 xs = applyAddressModePair( x0, x1, static_cast<int>( mip.width ), texture.addressMode( 0 ) );
        const int2 ys = applyAddressModePair( y0, y1, static_cast<int>( mip.height ), texture.addressMode( 1 ) );
        x0            = xs.x;
        x1            = xs.y;
        y0            = ys.x;
        y1            = ys.y;
    }

    if( mip.mipTail )
    {
        // The whole mip tail is one page with row-major levels.
        const uint32_t rowBytes = mip.width * bytesPerTexel;
        const uint32_t row0     = mip.mipTailOffset + static_cast<uint32_t>( y0 ) * rowBytes;
        const uint32_t row1     = mip.mipTailOffset + static_cast<uint32_t>( y1 ) * rowBytes;
        const uint32_t column0  = static_cast<uint32_t>( x0 ) * bytesPerTexel;
        const uint32_t column1  = static_cast<uint32_t>( x1 ) * bytesPerTexel;

        footprint.pageId[0] = footprint.pageId[1] = footprint.pageId[2] = footprint.pageId[3] = texture.mipTailPage;
        footprint.texelOffset[0]                                                              = row0 + column0;
        footprint.texelOffset[1]                                                              = row0 + column1;
        footprint.texelOffset[2]                                                              = row1 + column0;
        footprint.texelOffset[3]                                                              = row1 + column1;
        return footprint;
    }

    // The footprint crosses a tile edge or the texture border and may touch up to four pages. Tile and in-tile
    // coordinates are separable per axis, so those of the two corner taps (x0, y0) and (x1, y1) give the other two
    // taps as well: tap (x1, y0) is in tile (tile11.x, tile00.y), tap (x0, y1) in tile (tile00.x, tile11.y).
    const uint2    tile00   = texture.getTileCoords( static_cast<uint32_t>( x0 ), static_cast<uint32_t>( y0 ) );
    const uint2    tile11   = texture.getTileCoords( static_cast<uint32_t>( x1 ), static_cast<uint32_t>( y1 ) );
    const uint2    local00  = texture.getLocalCoords( static_cast<uint32_t>( x0 ), static_cast<uint32_t>( y0 ) );
    const uint2    local11  = texture.getLocalCoords( static_cast<uint32_t>( x1 ), static_cast<uint32_t>( y1 ) );
    const uint32_t rowBytes = texture.tileWidth() * bytesPerTexel;
    const uint32_t row0     = local00.y * rowBytes;
    const uint32_t row1     = local11.y * rowBytes;
    const uint32_t column0  = local00.x * bytesPerTexel;
    const uint32_t column1  = local11.x * bytesPerTexel;
    const uint32_t pageRow0 = mip.startPage + tile00.y * mip.tilesX;
    const uint32_t pageRow1 = mip.startPage + tile11.y * mip.tilesX;

    footprint.pageId[0]      = pageRow0 + tile00.x;  // tap (x0, y0)
    footprint.pageId[1]      = pageRow0 + tile11.x;  // tap (x1, y0)
    footprint.pageId[2]      = pageRow1 + tile00.x;  // tap (x0, y1)
    footprint.pageId[3]      = pageRow1 + tile11.x;  // tap (x1, y1)
    footprint.texelOffset[0] = texelOffsetInLevel( mip, pageShift, footprint.pageId[0], row0 + column0 );
    footprint.texelOffset[1] = texelOffsetInLevel( mip, pageShift, footprint.pageId[1], row0 + column1 );
    footprint.texelOffset[2] = texelOffsetInLevel( mip, pageShift, footprint.pageId[2], row1 + column0 );
    footprint.texelOffset[3] = texelOffsetInLevel( mip, pageShift, footprint.pageId[3], row1 + column1 );
    footprint.singlePage     = tile00.x == tile11.x && tile00.y == tile11.y;
    return footprint;
}

// Checks and references every distinct page of a footprint, given the residence word of its first page.
template <uint32_t NumTaps>
HIP_DEMAND_INLINE bool referenceFootprint( const DeviceContext& context, const Footprint<NumTaps>& footprint, uint32_t residenceWord )
{
    bool resident = referencePage( context, footprint.pageId[0], residenceWord );
    if( !footprint.singlePage )
    {
        for( uint32_t tap = 1; tap < NumTaps; ++tap )
        {
            bool repeated = false;
            for( uint32_t previous = 0; previous < tap; ++previous )
                repeated = repeated || footprint.pageId[previous] == footprint.pageId[tap];
            if( !repeated )
                resident = referencePage( context, footprint.pageId[tap] ) && resident;
        }
    }
    return resident;
}

// ---------------------------------------------------------------------------------------------------------------------
// Texel formats
//
// Texels are loaded as "filterable" floats and normalized once per sample. Unsigned normalized channels stay in integer
// scale until then, which saves one multiply per channel and texel when filtering. Other formats are fully decoded on
// load; signed channels must be clamped to -1 per texel, before blending.
// ---------------------------------------------------------------------------------------------------------------------

template <class Storage_>
struct UnsignedNormalizedChannel
{
    using Storage                = Storage_;
    static constexpr float scale = 1.0f / static_cast<float>( std::numeric_limits<Storage>::max() );

    HIP_DEMAND_INLINE static float toFilterable( Storage value ) { return static_cast<float>( value ); }
};

template <class Storage_>
struct SignedNormalizedChannel
{
    using Storage                = Storage_;
    static constexpr float scale = 1.0f;

    HIP_DEMAND_INLINE static float toFilterable( Storage value )
    {
        constexpr float inverseMax = 1.0f / static_cast<float>( std::numeric_limits<Storage>::max() );
        return fmaxf( -1.0f, static_cast<float>( value ) * inverseMax );
    }
};

struct HalfChannel
{
    using Storage                = unsigned short;
    static constexpr float scale = 1.0f;

    HIP_DEMAND_INLINE static float toFilterable( Storage value ) { return __half2float( __ushort_as_half( value ) ); }
};

struct FloatChannel
{
    using Storage                = float;
    static constexpr float scale = 1.0f;

    HIP_DEMAND_INLINE static float toFilterable( Storage value ) { return value; }
};

template <class Channel_, uint32_t NumChannels>
struct TexelFormat
{
    using Channel = Channel_;
    using Packed  = HIP_vector_type<typename Channel::Storage, NumChannels>;

    using Storage = typename Channel::Storage;

    // A 3-channel texel of 1- or 2-byte channels is loaded as one 4- or 8-byte word instead of three channel loads.
    // The word extends 1 or 2 bytes past the texel, which is always inside the same page: a tile holds a power-of-two
    // number of texels and the mip tail a multiple of 3 channels, and neither byte size can equal the page size.
    static constexpr bool loadAsWord = NumChannels == 3 && sizeof( Storage ) <= 2;
    using Word                       = std::conditional_t<sizeof( Storage ) == 1, uint32_t, uint64_t>;

    // Channels missing from the format read as zero.
    HIP_DEMAND_INLINE static float4 load( const uint8_t* texel )
    {
        if constexpr( loadAsWord )
        {
            Word word;
            __builtin_memcpy( &word, texel, sizeof( Word ) );
            return make_float4( Channel::toFilterable( static_cast<Storage>( word ) ),
                                Channel::toFilterable( static_cast<Storage>( word >> ( 8 * sizeof( Storage ) ) ) ),
                                Channel::toFilterable( static_cast<Storage>( word >> ( 16 * sizeof( Storage ) ) ) ), 0.0f );
        }
        else
        {
            const Packed packed = *reinterpret_cast<const Packed*>( texel );
            float4       value  = make_float4( Channel::toFilterable( packed.x ), 0.0f, 0.0f, 0.0f );
            if constexpr( NumChannels > 1 )
                value.y = Channel::toFilterable( packed.y );
            if constexpr( NumChannels > 2 )
                value.z = Channel::toFilterable( packed.z );
            if constexpr( NumChannels > 3 )
                value.w = Channel::toFilterable( packed.w );
            return value;
        }
    }

    // Converts loaded or blended filterable values to the sampled result.
    HIP_DEMAND_INLINE static float4 normalize( float4 value )
    {
        if constexpr( Channel::scale != 1.0f )
            value *= Channel::scale;
        return value;
    }
};

// Calls fn( TexelFormat<...>{} ) for the runtime format, or returns Result{} for an unsupported one.
template <class Result, uint32_t NumChannels, class Fn>
HIP_DEMAND_INLINE Result dispatchChannelFormat( hipArray_Format format, Fn&& fn )
{
    switch( format )
    {
        case HIP_AD_FORMAT_UNSIGNED_INT8:
            return fn( TexelFormat<UnsignedNormalizedChannel<uint8_t>, NumChannels>{} );
        case HIP_AD_FORMAT_SIGNED_INT8:
            return fn( TexelFormat<SignedNormalizedChannel<int8_t>, NumChannels>{} );
        case HIP_AD_FORMAT_UNSIGNED_INT16:
            return fn( TexelFormat<UnsignedNormalizedChannel<uint16_t>, NumChannels>{} );
        case HIP_AD_FORMAT_SIGNED_INT16:
            return fn( TexelFormat<SignedNormalizedChannel<int16_t>, NumChannels>{} );
        case HIP_AD_FORMAT_UNSIGNED_INT32:
            return fn( TexelFormat<UnsignedNormalizedChannel<uint32_t>, NumChannels>{} );
        case HIP_AD_FORMAT_SIGNED_INT32:
            return fn( TexelFormat<SignedNormalizedChannel<int32_t>, NumChannels>{} );
        case HIP_AD_FORMAT_HALF:
            return fn( TexelFormat<HalfChannel, NumChannels>{} );
        case HIP_AD_FORMAT_FLOAT:
            return fn( TexelFormat<FloatChannel, NumChannels>{} );
        default:
            return Result{};
    }
}

template <class Result, class Fn>
HIP_DEMAND_INLINE Result dispatchTexelFormat( hipArray_Format format, uint32_t numChannels, Fn&& fn )
{
    switch( numChannels )
    {
        case 1:
            return dispatchChannelFormat<Result, 1>( format, fn );
        case 2:
            return dispatchChannelFormat<Result, 2>( format, fn );
        case 3:
            return dispatchChannelFormat<Result, 3>( format, fn );
        case 4:
            return dispatchChannelFormat<Result, 4>( format, fn );
        default:
            return Result{};
    }
}

template <class Sample>
HIP_DEMAND_INLINE Sample toSample( float4 value )
{
    if constexpr( std::is_same<Sample, float>::value )
        return value.x;
    else if constexpr( std::is_same<Sample, float2>::value )
        return make_float2( value.x, value.y );
    else if constexpr( std::is_same<Sample, float3>::value )
        return make_float3( value.x, value.y, value.z );
    else
    {
        static_assert( std::is_same<Sample, float4>::value, "Unsupported texture sample type" );
        return value;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------------------------------------------------

// levelMemory is the address of the mip level's first page.
template <class Format, bool Wide>
HIP_DEMAND_INLINE float4 filterFootprint( const uint8_t*        levelMemory,
                                          const DeviceMipLevel& mip,
                                          uint32_t              pageShift,
                                          const PointFootprint& footprint )
{
    return Format::normalize(
        Format::load( texelAddress<Wide>( levelMemory, mip, pageShift, footprint.pageId[0], footprint.texelOffset[0] ) ) );
}

template <class Format, bool Wide>
HIP_DEMAND_INLINE float4 filterFootprint( const uint8_t*           levelMemory,
                                          const DeviceMipLevel&    mip,
                                          uint32_t                 pageShift,
                                          const BilinearFootprint& footprint )
{
    const uint8_t* texels[4];
    for( uint32_t tap = 0; tap < 4; ++tap )
        texels[tap] = texelAddress<Wide>( levelMemory, mip, pageShift, footprint.pageId[tap], footprint.texelOffset[tap] );

    const float4 value00 = Format::load( texels[0] );
    const float4 value10 = Format::load( texels[1] );
    const float4 value01 = Format::load( texels[2] );
    const float4 value11 = Format::load( texels[3] );

    const float fracX    = footprint.fracX;
    const float fracY    = footprint.fracY;
    const float weight00 = ( 1.0f - fracX ) * ( 1.0f - fracY );
    const float weight10 = fracX * ( 1.0f - fracY );
    const float weight01 = ( 1.0f - fracX ) * fracY;
    const float weight11 = fracX * fracY;
    return Format::normalize( weight00 * value00 + weight10 * value10 + weight01 * value01 + weight11 * value11 );
}

template <class Footprint>
HIP_DEMAND_INLINE Footprint makeFootprint( const DeviceTextureInfo& texture, const DeviceMipLevel& mip, uint32_t pageShift, float x, float y )
{
    if constexpr( Footprint::numTaps == 1 )
        return makePointFootprint( texture, mip, pageShift, x, y );
    else
        return makeBilinearFootprint( texture, mip, pageShift, x, y );
}

template <class Sample, class Footprint, uint32_t NumLevels, bool Wide>
HIP_DEMAND_INLINE Sample filterLevels( const DeviceContext&     context,
                                       const DeviceTextureInfo& texture,
                                       uint32_t                 pageShift,
                                       const DeviceMipLevel ( &mips )[NumLevels],
                                       const Footprint ( &footprints )[NumLevels],
                                       float t )
{
    static_assert( NumLevels == 1 || NumLevels == 2, "A sample blends at most two mip levels" );

    const uint8_t* levelMemory[NumLevels];
    for( uint32_t level = 0; level < NumLevels; ++level )
        levelMemory[level] = context.pageMemory.ptr + ( static_cast<size_t>( mips[level].startPage ) << pageShift );

    return dispatchTexelFormat<Sample>( texture.format(), texture.numChannels(), [&]( auto texelFormat ) {
        using Format = decltype( texelFormat );
        float4 value = filterFootprint<Format, Wide>( levelMemory[0], mips[0], pageShift, footprints[0] );
        if constexpr( NumLevels == 2 )
            value = ( 1.0f - t ) * value + t * filterFootprint<Format, Wide>( levelMemory[1], mips[1], pageShift, footprints[1] );
        return toSample<Sample>( value );
    } );
}

template <class Sample, class Footprint, uint32_t NumLevels>
HIP_DEMAND_INLINE Sample sampleMipLevels( const DeviceContext&   context,
                                          const TextureMetadata& texture,
                                          uint32_t               mipLevel,
                                          float                  t,
                                          float                  x,
                                          float                  y,
                                          uint32_t               requestedLevel,
                                          bool&                  resident )
{
    const uint32_t pageShift = pageSizeShift( context );

    // The page descriptors of all levels are loaded first, so that the scalar loads are issued together.
    DeviceMipLevel mips[NumLevels];
    for( uint32_t level = 0; level < NumLevels; ++level )
        mips[level] = texture.mipLevel( mipLevel + level );

    Footprint footprints[NumLevels];
    uint32_t  residenceWords[NumLevels];
    for( uint32_t level = 0; level < NumLevels; ++level )
        footprints[level] = makeFootprint<Footprint>( texture.info, mips[level], pageShift, x, y );
    for( uint32_t level = 0; level < NumLevels; ++level )
        residenceWords[level] = loadPageResidenceWord( context, footprints[level].pageId[0] );

    // Requests, for the missing level and for the pages of every footprint, are recorded only after every load of the
    // metadata: the compiler does not reorder loads across an atomic, and would read the metadata with per-lane loads
    // afterwards. Every level must be referenced, so the residence results are combined without short-circuiting.
    const bool fallback = requestedLevel < texture.info.cascadeLevel;
    if( fallback )
        requestCascade( context, texture.id, requestedLevel );

    bool pagesResident = true;
    for( uint32_t level = 0; level < NumLevels; ++level )
        pagesResident = referenceFootprint( context, footprints[level], residenceWords[level] ) && pagesResident;
    resident = pagesResident && !fallback;
    if( !pagesResident )
        return Sample{};

    if( texture.info.wideOffsets() )
        return filterLevels<Sample, Footprint, NumLevels, true>( context, texture.info, pageShift, mips, footprints, t );
    return filterLevels<Sample, Footprint, NumLevels, false>( context, texture.info, pageShift, mips, footprints, t );
}

template <class Sample, class Footprint>
HIP_DEMAND_INLINE Sample sampleLod( const DeviceContext& context, const TextureMetadata& texture, float x, float y, float lod, bool& resident )
{
    const uint32_t mipCount = texture.info.mipCount();
    lod                     = std::clamp( lod, 0.0f, static_cast<float>( mipCount - 1 ) );

    const bool pointMipmap = texture.info.mipmapFilterMode() == hipFilterModePoint;
    uint32_t   mipLevel    = static_cast<uint32_t>( floorf( pointMipmap ? lod + 0.5f : lod ) );
    float      t           = pointMipmap ? 0.0f : lod - static_cast<float>( mipLevel );

    // The finest level the filter reads is the one the texture must have pages for. Without them, the sample comes
    // from the finest level that has pages, as a fallback, and the loader is asked to extend the texture.
    const uint32_t requestedLevel = mipLevel;
    if( mipLevel < texture.info.cascadeLevel )
    {
        mipLevel = texture.info.cascadeLevel;
        t        = 0.0f;
    }

    // The next mip level cannot affect an exact-level sample or its residency.
    if( mipLevel + 1 >= mipCount || t == 0.0f )
        return sampleMipLevels<Sample, Footprint, 1>( context, texture, mipLevel, 0.0f, x, y, requestedLevel, resident );
    return sampleMipLevels<Sample, Footprint, 2>( context, texture, mipLevel, t, x, y, requestedLevel, resident );
}

template <class Sample>
HIP_DEMAND_INLINE Sample sampleTexture( const DeviceContext& context, const TextureMetadata& texture, float x, float y, float lod, bool& resident )
{
    if( texture.info.filterMode() == hipFilterModePoint )
        return sampleLod<Sample, PointFootprint>( context, texture, x, y, lod, resident );
    return sampleLod<Sample, BilinearFootprint>( context, texture, x, y, lod, resident );
}

// Samples one point of a texture without inlining, for the anisotropic loop of tex2DGrad(). Inlined into a loop with a
// runtime trip count, the sampler costs the kernel about 35 VGPRs more than in tail position, because the results of
// the format dispatch merge at the loop instead of returning; a call keeps that pressure out of the kernel, so only
// anisotropic samples pay for it.
template <class Sample>
__device__ __attribute__( ( noinline ) ) Sample
sampleTextureNoInline( const DeviceContext& context, TextureMetadata metadata, float x, float y, float lod, bool& resident )
{
    return sampleTexture<Sample>( context, metadata, x, y, lod, resident );
}

template <class Sample>
HIP_DEMAND_INLINE Sample
sampleTextureGrad( const DeviceContext& context, const TextureMetadata& metadata, float x, float y, float2 ddx, float2 ddy, bool& resident )
{
    // Footprint axes in level-0 texels; the caller's units are kept for stepping along the major axis.
    float2 axisX = ddx;
    float2 axisY = ddy;
    if( metadata.info.normalizedCoords() )
    {
        const float width  = static_cast<float>( metadata.info.width );
        const float height = static_cast<float>( metadata.info.height );
        axisX              = make_float2( ddx.x * width, ddx.y * height );
        axisY              = make_float2( ddy.x * width, ddy.y * height );
    }
    const float  lengthXSquared = axisX.x * axisX.x + axisX.y * axisX.y;
    const float  lengthYSquared = axisY.x * axisY.x + axisY.y * axisY.y;
    const bool   xIsMajor       = lengthXSquared >= lengthYSquared;
    const float  majorSquared   = xIsMajor ? lengthXSquared : lengthYSquared;
    const float  minorSquared   = xIsMajor ? lengthYSquared : lengthXSquared;
    const float2 majorStep      = xIsMajor ? ddx : ddy;

    // Number of samples: the axis ratio rounded up, capped by the texture. A footprint within one texel needs one.
    uint32_t samples = 1;
    if( metadata.info.maxAnisotropy() > 1 && majorSquared > 1.0f )
    {
        const float ratio = sqrtf( majorSquared / fmaxf( minorSquared, 1e-20f ) );
        samples           = std::clamp( static_cast<uint32_t>( ceilf( ratio ) ), 1u, metadata.info.maxAnisotropy() );
    }
    const float samplesScale = 1.0f / static_cast<float>( samples );
    const float lod          = 0.5f * log2f( fmaxf( majorSquared * samplesScale * samplesScale, 1.0f ) );

    if( samples == 1 )
        return sampleTexture<Sample>( context, metadata, x, y, lod, resident );

    resident = true;
    Sample sum{};
    for( uint32_t i = 0; i < samples; ++i )
    {
        const float t              = ( static_cast<float>( i ) + 0.5f ) * samplesScale - 0.5f;
        bool        sampleResident = false;
        sum += sampleTextureNoInline<Sample>( context, metadata, x + t * majorStep.x, y + t * majorStep.y, lod, sampleResident );
        resident = resident && sampleResident;
    }
    return sum * samplesScale;
}

HIP_DEMAND_INLINE void splitUdimCoordinate( float x, uint32_t cells, uint32_t addressMode, uint32_t& cell, float& local )
{
    const float extent = static_cast<float>( cells );
    switch( static_cast<hipTextureAddressMode>( addressMode ) )
    {
        case hipAddressModeWrap:
            x -= floorf( x / extent ) * extent;
            break;
        case hipAddressModeMirror:
        {
            const float period = 2.0f * extent;
            x -= floorf( x / period ) * period;
            x = x > extent ? period - x : x;
            break;
        }
        default:
            break;
    }
    x = std::clamp( x, 0.0f, extent );

    cell  = min( static_cast<uint32_t>( x ), cells - 1 );
    local = x - static_cast<float>( cell );
}

// ---------------------------------------------------------------------------------------------------------------------
// Public sampling API
// ---------------------------------------------------------------------------------------------------------------------

template <class Sample>
HIP_DEMAND_INLINE Sample tex2DLod( const DeviceContext& context, uint32_t textureId, float x, float y, float lod, bool& resident )
{
    // Invalid texture ids are reported as resident so that callers do not wait for them.
    resident = true;
    if( textureId >= context.textureInfos.len )
        return Sample{};

    const TextureMetadata metadata = loadTextureMetadata( context, textureId, resident );
    if( !resident )
        return Sample{};

    const Sample sample = sampleTexture<Sample>( context, metadata, x, y, lod, resident );
    referenceTextureInfo( context, textureId );
    return sample;
}

template <class Sample>
HIP_DEMAND_INLINE Sample tex2D( const DeviceContext& context, uint32_t textureId, float x, float y, bool& resident )
{
    return tex2DLod<Sample>( context, textureId, x, y, 0.0f, resident );
}

template <class Sample>
HIP_DEMAND_INLINE Sample tex2DGrad( const DeviceContext& context, uint32_t textureId, float x, float y, float2 ddx, float2 ddy, bool& resident )
{
    resident = true;
    if( textureId >= context.textureInfos.len )
        return Sample{};

    const TextureMetadata metadata = loadTextureMetadata( context, textureId, resident );
    if( !resident )
        return Sample{};

    const Sample sample = sampleTextureGrad<Sample>( context, metadata, x, y, ddx, ddy, resident );
    referenceTextureInfo( context, textureId );
    return sample;
}

template <class Sample>
HIP_DEMAND_INLINE Sample tex2DGradUdim( const DeviceContext& context,
                                        uint32_t             textureId,
                                        uint32_t             channel,
                                        float                x,
                                        float                y,
                                        float2               ddx,
                                        float2               ddy,
                                        bool&                resident )
{
    resident = true;
    if( textureId >= context.textureInfos.len )
        return Sample{};

    const TextureMetadata entry = loadTextureMetadata( context, textureId, resident );
    if( !resident )
        return Sample{};

    Sample sample{};
    if( !entry.info.isUdim() )
        sample = sampleTextureGrad<Sample>( context, entry, x, y, ddx, ddy, resident );
    else if( channel < entry.info.udimChannels() )
    {
        uint32_t u, v;
        float    localX, localY;
        splitUdimCoordinate( x, entry.info.udim(), entry.info.addressMode( 0 ), u, localX );
        splitUdimCoordinate( y, entry.info.vdim(), entry.info.addressMode( 1 ), v, localY );
        const uint32_t subTextureId =
            entry.info.udimStartTexture + ( v * entry.info.udim() + u ) * entry.info.udimChannels() + channel;

        const TextureMetadata subTexture = loadTextureMetadata( context, subTextureId, resident );
        if( resident && !subTexture.info.isEmpty() )
        {
            sample = sampleTextureGrad<Sample>( context, subTexture, localX, localY, ddx, ddy, resident );
            referenceTextureInfo( context, subTextureId );
        }
    }

    referenceTextureInfo( context, textureId );
    return sample;
}

}  // namespace hip_demand::vmm
