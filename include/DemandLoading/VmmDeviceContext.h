#pragma once

/// @file VmmDeviceContext.h
/// @brief Device context of the VMM demand loader, and the span type its buffers use.
/// @note Requires the HIP headers. DeviceContext.h, the context of the hardware-texture loader, does not: an
/// application may include that one where HIP's vector types cannot be declared.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

#if defined( __HIPCC__ )
#define HIP_DEMAND_INLINE __device__ __forceinline__
#else
#define HIP_DEMAND_INLINE inline
#endif

namespace hip_demand {

template <typename T>
using DevicePtr = T*;

template <typename T>
struct DeviceSpan
{
    DevicePtr<T> ptr = nullptr;
    size_t       len = 0;

    DeviceSpan() = default;

    explicit DeviceSpan( DevicePtr<T> ptr_, size_t len_ )
        : ptr( ptr_ )
        , len( len_ )
    {
    }

    HIP_DEMAND_INLINE size_t sizeInBytes() const { return len * sizeof( T ); }
    HIP_DEMAND_INLINE DeviceSpan<T> subspan( size_t offset, size_t newLen ) const
    {
        return DeviceSpan<T>{ ptr + offset, newLen };
    }
};

}  // namespace hip_demand

namespace hip_demand::vmm {

constexpr uint32_t INVALID_TEXTURE      = ~0u;
constexpr uint32_t INVALID_PAGE         = ~0u;
constexpr uint32_t MAX_TILED_MIP_LEVELS = 9;

struct DeviceMipLevel
{
    uint32_t width         = 0;
    uint32_t height        = 0;
    uint32_t tilesX        = 0;
    uint32_t tilesY        = 0;
    uint32_t startPage     = INVALID_PAGE;
    uint32_t mipTailOffset = 0;
    bool     mipTail       = false;

    HIP_DEMAND_INLINE uint32_t pageCount() const { return tilesX * tilesY; }
};

struct DeviceTextureInfo
{
    uint32_t width       = 0;
    uint32_t height      = 0;
    uint32_t sampling    = 0;
    uint32_t texel       = 0;
    uint32_t layout      = 0;
    uint32_t cascadeLevel     = 0;
    uint32_t mipTailPage      = INVALID_PAGE;
    uint32_t udimStartTexture = INVALID_TEXTURE;
    uint32_t udimGrid         = 0;

    // The first page of each tiled level. Tiled levels finer than cascadeLevel have no pages and their word is unset.
    // Tail levels have no word: they share mipTailPage at the offsets mipTailLevelOffset() computes.
    uint32_t mips[MAX_TILED_MIP_LEVELS]{};

    // --- sampling ---
    HIP_DEMAND_INLINE uint32_t addressMode( uint32_t axis ) const { return bits( sampling, 2 * axis, 2 ); }
    HIP_DEMAND_INLINE uint32_t filterMode() const { return bits( sampling, 4, 1 ); }
    HIP_DEMAND_INLINE uint32_t mipmapFilterMode() const { return bits( sampling, 5, 1 ); }
    HIP_DEMAND_INLINE bool     normalizedCoords() const { return bits( sampling, 6, 1 ) != 0; }
    // A mip level's pages span more than 4 GiB; see texelOffsetInLevel().
    HIP_DEMAND_INLINE bool     wideOffsets() const { return bits( sampling, 7, 1 ) != 0; }
    HIP_DEMAND_INLINE uint32_t maxAnisotropy() const { return bits( sampling, 8, 5 ); }  // 1..16; see tex2DGrad().
    HIP_DEMAND_INLINE bool isEmpty() const { return bits( sampling, 14, 1 ) != 0; }
    HIP_DEMAND_INLINE bool isLoaded() const { return bits( sampling, 15, 1 ) != 0; }

    HIP_DEMAND_INLINE void setAddressMode( uint32_t axis, uint32_t mode ) { setBits( sampling, 2 * axis, 2, mode ); }
    HIP_DEMAND_INLINE void setFilterMode( uint32_t mode ) { setBits( sampling, 4, 1, mode ); }
    HIP_DEMAND_INLINE void setMipmapFilterMode( uint32_t mode ) { setBits( sampling, 5, 1, mode ); }
    HIP_DEMAND_INLINE void setNormalizedCoords( bool normalized ) { setBits( sampling, 6, 1, normalized ? 1u : 0u ); }
    HIP_DEMAND_INLINE void setWideOffsets( bool wide ) { setBits( sampling, 7, 1, wide ? 1u : 0u ); }
    HIP_DEMAND_INLINE void setMaxAnisotropy( uint32_t samples ) { setBits( sampling, 8, 5, samples ); }
    HIP_DEMAND_INLINE void setEmpty( bool empty ) { setBits( sampling, 14, 1, empty ? 1u : 0u ); }
    HIP_DEMAND_INLINE void setLoaded( bool loaded ) { setBits( sampling, 15, 1, loaded ? 1u : 0u ); }

    // --- udim ---
    HIP_DEMAND_INLINE bool     isUdim() const { return udimStartTexture != INVALID_TEXTURE; }
    HIP_DEMAND_INLINE uint32_t udim() const { return bits( udimGrid, 0, 16 ); }
    HIP_DEMAND_INLINE uint32_t vdim() const { return bits( udimGrid, 16, 16 ); }
    HIP_DEMAND_INLINE uint32_t udimChannels() const { return bits( sampling, 16, 8 ); }  // Textures per grid cell.

    HIP_DEMAND_INLINE void setUdimGrid( uint32_t startTexture, uint32_t udim_, uint32_t vdim_, uint32_t channels )
    {
        udimStartTexture = startTexture;
        udimGrid         = 0;
        setBits( udimGrid, 0, 16, udim_ );
        setBits( udimGrid, 16, 16, vdim_ );
        setBits( sampling, 16, 8, channels );
    }

    // --- texel ---
    HIP_DEMAND_INLINE hipArray_Format format() const { return static_cast<hipArray_Format>( bits( texel, 0, 8 ) ); }
    HIP_DEMAND_INLINE uint32_t        numChannels() const { return bits( texel, 8, 8 ); }
    HIP_DEMAND_INLINE uint32_t        bytesPerTexel() const { return bits( texel, 16, 16 ); }

    HIP_DEMAND_INLINE void setTexel( hipArray_Format format_, uint32_t numChannels_, uint32_t bytesPerTexel_ )
    {
        texel = 0;
        setBits( texel, 0, 8, static_cast<uint32_t>( format_ ) );
        setBits( texel, 8, 8, numChannels_ );
        setBits( texel, 16, 16, bytesPerTexel_ );
    }

    // --- layout ---
    HIP_DEMAND_INLINE uint32_t tileWidthShift() const { return bits( layout, 0, 8 ); }  // Tiles are power-of-two sized.
    HIP_DEMAND_INLINE uint32_t tileHeightShift() const { return bits( layout, 8, 8 ); }
    HIP_DEMAND_INLINE uint32_t mipCount() const { return bits( layout, 16, 8 ); }
    HIP_DEMAND_INLINE uint32_t mipTailFirstLevel() const
    {
        return bits( layout, 24, 8 );
    }  // Equals mipCount() without a tail.

    HIP_DEMAND_INLINE void setLayout( uint32_t tileWidthShift_, uint32_t tileHeightShift_, uint32_t mipCount_, uint32_t mipTailFirstLevel_ )
    {
        layout = 0;
        setBits( layout, 0, 8, tileWidthShift_ );
        setBits( layout, 8, 8, tileHeightShift_ );
        setBits( layout, 16, 8, mipCount_ );
        setBits( layout, 24, 8, mipTailFirstLevel_ );
    }

    HIP_DEMAND_INLINE uint32_t tileWidth() const { return 1u << tileWidthShift(); }
    HIP_DEMAND_INLINE uint32_t tileHeight() const { return 1u << tileHeightShift(); }
    HIP_DEMAND_INLINE uint32_t tileWidthMask() const { return tileWidth() - 1; }
    HIP_DEMAND_INLINE uint32_t tileHeightMask() const { return tileHeight() - 1; }

    HIP_DEMAND_INLINE uint2 getTileCoords( uint32_t x, uint32_t y ) const
    {
        return { x >> tileWidthShift(), y >> tileHeightShift() };
    }

    HIP_DEMAND_INLINE uint2 getLocalCoords( uint32_t x, uint32_t y ) const
    {
        return { x & tileWidthMask(), y & tileHeightMask() };
    }

    HIP_DEMAND_INLINE uint32_t levelWidth( uint32_t level ) const { return std::max( width >> level, 1u ); }
    HIP_DEMAND_INLINE uint32_t levelHeight( uint32_t level ) const { return std::max( height >> level, 1u ); }

    HIP_DEMAND_INLINE uint32_t mipTailLevelOffset( uint32_t level ) const
    {
        uint32_t offset = 0;
        for( uint32_t tailLevel = mipTailFirstLevel(); tailLevel < level; ++tailLevel )
            offset += levelWidth( tailLevel ) * levelHeight( tailLevel ) * bytesPerTexel();
        return offset;
    }

    HIP_DEMAND_INLINE DeviceMipLevel getMipLevel( uint32_t level ) const
    {
        return getMipLevel( level, level < mipTailFirstLevel() ? mips[level] : INVALID_PAGE );
    }

    HIP_DEMAND_INLINE DeviceMipLevel getMipLevel( uint32_t level, uint32_t startPage ) const
    {
        DeviceMipLevel mip{};
        mip.mipTail   = level >= mipTailFirstLevel();
        mip.startPage = mip.mipTail ? mipTailPage : startPage;
        if( mip.mipTail )
            mip.mipTailOffset = mipTailLevelOffset( level );
        mip.width  = levelWidth( level );
        mip.height = levelHeight( level );
        mip.tilesX = ( mip.width >> tileWidthShift() ) + static_cast<uint32_t>( ( mip.width & tileWidthMask() ) != 0 );
        mip.tilesY = ( mip.height >> tileHeightShift() ) + static_cast<uint32_t>( ( mip.height & tileHeightMask() ) != 0 );
        return mip;
    }

  private:
    HIP_DEMAND_INLINE static uint32_t bits( uint32_t word, uint32_t shift, uint32_t count )
    {
        return ( word >> shift ) & ( ( 1u << count ) - 1 );
    }

    HIP_DEMAND_INLINE static void setBits( uint32_t& word, uint32_t shift, uint32_t count, uint32_t value )
    {
        const uint32_t mask = ( ( 1u << count ) - 1 ) << shift;
        word                = ( word & ~mask ) | ( ( value << shift ) & mask );
    }
};

struct ResourceTable
{
    struct Range
    {
        uint32_t start = 0;
        uint32_t count = 0;

        HIP_DEMAND_INLINE Range() {}
        HIP_DEMAND_INLINE Range( uint32_t start_, uint32_t count_ )
            : start( start_ )
            , count( count_ )
        {
        }

        HIP_DEMAND_INLINE bool     contains( uint32_t id ) const { return start <= id && id < end(); }
        HIP_DEMAND_INLINE uint32_t end() const { return start + count; }
        HIP_DEMAND_INLINE uint32_t getResourceId( uint32_t id ) const { return start + id; }
        HIP_DEMAND_INLINE uint32_t getLocalId( uint32_t id ) const { return id - start; }
    };

    Range textureTiles{};
    Range textureInfos{};
    // MAX_TILED_MIP_LEVELS per texture: a request for the texture to get pages down to that level.
    Range cascades{};

    HIP_DEMAND_INLINE uint32_t count() const { return textureTiles.count + textureInfos.count + cascades.count; }
};

enum class CounterIndex : uint32_t
{
    RequestedResources = 0,
    EvictionCandidates,
    NumCounters
};

struct EvictionCandidate
{
    uint32_t pageId = 0;
    uint32_t lru    = 0;
};

struct DeviceContext
{
    ResourceTable                 resourceTable{};
    DeviceSpan<uint8_t>           pageMemory{};
    DeviceSpan<uint32_t>          residenceBits{};
    DeviceSpan<DeviceTextureInfo> textureInfos{};
    DeviceSpan<uint32_t>          referenceBits{};
    DeviceSpan<uint32_t>          requestedResources{};
    DeviceSpan<EvictionCandidate> evictionCandidates{};
    DeviceSpan<uint32_t>          processedResources{};
    DeviceSpan<uint32_t>          lru{};
    DeviceSpan<uint32_t>          counters{};
    size_t                        pageSize          = 0;
    bool                          requestIfResident = false;
};

}  // namespace hip_demand::vmm
