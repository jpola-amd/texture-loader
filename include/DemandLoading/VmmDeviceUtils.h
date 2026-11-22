#pragma once

#include "DeviceContext.h"


namespace hip_demand::vmm {

HIP_DEMAND_INLINE void getWordIdxAndBitIdx( uint32_t idx, uint32_t& wordIdx, uint32_t& bitIdx )
{
    wordIdx = idx >> 5;   // idx / 32
    bitIdx  = idx & 31u;  // idx % 32
}

HIP_DEMAND_INLINE void atomicSetBit(const DeviceSpan<uint32_t>& span, uint32_t index)
{
    uint32_t wordIdx = 0;
    uint32_t bitIdx  = 0;
    getWordIdxAndBitIdx( index, wordIdx, bitIdx );
    atomicOr( &span.ptr[wordIdx], 1u << bitIdx );
}

HIP_DEMAND_INLINE void setBit(const DeviceSpan<uint32_t>& span, uint32_t index)
{
    uint32_t wordIdx = 0;
    uint32_t bitIdx  = 0;
    getWordIdxAndBitIdx( index, wordIdx, bitIdx );
    span.ptr[wordIdx] |= 1u << bitIdx;
}

HIP_DEMAND_INLINE void atomicUnsetBit(const DeviceSpan<uint32_t>& span, uint32_t index)
{
    uint32_t wordIdx = 0;
    uint32_t bitIdx  = 0;
    getWordIdxAndBitIdx( index, wordIdx, bitIdx );
    atomicAnd( &span.ptr[wordIdx], ~( 1u << bitIdx ) );
}

HIP_DEMAND_INLINE void unsetBit(const DeviceSpan<uint32_t>& span, uint32_t index)
{
    uint32_t wordIdx = 0;
    uint32_t bitIdx  = 0;
    getWordIdxAndBitIdx( index, wordIdx, bitIdx );
    span.ptr[wordIdx] &= ~( 1u << bitIdx );
}

HIP_DEMAND_INLINE bool checkBitSet(const DeviceSpan<uint32_t>& span, uint32_t index)
{
    uint32_t wordIdx = 0;
    uint32_t bitIdx  = 0;
    getWordIdxAndBitIdx( index, wordIdx, bitIdx );
    return ( span.ptr[wordIdx] & ( 1u << bitIdx ) ) != 0;
}

HIP_DEMAND_INLINE uint32_t getUint4( const DeviceSpan<uint32_t>& words, const uint32_t index )
{
    const uint32_t wordIndex = index >> 3;
    return ( words.ptr[wordIndex] >> 4u * ( index & 0x7u ) ) & 0xf;
}

HIP_DEMAND_INLINE void clearUint4( const DeviceSpan<uint32_t>& words, const uint32_t index )
{
    const uint32_t wordIndex = index >> 3u;
    words.ptr[wordIndex] &= ~( 0xf << ( 4u * ( index & 0x7u ) ) );
}

HIP_DEMAND_INLINE void atomicClearUint4( const DeviceSpan<uint32_t>& words, const uint32_t index )
{
    const uint32_t wordIndex = index >> 3u;
    atomicAnd( &words.ptr[wordIndex], ~( 0xf << ( 4u * ( index & 0x7u ) ) ) );
}

HIP_DEMAND_INLINE void addUint4( const DeviceSpan<uint32_t>& words, const uint32_t index, const uint32_t val )
{
    const uint32_t wordIndex = index >> 3;
    words.ptr[wordIndex] += val << ( 4u * ( index & 0x7u ) );
}

}  // namespace hip_demand::vmm
