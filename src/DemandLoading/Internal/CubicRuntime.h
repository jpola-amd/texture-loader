// SPDX-License-Identifier: MIT
#pragma once
#include "HipCalls.h"
#include "TextureRuntime.h"
#include <algorithm>

namespace hip_demand {
namespace internal {
inline contract_v1::AddressMode cubicAddress( hipTextureAddressMode mode )
{
    switch( mode )
    {
        case hipAddressModeWrap:
            return contract_v1::AddressMode::Wrap;
        case hipAddressModeMirror:
            return contract_v1::AddressMode::Mirror;
        case hipAddressModeBorder:
            return contract_v1::AddressMode::Border;
        default:
            return contract_v1::AddressMode::Clamp;
    }
}
inline contract_v1::SamplerDesc cubicDescriptor( const TextureDesc& desc, uint32_t a )
{
    contract_v1::SamplerDesc result;
    result.addressMode[0] = cubicAddress( desc.addressMode[0] );
    result.addressMode[1] = cubicAddress( desc.addressMode[1] );
    result.mipFilter =
        desc.mipmapFilterMode == hipFilterModePoint ? contract_v1::FilterMode::Point : contract_v1::FilterMode::Linear;
    result.maxAnisotropy    = a ? a : 1;
    result.normalizedCoords = desc.normalizedCoords;
    result.sRGB             = desc.sRGB;
    result.generateMipmaps  = desc.generateMipmaps;
    result.maxMipLevels     = std::min( desc.maxMipLevel, 32u );
    return result;
}
inline hipError_t createCubicPoints( const HipCalls&     calls,
                                     hipArray_t          array,
                                     hipMipmappedArray_t mipmap,
                                     uint32_t            levels,
                                     const TextureDesc&  desc,
                                     bool                floating,
                                     TextureObject ( &points )[32] )
{
    TextureDesc pointDesc      = desc;
    pointDesc.filterMode       = hipFilterModePoint;
    pointDesc.mipmapFilterMode = hipFilterModePoint;
    // Reconstruction converts raw byte RGB once at full precision. FLOAT
    // uploads have already been linearized by the existing source adapter.
    pointDesc.sRGB           = false;
    pointDesc.addressMode[0] = pointDesc.addressMode[1] = hipAddressModeClamp;
    const auto sampler                                  = makeSampler( pointDesc, floating, false, 1, 0 );
    for( uint32_t m = 0; m < levels; ++m )
    {
        hipResourceDesc resource{};
        resource.resType         = hipResourceTypeArray;
        resource.res.array.array = array;
        auto error               = hipSuccess;
        if( mipmap )
            error = calls.call( HipOperation::GetLevel,
                                [&] { return hipGetMipmappedArrayLevel( &resource.res.array.array, mipmap, m ); } );
        if( error != hipSuccess )
            return error;
        hipTextureObject_t object = 0;
        error                     = calls.call( HipOperation::CreateSampler,
                                                [&] { return hipCreateTextureObject( &object, &resource, &sampler, nullptr ); } );
        points[m]                 = reinterpret_cast<TextureObject>( object );
        if( error != hipSuccess )
            return error;
    }
    return hipSuccess;
}
inline hipError_t destroyCubicPoints( const HipCalls& calls, TextureObject ( &points )[32] )
{
    for( auto& point : points )
    {
        if( !point )
            continue;
        const auto error = calls.call( HipOperation::DestroySampler, [&] {
            return hipDestroyTextureObject( reinterpret_cast<hipTextureObject_t>( point ) );
        } );
        if( error != hipSuccess )
            return error;
        point = 0;
    }
    return hipSuccess;
}
}  // namespace internal
}  // namespace hip_demand
