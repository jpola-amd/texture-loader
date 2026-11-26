#pragma once

/// @file LinearMipFootprint.h
/// @brief The mip level and the anisotropic taps of a gradient lookup, shared by the samplers that blend mip
/// levels in software: tex2DGradLinearMip() of TextureSampling.h and of VmmTextureSampling.h.

#include <hip/hip_runtime.h>

#include "ShaderUtil/TextureUtil.h"

namespace hip_demand {
namespace detail {

struct LinearMipFootprint
{
    float  mipLevel;  // Fractional level within [0, numMipLevels - 1]; whole for point mip filtering.
    int    taps;      // Samples to average, spread along `axis` and centered on the lookup.
    float2 axis;      // The longer gradient, in the units of the lookup's coordinates.
};

// Selects the mip level from the footprint. For an anisotropic footprint the level only covers the minor axis,
// so, as the hardware does with maxAnisotropy > 1, several taps are spread along the major axis. One would alias.
__device__ __forceinline__ LinearMipFootprint
selectLinearMipFootprint( int width, int height, int numMipLevels, bool pointMips, float2 ddx, float2 ddy, float maxAnisotropy )
{
    LinearMipFootprint footprint;
    float              ml = getMipLevel( ddx, ddy, width, height, 1.0f / maxAnisotropy );
    if( pointMips )
        ml = ceilf( ml - 0.5f );
    ml                 = fminf( fmaxf( ml, 0.0f ), (float)( numMipLevels - 1 ) );
    footprint.mipLevel = ml;

    // Taps: the major to minor ratio of the footprint ellipse, rounded and at most maxAnisotropy.
    // Rounding (not ceil) matches the CUDA hardware best on Arnold's testsuite: a mildly
    // anisotropic footprint keeps one tap. One tap when the footprint is within a texel, or at the
    // coarsest level of a mip chain, where the taps would read the same texels. A texture without
    // mips counts as that level: one sample is what matches Arnold's CUDA references.
    const float2 tx     = make_float2( ddx.x * width, ddx.y * height );
    const float2 ty     = make_float2( ddy.x * width, ddy.y * height );
    const float  A      = ty.x * ty.x + ty.y * ty.y;
    const float  B      = -2.0f * ( tx.x * ty.x + tx.y * ty.y );
    const float  C      = tx.x * tx.x + tx.y * tx.y;
    const float  root   = sqrtf( fmaxf( ( A - C ) * ( A - C ) + B * B, 0.0f ) );
    const float  major2 = ( A + C + root ) * 0.5f;
    const float  minor2 = ( A + C - root ) * 0.5f;
    footprint.taps      = 1;
    const bool coarsest = ml >= (float)( numMipLevels - 1 );
    if( major2 > 1.0f && !coarsest )
    {
        const float ratio = minor2 > 0.0f ? sqrtf( major2 / minor2 ) : maxAnisotropy;
        footprint.taps    = (int)fminf( fmaxf( floorf( ratio + 0.5f ), 1.0f ), fmaxf( maxAnisotropy, 1.0f ) );
    }
    // Spread the taps along the longer gradient
    footprint.axis = C >= A ? ddx : ddy;
    return footprint;
}

}  // namespace detail
}  // namespace hip_demand
