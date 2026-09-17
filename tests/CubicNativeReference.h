// SPDX-License-Identifier: MIT
#pragma once
#include "CubicReference.h"

namespace cubic_reference {
inline Pixel bilinear( const Level& l, double s, double t )
{
    const double x = s * l.width - .5, y = t * l.height - .5;
    const int    ix = int( std::floor( x ) ), iy = int( std::floor( y ) );
    const double fx = x - ix, fy = y - iy;
    Pixel        result{};
    for( int j = 0; j < 2; ++j )
        for( int i = 0; i < 2; ++i )
        {
            const auto& p = l.pixels[address( iy + j, l.height, Clamp ) * l.width + address( ix + i, l.width, Clamp )];
            for( int k = 0; k < 4; ++k )
                result[k] += p[k] * ( i ? fx : 1 - fx ) * ( j ? fy : 1 - fy );
        }
    return result;
}
inline Pixel native( const std::vector<Level>& levels, double s, double t, double lod, bool linear )
{
    lod = std::clamp( lod, 0., double( levels.size() - 1 ) );
    if( !linear )
        lod = std::floor( lod + .5 );
    const unsigned low    = unsigned( lod );
    auto           result = bilinear( levels[low], s, t );
    if( linear && lod > low )
    {
        const auto next = bilinear( levels[low + 1], s, t );
        for( int k = 0; k < 4; ++k )
            result[k] += ( lod - low ) * ( next[k] - result[k] );
    }
    return result;
}
struct Smart
{
    Result                               expected{};
    Pixel                                dsBound{}, dtBound{};
    std::array<std::array<double, 2>, 5> nativeCoordinates{};
    std::array<Pixel, 5>                 nativeValues{};
};
inline Smart smart( const std::vector<Level>& levels, double s, double t, Footprint f, bool linear, bool derivative )
{
    double lod = derivative ? std::clamp( f.lod, 0., double( levels.size() - 1 ) ) : 0;
    if( !linear )
        lod = std::ceil( lod - .5 );
    const unsigned low   = unsigned( std::max( lod, 0. ) );
    auto           cubic = reconstruct( levels[low], s, t, Clamp, Clamp );
    const double   w = levels[low].width, h = levels[low].height;
    const double   x = s * w - .5, y = t * h - .5, ix = std::floor( x ), iy = std::floor( y );
    const double   i = ix + ( x - ix > .5 ? .5 : 0 ), j = iy + ( y - iy > .5 ? .5 : 0 );
    Smart          result;
    result.nativeCoordinates = { { { s, t },
                                   { ( i + .5 ) / w, ( j + .5 ) / h },
                                   { ( i + 1 ) / w, ( j + .5 ) / h },
                                   { ( i + .5 ) / w, ( j + 1 ) / h },
                                   { ( i + 1 ) / w, ( j + 1 ) / h } } };
    for( unsigned n = 0; n < 5; ++n )
        result.nativeValues[n] = native( levels, result.nativeCoordinates[n][0], result.nativeCoordinates[n][1], f.lod, linear );
    for( unsigned k = 0; k < 4; ++k )
    {
        const double a = result.nativeValues[1][k], b = result.nativeValues[2][k];
        const double c = result.nativeValues[3][k], d = result.nativeValues[4][k];
        const double u = 2 * ( x - i ), v = 2 * ( y - j );
        const double ns          = 2 * w * ( ( 1 - v ) * ( b - a ) + v * ( d - c ) );
        const double nt          = 2 * h * ( ( 1 - u ) * ( c - a ) + u * ( d - b ) );
        result.expected.value[k] = f.weight * cubic.value[k] + ( 1 - f.weight ) * result.nativeValues[0][k];
        result.expected.ds[k]    = f.weight * cubic.ds[k] + ( 1 - f.weight ) * ns;
        result.expected.dt[k]    = f.weight * cubic.dt[k] + ( 1 - f.weight ) * nt;
        result.dsBound[k]        = f.weight * derivativeTolerance( cubic.ds[k] / levels[0].width ) * levels[0].width
                            + ( 1 - f.weight ) * 2 * w
                                  * ( std::abs( 1 - v ) * ( valueTolerance( a ) + valueTolerance( b ) )
                                      + std::abs( v ) * ( valueTolerance( c ) + valueTolerance( d ) ) );
        result.dtBound[k] = f.weight * derivativeTolerance( cubic.dt[k] / levels[0].height ) * levels[0].height
                            + ( 1 - f.weight ) * 2 * h
                                  * ( std::abs( 1 - u ) * ( valueTolerance( a ) + valueTolerance( c ) )
                                      + std::abs( u ) * ( valueTolerance( b ) + valueTolerance( d ) ) );
    }
    return result;
}
}  // namespace cubic_reference
