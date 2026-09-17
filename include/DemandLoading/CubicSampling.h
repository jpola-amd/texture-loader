// SPDX-License-Identifier: MIT
#pragma once
// Include hip/hip_runtime.h first and compile with HIP. See CUBIC.md.
#include "DemandLoading/CubicContext.h"
#include "DemandLoading/WholeMipSampling.h"

namespace hip_demand {
namespace cubic_v1 {
namespace detail {
using contract_v1::AddressMode;
using contract_v1::evaluate;
using contract_v1::FilterMode;
using contract_v1::GpuKey;
using contract_v1::InvalidSlot;
using contract_v1::mipDimension;
using contract_v1::MipRange;
using contract_v1::Outcome;
using contract_v1::RegistrationState;
using contract_v1::SampleDecision;
using contract_v1::SamplingPolicy;
using contract_v1::valid;
using contract_v1::validate;

__device__ inline bool finite( float x )
{
    return x >= -3.402823466e38F && x <= 3.402823466e38F;
}
__device__ inline Outcome validateConfig( const Config& config )
{
    if( config.abi.version != Version || config.abi.byteSize != sizeof( Config ) )
        return Outcome::AbiMismatch;
    if( config.spatial > SpatialMode::SmartBicubic || !finite( config.maxAnisotropy ) || config.maxAnisotropy < 1
        || config.conservative > 1 || config.reserved )
        return Outcome::InvalidInput;
    return Outcome::Success;
}
__device__ inline Outcome lookup( const DeviceContext& ctx, GpuKey key, uint64_t revision, const Entry*& entry )
{
    if( ctx.abi.version != Version || ctx.abi.byteSize != sizeof( DeviceContext ) )
        return Outcome::AbiMismatch;
    if( !ctx.incarnation || !ctx.entries || !ctx.count || ctx.backing > Backing::WholeMip )
        return Outcome::InvalidInput;
    if( !valid( key ) || key.incarnation != ctx.incarnation || key.slot >= ctx.count )
        return Outcome::InvalidKey;
    if( ctx.backing == Backing::Legacy )
    {
        const auto& legacy = ctx.legacy;
        if( !legacy.textures || !legacy.residentFlags || !legacy.requests || !legacy.requestCount
            || !legacy.requestOverflow || legacy.maxTextures < ctx.count || !legacy.maxRequests )
            return Outcome::InvalidInput;
    }
    else
    {
        const whole_mip_v1::Entry* original = nullptr;
        const auto                 result   = whole_mip_v1::detail::lookup( ctx.whole, key, revision, original );
        if( result != Outcome::Success )
            return result;
        if( ctx.count != ctx.whole.numSamplers || ctx.incarnation != ctx.whole.incarnation )
            return Outcome::InvalidInput;
    }
    const auto& e = ctx.entries[key.slot];
    if( !( e.texture.key == key ) || !revision || e.texture.revision != revision || e.texture.state != RegistrationState::Live )
        return Outcome::InvalidKey;
    const Outcome descriptor = validate( e.descriptor );
    if( descriptor != Outcome::Success )
        return descriptor;
    if( e.texture.reserved || e.reserved || e.pointSRGB > 1 || !e.descriptor.normalizedCoords )
        return Outcome::InvalidInput;
    // A failed lazy read can have no dimensions. Preserve its terminal status
    // without planning a footprint or issuing another ordinary image request.
    if( e.texture.residency != Outcome::Success && e.texture.residency != Outcome::Pending && e.texture.residency != Outcome::Deferred )
        return e.texture.residency;
    const auto& mips = e.texture.mips;
    if( e.texture.residency == Outcome::Deferred && !mips.originalLevels )
    {
        if( ctx.backing != Backing::Legacy || mips.originalWidth || mips.originalHeight
            || mips.firstResidentMip != InvalidSlot || mips.resourceWidth || mips.resourceHeight || mips.resourceLevels
            || e.texture.textureObject || ctx.legacy.textures[key.slot] || isTextureResident( ctx.legacy, key.slot ) )
            return Outcome::InvalidInput;
        for( auto point : e.points )
            if( point )
                return Outcome::InvalidInput;
        entry = &e;
        return Outcome::Success;
    }
    if( validate( mips ) != Outcome::Success )
        return Outcome::InvalidInput;
    uint32_t levels = e.descriptor.mipPolicy == contract_v1::MipPolicy::Disabled ?
                          1 :
                          contract_v1::fullMipCount( e.texture.mips.originalWidth, e.texture.mips.originalHeight );
    if( e.descriptor.maxMipLevels && levels > e.descriptor.maxMipLevels )
        levels = e.descriptor.maxMipLevels;
    if( levels != e.texture.mips.originalLevels )
        return Outcome::Unsupported;
    const bool resident = e.texture.mips.firstResidentMip != InvalidSlot;
    if( resident != ( e.texture.residency == Outcome::Success ) || resident != ( e.texture.textureObject != 0 ) )
        return Outcome::InvalidTransition;
    if( ctx.backing == Backing::Legacy && resident
        && ( !isTextureResident( ctx.legacy, key.slot ) || ctx.legacy.textures[key.slot] != e.texture.textureObject ) )
        return Outcome::InvalidTransition;
    if( ctx.backing == Backing::WholeMip && ctx.whole.entries[key.slot].texture.textureObject != e.texture.textureObject )
        return Outcome::InvalidTransition;
    entry = &e;
    return Outcome::Success;
}

struct Plan
{
    double   lod = 0, fraction = 0, cubic = 1;
    uint32_t low = 0, high = 0;
    float2   dx{}, dy{};
    MipRange required{};
};

__device__ inline Plan plan( const Entry& e, const Config& config, double lod, bool gradients, float2 dx, float2 dy, bool derivative )
{
    Plan        p;
    const auto& m = e.texture.mips;
    if( gradients )
    {
        double       x0 = dx.x, x1 = dx.y, y0 = dy.x, y1 = dy.y;
        const double rho =
            fmax( hypot( x0 * m.originalWidth, x1 * m.originalHeight ), hypot( y0 * m.originalWidth, y1 * m.originalHeight ) );
        if( config.spatial == SpatialMode::SmartBicubic )
            p.cubic = fmax( 0., fmin( 1., 2 - rho ) );
        const double q  = .99 / double( m.originalWidth > m.originalHeight ? m.originalWidth : m.originalHeight );
        double       nx = hypot( x0, x1 ), ny = hypot( y0, y1 );
        if( nx == 0 )
        {
            x0 = q;
            nx = q;
        }
        if( ny == 0 )
        {
            y1 = q;
            ny = q;
        }
        if( nx < q )
        {
            x0 *= q / nx;
            x1 *= q / nx;
            nx = q;
        }
        if( ny < q )
        {
            y0 *= q / ny;
            y1 *= q / ny;
            ny = q;
        }
        if( !config.conservative )
        {
            const double sx = fmin( 1., 16 * ny / nx ), sy = fmin( 1., 16 * nx / ny );
            x0 *= sx;
            x1 *= sx;
            y0 *= sy;
            y1 *= sy;
        }
        p.dx = make_float2( float( x0 ), float( x1 ) );
        p.dy = make_float2( float( y0 ), float( y1 ) );
        x0 *= m.originalWidth;
        x1 *= m.originalHeight;
        y0 *= m.originalWidth;
        y1 *= m.originalHeight;
        const double u = y0 * y0 + y1 * y1, v = x0 * x0 + x1 * x1, dot = x0 * y0 + x1 * y1;
        const double major       = ( u + v + hypot( u - v, 2 * dot ) ) * .5;
        const double determinant = x0 * y1 - x1 * y0;
        const double minor       = major ? determinant * determinant / major : 0;
        const double a           = config.maxAnisotropy;
        lod                      = .5 * log2( fmax( minor, major / ( a * a ) ) );
        if( config.spatial == SpatialMode::SmartBicubic && !derivative )
            lod = 0;
    }
    p.lod = fmax( 0., fmin( double( m.originalLevels - 1 ), lod ) );
    if( e.descriptor.mipFilter == FilterMode::Point )
        p.lod = fmax( 0., ceil( p.lod - .5 ) );
    p.low  = uint32_t( p.lod );
    p.high = p.low;
    if( config.spatial == SpatialMode::Bicubic && e.descriptor.mipFilter == FilterMode::Linear && p.lod > p.low )
    {
        p.high     = p.low + 1;
        p.fraction = p.lod - p.low;
    }
    p.required = p.cubic < 1 ? MipRange{ 0, m.originalLevels - 1 } : MipRange{ p.low, p.high };
    return p;
}

__device__ inline bool address( double index, uint32_t size, AddressMode mode, uint32_t& result )
{
    if( mode == AddressMode::Border && ( index < 0 || index >= size ) )
        return false;
    if( mode == AddressMode::Clamp || mode == AddressMode::Border )
    {
        result = uint32_t( fmax( 0., fmin( double( size - 1 ), index ) ) );
        return true;
    }
    const double period  = mode == AddressMode::Mirror ? 2. * size : double( size );
    double       wrapped = fmod( index, period );
    if( wrapped < 0 )
        wrapped += period;
    if( mode == AddressMode::Mirror && wrapped >= size )
        wrapped = period - 1 - wrapped;
    result = uint32_t( wrapped );
    return true;
}
__device__ inline void weights( double a, double* w, double* d )
{
    const double b = 1 - a, aa = a * a;
    w[0] = b * b * b / 6;
    w[1] = ( 4 + aa * ( 3 * a - 6 ) ) / 6;
    w[2] = ( 1 + a * ( 3 + a * ( 3 - 3 * a ) ) ) / 6;
    w[3] = aa * a / 6;
    d[0] = -.5 * b * b;
    d[1] = a * ( 1.5 * a - 2 );
    d[2] = .5 + a - 1.5 * aa;
    d[3] = .5 * aa;
}
struct Values
{
    double value[4]{}, ds[4]{}, dt[4]{};
};
__device__ inline Values reconstruct( const Entry& entry, uint32_t level, double s, double t )
{
    Values         result;
    const auto&    m = entry.texture.mips;
    const uint32_t w = mipDimension( m.originalWidth, level ), h = mipDimension( m.originalHeight, level );
    const double   x = s * w - .5, y = t * h - .5, i = floor( x ), j = floor( y );
    double         wx[4], wy[4], dx[4], dy[4];
    weights( x - i, wx, dx );
    weights( y - j, wy, dy );
    const auto point = reinterpret_cast<hipTextureObject_t>( entry.points[level - m.firstResidentMip] );
    for( unsigned b = 0; b < 4; ++b )
        for( unsigned a = 0; a < 4; ++a )
        {
            uint32_t xx, yy;
            if( !address( i + a - 1, w, entry.descriptor.addressMode[0], xx )
                || !address( j + b - 1, h, entry.descriptor.addressMode[1], yy ) )
                continue;
            const float4 fetched = ::tex2D<float4>( point, float( ( xx + .5 ) / w ), float( ( yy + .5 ) / h ) );
            double       channels[4]{ fetched.x, fetched.y, fetched.z, fetched.w };
            if( entry.pointSRGB )
                for( unsigned k = 0; k < 3; ++k )
                    channels[k] = channels[k] <= .04045 ? channels[k] / 12.92 : pow( ( channels[k] + .055 ) / 1.055, 2.4 );
            for( unsigned k = 0; k < 4; ++k )
            {
                result.value[k] += channels[k] * wx[a] * wy[b];
                result.ds[k] += channels[k] * dx[a] * wy[b] * w;
                result.dt[k] += channels[k] * wx[a] * dy[b] * h;
            }
        }
    return result;
}
__device__ inline float4 native( const Entry& entry, const Plan& p, double s, double t )
{
    auto         dx = p.dx, dy = p.dy;
    const auto&  m     = entry.texture.mips;
    const double scale = exp2( double( m.firstResidentMip ) );
    const float  sx    = float( m.originalWidth / ( m.resourceWidth * scale ) );
    const float  sy    = float( m.originalHeight / ( m.resourceHeight * scale ) );
    dx.x *= sx;
    dy.x *= sx;
    dx.y *= sy;
    dy.y *= sy;
    return ::tex2DGrad<float4>( reinterpret_cast<hipTextureObject_t>( entry.texture.textureObject ), float( s ),
                                float( t ), dx, dy );
}
__device__ inline void components( float4 p, double* c )
{
    c[0] = p.x;
    c[1] = p.y;
    c[2] = p.z;
    c[3] = p.w;
}
__device__ inline float4 rgba( const double* c )
{
    return make_float4( float( c[0] ), float( c[1] ), float( c[2] ), float( c[3] ) );
}

__device__ inline SampleDecision sample( const DeviceContext& ctx,
                                         GpuKey               key,
                                         uint64_t             revision,
                                         const Config&        config,
                                         float                s,
                                         float                t,
                                         float                lod,
                                         bool                 gradients,
                                         float2               dx,
                                         float2               dy,
                                         float4*              value,
                                         float4*              ds,
                                         float4*              dt,
                                         float2               jitter )
{
    if( value )
        *value = make_float4( 0, 0, 0, 0 );
    if( ds )
        *ds = make_float4( 0, 0, 0, 0 );
    if( dt )
        *dt = make_float4( 0, 0, 0, 0 );
    auto outcome = validateConfig( config );
    if( outcome != Outcome::Success )
        return { outcome };
    if( !finite( s ) || !finite( t ) || !finite( lod ) || !finite( dx.x ) || !finite( dx.y ) || !finite( dy.x ) || !finite( dy.y )
        || !finite( jitter.x ) || !finite( jitter.y ) || ( !gradients && config.spatial == SpatialMode::SmartBicubic ) )
        return { Outcome::InvalidInput };
    const Entry* entry = nullptr;
    outcome            = lookup( ctx, key, revision, entry );
    if( outcome != Outcome::Success )
        return { outcome };
    if( config.spatial == SpatialMode::SmartBicubic && config.maxAnisotropy != float( entry->descriptor.maxAnisotropy ) )
        return { Outcome::InvalidInput };
    if( entry->texture.residency == Outcome::Deferred && !entry->texture.mips.originalLevels )
    {
        recordTextureRequest( ctx.legacy, key.slot );
        const auto status = __atomic_load_n( ctx.legacy.requestOverflow, __ATOMIC_RELAXED ) ? Outcome::RequestOverflow :
                                                                                              Outcome::Pending;
        return { status, contract_v1::SampleValidity::Missing };
    }
    const auto p = plan( *entry, config, lod, gradients, dx, dy, ds || dt );
    auto decision = evaluate( entry->texture.mips, { Outcome::Success, p.required, float( p.lod ) }, SamplingPolicy::Strict );
    if( decision.needsRequest() )
    {
        if( ctx.backing == Backing::WholeMip )
            whole_mip_v1::detail::demand( ctx.whole, key, revision, decision );
        else
        {
            recordTextureRequest( ctx.legacy, key.slot );
            if( __atomic_load_n( ctx.legacy.requestOverflow, __ATOMIC_RELAXED ) )
                decision.outcome = Outcome::RequestOverflow;
        }
    }
    if( !decision.contributesStrictSample() )
        return decision;
    if( p.cubic > 0 )
    {
        if( !entry->points[p.low - entry->texture.mips.firstResidentMip] || !entry->points[p.high - entry->texture.mips.firstResidentMip] )
            return { Outcome::InvalidTransition };
    }
    if( !value && !ds && !dt )
        return decision;
    const double u = double( s ) + double( jitter.x ) / entry->texture.mips.originalWidth;
    const double v = double( t ) + double( jitter.y ) / entry->texture.mips.originalHeight;
    Values       result;
    if( p.cubic > 0 )
    {
        result = reconstruct( *entry, p.low, u, v );
        if( p.high != p.low )
        {
            const auto next = reconstruct( *entry, p.high, u, v );
            for( int k = 0; k < 4; ++k )
            {
                result.value[k] += p.fraction * ( next.value[k] - result.value[k] );
                result.ds[k] += p.fraction * ( next.ds[k] - result.ds[k] );
                result.dt[k] += p.fraction * ( next.dt[k] - result.dt[k] );
            }
        }
    }
    if( p.cubic < 1 )
    {
        Values n;
        if( value )
            components( native( *entry, p, u, v ), n.value );
        if( ds || dt )
        {
            const uint32_t w = mipDimension( entry->texture.mips.originalWidth, p.low );
            const uint32_t h = mipDimension( entry->texture.mips.originalHeight, p.low );
            const double   x = u * w - .5, y = v * h - .5, i = floor( x ), j = floor( y );
            const double   ii = i + ( x - i > .5 ? .5 : 0 ), jj = j + ( y - j > .5 ? .5 : 0 );
            double         a[4], b[4], c[4], d[4];
            components( native( *entry, p, ( ii + .5 ) / w, ( jj + .5 ) / h ), a );
            components( native( *entry, p, ( ii + 1 ) / w, ( jj + .5 ) / h ), b );
            components( native( *entry, p, ( ii + .5 ) / w, ( jj + 1 ) / h ), c );
            components( native( *entry, p, ( ii + 1 ) / w, ( jj + 1 ) / h ), d );
            for( int k = 0; k < 4; ++k )
            {
                n.ds[k] = 2 * w * ( ( b[k] - a[k] ) + ( 2 * ( y - jj ) ) * ( ( d[k] - c[k] ) - ( b[k] - a[k] ) ) );
                n.dt[k] = 2 * h * ( ( c[k] - a[k] ) + ( 2 * ( x - ii ) ) * ( ( d[k] - b[k] ) - ( c[k] - a[k] ) ) );
            }
        }
        for( int k = 0; k < 4; ++k )
        {
            result.value[k] = p.cubic * result.value[k] + ( 1 - p.cubic ) * n.value[k];
            result.ds[k]    = p.cubic * result.ds[k] + ( 1 - p.cubic ) * n.ds[k];
            result.dt[k]    = p.cubic * result.dt[k] + ( 1 - p.cubic ) * n.dt[k];
        }
    }
    if( value )
        *value = rgba( result.value );
    if( ds )
        *ds = rgba( result.ds );
    if( dt )
        *dt = rgba( result.dt );
    return decision;
}
}  // namespace detail

__device__ inline contract_v1::SampleDecision tex2DLod( const DeviceContext& context,
                                                        contract_v1::GpuKey  key,
                                                        uint64_t             revision,
                                                        const Config&        config,
                                                        float                s,
                                                        float                t,
                                                        float                lod,
                                                        float4*              value,
                                                        float4*              ds     = nullptr,
                                                        float4*              dt     = nullptr,
                                                        float2               jitter = {} )
{
    return detail::sample( context, key, revision, config, s, t, lod, false, {}, {}, value, ds, dt, jitter );
}
__device__ inline contract_v1::SampleDecision tex2DGrad( const DeviceContext& context,
                                                         contract_v1::GpuKey  key,
                                                         uint64_t             revision,
                                                         const Config&        config,
                                                         float                s,
                                                         float                t,
                                                         float2               dx,
                                                         float2               dy,
                                                         float4*              value,
                                                         float4*              ds     = nullptr,
                                                         float4*              dt     = nullptr,
                                                         float2               jitter = {} )
{
    return detail::sample( context, key, revision, config, s, t, 0, true, dx, dy, value, ds, dt, jitter );
}
__device__ inline contract_v1::SampleDecision tex2D( const DeviceContext& context,
                                                     contract_v1::GpuKey  key,
                                                     uint64_t             revision,
                                                     const Config&        config,
                                                     float                s,
                                                     float                t,
                                                     float4*              value,
                                                     float4*              ds = nullptr,
                                                     float4*              dt = nullptr )
{
    return tex2DLod( context, key, revision, config, s, t, 0, value, ds, dt );
}
}  // namespace cubic_v1
}  // namespace hip_demand
